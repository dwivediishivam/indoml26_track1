"""Noise-event and clean-speech sources for the synthetic mixer.

Noise sources hold a lazily built list of NoiseItem with sampling probabilities:
    src.pick(rng) -> NoiseItem          src.load(item) -> float32[item.n]   (read-only; callers copy)
Speech sources return clean speech:
    src.sample(rng, n) -> float32[n]    n=None returns one natural-length clip/region.
All audio is float32 mono at contract.SR. Tags are normalised with norm_tag ("<horn>" -> "horn").

Staged training data (Phase 2 stage_data.py) on the cluster lives at ~/indoml-work/data:
    READY, manifest.jsonl (one row per clip: offset, samples, tier, spans=[[on, off, category, tag]]), audio.f16.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from common.contract import CATEGORIES, SR, VAL_DIR, WORK, clip_id, folds

STAGED_DIR = Path("~/indoml-work/data").expanduser()
CACHE_DIR = WORK / "synthgen"
AUDIO_EXTS = (".wav", ".flac", ".ogg")


def norm_tag(tag) -> str:
    return str(tag or "").strip().strip("<>").strip().lower()


def ids_in_folds(fold_ids) -> set:
    """Validation clip ids whose fold (work/folds.json) is in fold_ids, e.g. held-out CV folds."""
    fold_ids = set(fold_ids)
    return {k for k, f in folds().items() if f in fold_ids} if fold_ids else set()


def to_mono_sr(wav, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR)
        wav = resample_poly(wav, SR // g, int(sr) // g).astype(np.float32)
    return wav


def read_audio(path, start: int = 0, stop: int | None = None) -> np.ndarray:
    """start/stop are in file frames; output is resampled to SR."""
    wav, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=False)
    return to_mono_sr(wav, sr)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _cached(cache, kind: str, sig: dict, build):
    """Load build() output from a JSON cache keyed by sig; rebuild (atomically, worker-safe) on mismatch."""
    path = Path(cache) if cache else CACHE_DIR / f"{kind}_{hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:10]}.json"
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("sig") == sig:
            return d["items"]
    items = build()
    _atomic_json(path, {"sig": sig, "items": items})
    return items


class StagedBlob:
    """Read-only view of the staged training data; the memmap is opened lazily per process."""

    def __init__(self, data_dir=STAGED_DIR):
        self.dir = Path(data_dir).expanduser()
        if not (self.dir / "READY").exists():
            raise FileNotFoundError(f"Staged data is not ready: {self.dir / 'READY'}")
        self._blob = None

    def rows(self):
        with (self.dir / "manifest.jsonl").open() as handle:
            for i, line in enumerate(handle):
                yield i, json.loads(line)

    def read(self, offset: int, n: int) -> np.ndarray:
        if self._blob is None:
            self._blob = np.memmap(self.dir / "audio.f16", dtype=np.float16, mode="r")
        return np.asarray(self._blob[offset:offset + n], dtype=np.float32)

    def signature(self, **params) -> dict:
        st = (self.dir / "manifest.jsonl").stat()
        return dict(manifest=str(self.dir / "manifest.jsonl"), size=st.st_size, mtime=int(st.st_mtime), **params)

    def __getstate__(self):  # never pickle the memmap into DataLoader workers
        return {**self.__dict__, "_blob": None}


# ----------------------------------------------------------------------------------------------- noise sources

@dataclass(frozen=True)
class NoiseItem:
    sid: str
    category: str
    tag: str
    n: int            # samples at SR
    ref: tuple = ()   # source-specific locator

    @property
    def duration(self) -> float:
        return self.n / SR


class NoiseSource:
    """Base class. Subclasses implement _build() -> (items, base_weights) and load(item).

    tag_prior: optional {tag: frequency}; per-item weights are rescaled so the tag mix follows it
               (tags absent from the prior get the smallest prior value).
    crop_prob: per-source override of cfg['crop']['prob'] (None = use the mixer config).
    """

    def __init__(self, tag_prior: dict | None = None, uniform: bool = False, crop_prob: float | None = None):
        self.tag_prior, self.uniform, self.crop_prob = tag_prior, uniform, crop_prob
        self._items, self._cdf = None, None

    @property
    def items(self) -> list:
        if self._items is None:
            items, base = self._build()
            if not items:
                raise ValueError(f"{type(self).__name__}: no usable noise items")
            w = np.ones(len(items)) if self.uniform else np.asarray(base, dtype=np.float64)
            if self.tag_prior:
                prior = {norm_tag(k): float(v) for k, v in self.tag_prior.items()}
                floor = min(prior.values())
                mass = defaultdict(float)
                for it, wi in zip(items, w):
                    mass[it.tag] += wi
                w = np.array([wi * prior.get(it.tag, floor) / mass[it.tag] for it, wi in zip(items, w)])
            self._items, self._cdf = items, np.cumsum(w / w.sum())
        return self._items

    def __len__(self):
        return len(self.items)

    def pick(self, rng: np.random.Generator) -> NoiseItem:
        items = self.items
        return items[min(int(np.searchsorted(self._cdf, rng.random(), side="right")), len(items) - 1)]

    def _build(self):
        raise NotImplementedError

    def load(self, item: NoiseItem) -> np.ndarray:
        raise NotImplementedError


class BankSource(NoiseSource):
    """Organiser snippet bank written by bank/: work/bank/snippets.npz (id -> float32 wav) + snippets.json.

    Sampled in proportion to `uses` (how often the snippet occurs in validation synthetic clips), or
    uniformly. exclude_source_clips drops snippets first found in those clips (strict CV; the blind set
    reuses the same bank, so this is off by default).
    """

    def __init__(self, bank_dir=WORK / "bank", uniform: bool = False, exclude_source_clips=(),
                 tag_prior: dict | None = None, crop_prob: float | None = None):
        super().__init__(tag_prior, uniform, crop_prob)
        self.bank_dir, self.exclude = Path(bank_dir), {clip_id(str(c)) for c in exclude_source_clips}
        self._wav = {}

    def _build(self):
        meta = json.loads((self.bank_dir / "snippets.json").read_text())
        keep = [m for m in meta if clip_id(str(m.get("source_clip") or "")) not in self.exclude]
        with np.load(self.bank_dir / "snippets.npz", allow_pickle=False) as z:
            self._wav = {str(m["id"]): np.asarray(z[str(m["id"])], dtype=np.float32) for m in keep}
        items = [NoiseItem(f"bank:{m['id']}", m["category"], norm_tag(m["tag"]), len(self._wav[str(m["id"])]), (str(m["id"]),))
                 for m in keep]
        return items, [max(float(m.get("uses") or 1), 1.0) for m in keep]

    def load(self, item):
        if not self._wav:
            self.items
        return self._wav[item.ref[0]]


class GoldEventSource(NoiseSource):
    """Annotated spans cut from Gold training clips, kept only if no other span overlaps them
    (within `isolation` s) and min_dur <= duration <= max_dur. The cut still contains that clip's
    speech, and its edges are only as exact as the human annotation. Index cached as JSON.
    Items are uniform per span; pass tag_prior to match the synthetic tag mix.
    """

    def __init__(self, data_dir=STAGED_DIR, tiers=("gold",), min_dur: float = 0.12, max_dur: float = 9.5,
                 isolation: float = 0.0, cache=None, tag_prior: dict | None = None, crop_prob: float | None = None):
        super().__init__(tag_prior, True, crop_prob)
        self.blob, self.tiers, self.cache = StagedBlob(data_dir), tuple(tiers), cache
        self.min_dur, self.max_dur, self.isolation = float(min_dur), float(max_dur), float(isolation)

    def _scan(self):
        out = []
        for i, row in self.blob.rows():
            if row["tier"] not in self.tiers:
                continue
            dur, spans = row["samples"] / SR, row["spans"]
            for j, (a, b, cat, tag, *_) in enumerate(spans):
                b = min(float(b), dur)
                if cat not in CATEGORIES or not self.min_dur <= b - a <= self.max_dur:
                    continue
                if any(k != j and max(a - self.isolation, s[0]) < min(b + self.isolation, s[1]) for k, s in enumerate(spans)):
                    continue
                s0 = int(round(a * SR))
                out.append([row["offset"] + s0, int(round(b * SR)) - s0, cat, norm_tag(tag), i])
        return out

    def _build(self):
        sig = self.blob.signature(kind="gold_events", tiers=list(self.tiers), min_dur=self.min_dur,
                                  max_dur=self.max_dur, isolation=self.isolation)
        rows = _cached(self.cache, "gold_events", sig, self._scan)
        items = [NoiseItem(f"gold:{r[4]}:{r[0]}", r[2], r[3], r[1], (r[0],)) for r in rows]
        return items, np.ones(len(items))

    def load(self, item):
        return self.blob.read(item.ref[0], item.n)


class FolderSource(NoiseSource):
    """External isolated-event audio under `folder`, labelled by a CSV with columns
    category, tag and either `file` (path relative to folder) or `dir` (applies to every file below it;
    the longest matching dir wins), plus an optional `weight`. Leading/trailing audio quieter than
    peak - trim_db is trimmed so labels hug the event (trim_db=None disables trimming).
    """

    def __init__(self, folder, mapping_csv, trim_db: float | None = 40.0, min_dur: float = 0.12,
                 cache=None, tag_prior: dict | None = None, crop_prob: float | None = 0.5):
        super().__init__(tag_prior, False, crop_prob)
        self.folder, self.csv, self.cache = Path(folder).expanduser(), Path(mapping_csv).expanduser(), cache
        self.trim_db, self.min_dur = trim_db, float(min_dur)

    def _trim(self, wav: np.ndarray, sr: int):
        if self.trim_db is None:
            return 0, len(wav)
        hop = max(1, sr // 100)
        m = len(wav) // hop
        if m == 0:
            return 0, len(wav)
        e = np.sqrt((wav[:m * hop].reshape(m, hop) ** 2).mean(1) + 1e-12)
        on = np.flatnonzero(20 * np.log10(e / e.max()) > -self.trim_db)
        return int(on[0] * hop), int(min(len(wav), (on[-1] + 1) * hop))

    def _scan(self):
        files, dirs = {}, []
        with self.csv.open(newline="") as handle:
            for r in csv.DictReader(handle):
                lab = (r["category"].strip(), norm_tag(r["tag"]), float(r.get("weight") or 1.0))
                if r.get("file"):
                    files[Path(r["file"].strip()).as_posix()] = lab
                elif r.get("dir"):  # "." = whole folder
                    dirs.append((Path(r["dir"].strip()).as_posix().strip("./"), lab))
        dirs.sort(key=lambda d: -len(d[0]))
        out = []
        for p in sorted(q for q in self.folder.rglob("*") if q.suffix.lower() in AUDIO_EXTS):
            rel = p.relative_to(self.folder).as_posix()
            lab = files.get(rel) or next((l for d, l in dirs if not d or rel.startswith(d + "/")), None)
            if lab is None or lab[0] not in CATEGORIES:
                continue
            wav, sr = sf.read(str(p), dtype="float32", always_2d=True)
            s0, s1 = self._trim(wav.mean(1), sr)
            n = int(round((s1 - s0) * SR / sr))
            if n >= self.min_dur * SR:
                out.append([rel, s0, s1, n, *lab])
        return out

    def _build(self):
        st = self.csv.stat()
        sig = dict(kind="folder", folder=str(self.folder), csv=str(self.csv), size=st.st_size,
                   mtime=int(st.st_mtime), trim_db=self.trim_db, min_dur=self.min_dur)
        rows = _cached(self.cache, "folder", sig, self._scan)
        items = [NoiseItem(f"folder:{r[0]}", r[4], r[5], r[3], (r[0], r[1], r[2])) for r in rows]
        return items, [r[6] for r in rows]

    def load(self, item):
        rel, s0, s1 = item.ref
        wav = read_audio(self.folder / rel, s0, s1)
        return wav[:item.n] if len(wav) >= item.n else np.pad(wav, (0, item.n - len(wav)))


# ---------------------------------------------------------------------------------------------- speech sources

def xfade_concat(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(a), len(b))
    if k == 0:
        return np.concatenate([a, b])
    r = np.linspace(0.0, 1.0, k, dtype=np.float32)
    return np.concatenate([a[:-k], a[-k:] * (1 - r) + b[:k] * r, b[k:]])


class SpeechSource:
    """Base class. Subclasses set self.lengths (int samples per clip/region) via _index() and implement _read.

    sample(rng, n): with probability 1 - splice_prob, a random n-sample crop of one item long enough
    (chosen in proportion to length); otherwise, or if none is long enough, random items are
    concatenated with a 10 ms crossfade and cropped.
    """
    xfade = int(0.01 * SR)

    def __init__(self, splice_prob: float = 0.0):
        self.splice_prob, self._lengths = splice_prob, None

    @property
    def lengths(self) -> np.ndarray:
        if self._lengths is None:
            self._lengths = np.asarray(self._index(), dtype=np.int64)
            if not len(self._lengths):
                raise ValueError(f"{type(self).__name__}: no usable speech")
        return self._lengths

    def _index(self) -> list:
        raise NotImplementedError

    def _read(self, i: int, start: int, n: int) -> np.ndarray:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        lens = self.lengths
        if n is None:
            i = int(rng.integers(len(lens)))
            return self._read(i, 0, int(lens[i]))
        ok = np.flatnonzero(lens >= n)
        if len(ok) and rng.random() >= self.splice_prob:
            i = int(rng.choice(ok, p=lens[ok] / lens[ok].sum()))
            s = int(rng.integers(0, lens[i] - n + 1))
            return self._read(i, s, n)
        x = np.zeros(0, dtype=np.float32)
        while len(x) < n:
            i = int(rng.integers(len(lens)))
            x = xfade_concat(x, self._read(i, 0, int(lens[i])), self.xfade)
        s = int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n].copy()


class ValCleanSource(SpeechSource):
    """Organiser clean references (data/val/validation/syntheticCleanRefAudio). exclude_ids (clip ids,
    e.g. ids_in_folds(heldout)) are never used, so held-out folds stay unseen."""

    def __init__(self, exclude_ids=(), clean_dir=VAL_DIR / "syntheticCleanRefAudio", splice_prob: float = 0.0):
        super().__init__(splice_prob)
        self.clean_dir, self.exclude = Path(clean_dir), {clip_id(str(c)) for c in exclude_ids}
        self.paths, self.srs = [], []

    def _index(self):
        self.paths = sorted(p for p in self.clean_dir.glob("*.wav") if clip_id(p.name) not in self.exclude)
        infos = [sf.info(str(p)) for p in self.paths]
        self.srs = [i.samplerate for i in infos]
        return [int(i.frames * SR // i.samplerate) for i in infos]

    def _read(self, i, start, n):
        if self.srs[i] == SR:
            x = read_audio(self.paths[i], start, start + n)
        else:
            x = read_audio(self.paths[i])[start:start + n]
        return x if len(x) == n else np.pad(x, (0, n - len(x)))


class GoldSpeechSource(SpeechSource):
    """Event-free regions of Gold/Silver training clips: at least `margin` s from any annotated span
    and at least `min_len` s long. Index cached as JSON. (Unannotated noise may still be present.)"""

    def __init__(self, data_dir=STAGED_DIR, tiers=("gold", "silver"), margin: float = 0.05, min_len: float = 1.5,
                 cache=None, splice_prob: float = 0.0):
        super().__init__(splice_prob)
        self.blob, self.tiers, self.cache = StagedBlob(data_dir), tuple(tiers), cache
        self.margin, self.min_len = float(margin), float(min_len)
        self.offsets = None

    def _scan(self):
        out = []
        for _, row in self.blob.rows():
            if row["tier"] not in self.tiers:
                continue
            dur, t = row["samples"] / SR, 0.0
            for a, b in sorted((max(0.0, s[0] - self.margin), min(dur, s[1] + self.margin)) for s in row["spans"]) + [(dur, dur)]:
                if a - t >= self.min_len:
                    s0 = int(math.ceil(t * SR))
                    out.append([row["offset"] + s0, int(a * SR) - s0])
                t = max(t, b)
        return out

    def _index(self):
        sig = self.blob.signature(kind="gold_speech", tiers=list(self.tiers), margin=self.margin, min_len=self.min_len)
        rows = _cached(self.cache, "gold_speech", sig, self._scan)
        self.offsets = [r[0] for r in rows]
        return [r[1] for r in rows]

    def _read(self, i, start, n):
        return self.blob.read(self.offsets[i] + start, n)
