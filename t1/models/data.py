"""Training data: per-tier datasets, mixture batch sampler, collate, waveform augmentations.

Every source yields the synthgen item format:
    wav [N] float32, length int, target [24, T] float32 (T = ceil(length / 320)),
    clip_cats [7] float32, tier str, weight float, boundary_weight float
`weight` / `boundary_weight` are per-item multipliers (1.0 for real data); the per-tier weights live
in the loss config so they are configured in exactly one place.
"""
from __future__ import annotations

import importlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from common import contract
from common.contract import CH_CAT, CH_UNION, N_CH, SR
from common.targets import clip_categories, frame_targets
from models.config import STRONG_TIERS, TIERS

HOP = 320                      # 20 ms at 16 kHz
TIER_ID = {t: i for i, t in enumerate(TIERS)}
ITEM_KEYS = ("wav", "length", "target", "clip_cats", "tier", "weight", "boundary_weight")


def frames_for(samples: int) -> int:
    return max(1, -(-int(samples) // HOP))


def set_data_roots(val_dir=None, blind_dir=None, work_dir=None):
    """Redirect common.contract paths (the cluster layout differs). Call before building datasets."""
    for attr, value, must_contain in (("VAL_DIR", val_dir, "validationMetadata.json"),
                                      ("BLIND_DIR", blind_dir, None), ("WORK", work_dir, "folds.json")):
        if not value:
            continue
        p = Path(value).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"{attr} override is not a directory: {p}")
        if must_contain and not (p / must_contain).exists():
            raise FileNotFoundError(f"{attr} override {p} has no {must_contain}")
        setattr(contract, attr, p)


# ============================================================================ item building
def choose_crop(n_samples: int, max_samples: int, events, event_prob: float, background_prob: float) -> int:
    """Start sample of an event-biased random crop (0 when the clip fits)."""
    room = n_samples - max_samples
    if room <= 0:
        return 0
    spans = [(float(e[0]), float(e[1])) for e in events if float(e[1]) > float(e[0])]
    if spans and random.random() < background_prob:
        for _ in range(16):          # hard negative: a window containing no event
            s = random.randint(0, room)
            lo, hi = s / SR, (s + max_samples) / SR
            if not any(max(lo, a) < min(hi, b) for a, b in spans):
                return s
    if spans and random.random() < event_prob:
        a, b = random.choice(spans)
        a_s, b_s = int(a * SR), int(math.ceil(b * SR))
        if b_s - a_s <= max_samples:          # whole event inside the window, at a random position
            lo, hi = max(0, b_s - max_samples), min(room, a_s)
        else:                                 # event longer than the window: window inside the event
            lo, hi = max(0, a_s), min(room, b_s - max_samples)
        if lo <= hi:
            return random.randint(lo, hi)
    return random.randint(0, room)


def crop_events(events, start_s: float, dur_s: float):
    """Shift events into crop coordinates and clip to [0, dur]. Events cut by the crop end exactly on the
    crop edge, so frame_targets gives them no boundary target at the cut."""
    out = []
    for ev in events:
        a, b = float(ev[0]) - start_s, float(ev[1]) - start_s
        if b <= 0 or a >= dur_s:
            continue
        out.append((max(0.0, a), min(dur_s, b), ev[2], *ev[3:]))
    return out


def time_shift(wav: np.ndarray, target: np.ndarray, max_frames: int, max_samples: int):
    """Shift audio and labels together by a whole number of 20 ms frames (zero padding / dropping)."""
    m = random.randint(-max_frames, max_frames)
    if m == 0:
        return wav, target
    if m > 0 and target[CH_UNION, 0] > 0:
        m = -m        # delaying an event truncated at the clip start would fabricate an untargeted onset
    if m > 0:
        wav = np.concatenate([np.zeros(m * HOP, np.float32), wav])
        target = np.concatenate([np.zeros((N_CH, m), np.float32), target], axis=1)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
            target = target[:, :frames_for(max_samples)]
    else:
        k = -m
        if k >= target.shape[1] - 1:
            return wav, target
        wav = wav[k * HOP:]
        target = target[:, k:]
    if target.shape[1] != frames_for(len(wav)):
        raise AssertionError("time_shift produced misaligned targets")
    return wav, target


def apply_gain(wav: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db > 0:
        wav = wav * np.float32(10 ** (random.uniform(-gain_db, gain_db) / 20))
    return wav


def strong_item(wav_full_reader, n_samples: int, events, tier: str, data_cfg, aug_cfg, sigma: float, train=True):
    """wav_full_reader(start, stop) -> float32 audio of [start, stop)."""
    max_samples = int(round(data_cfg["crop_seconds"] * SR))
    if train:
        start = choose_crop(n_samples, max_samples, events, data_cfg["event_crop_prob"], data_cfg["background_crop_prob"])
    else:
        start = 0
    stop = min(n_samples, start + max_samples)
    wav = np.asarray(wav_full_reader(start, stop), dtype=np.float32)
    if len(wav) != stop - start or len(wav) == 0:
        raise RuntimeError(f"audio read returned {len(wav)} samples, expected {stop - start}")
    dur = len(wav) / SR
    evs = crop_events(events, start / SR, dur)
    target = frame_targets(evs, dur, sigma_frames=sigma, n=frames_for(len(wav)))
    if train and aug_cfg["time_shift_prob"] > 0 and random.random() < aug_cfg["time_shift_prob"]:
        wav, target = time_shift(wav, target, int(aug_cfg["time_shift_max_frames"]), max_samples)
    if train:
        wav = apply_gain(wav, aug_cfg["gain_db"])
    return {"wav": np.ascontiguousarray(wav, dtype=np.float32), "length": len(wav), "target": target,
            "clip_cats": (target[CH_CAT].max(axis=1) > 0.5).astype(np.float32), "tier": tier,
            "weight": 1.0, "boundary_weight": 1.0}


def bronze_item(wav: np.ndarray, categories, aug_cfg, train=True):
    wav = np.asarray(wav, dtype=np.float32)
    if train:
        wav = apply_gain(wav, aug_cfg["gain_db"])
    return {"wav": np.ascontiguousarray(wav), "length": len(wav),
            "target": np.zeros((N_CH, frames_for(len(wav))), np.float32),
            "clip_cats": clip_categories(list(categories)), "tier": "bronze", "weight": 1.0, "boundary_weight": 1.0}


# ============================================================================ datasets
class _RetryMixin:
    """A failing item is logged and replaced by a random other item; more than MAX_FAILURES failures in one
    worker (or 4 in a row) raise, so systematic bugs still stop training immediately."""
    MAX_FAILURES = 50

    def __getitem__(self, index):
        for _ in range(4):
            try:
                return self._get(index)
            except Exception as exc:  # noqa: BLE001 - logged and bounded
                self._failures = getattr(self, "_failures", 0) + 1
                print(f"[data] {type(self).__name__}[{index}] failed: {type(exc).__name__}: {exc} "
                      f"(failure {self._failures}/{self.MAX_FAILURES}); resampling", file=sys.stderr, flush=True)
                if self._failures > self.MAX_FAILURES:
                    raise
                index = random.randrange(len(self))
        raise RuntimeError(f"{type(self).__name__}: 4 consecutive item failures")


class StagedTierDataset(_RetryMixin, Dataset):
    """One tier (gold / silver / bronze) of the Phase 2 staged blob: READY, manifest.jsonl, audio.f16."""

    def __init__(self, staged_dir, tier: str, data_cfg, aug_cfg, train=True):
        if tier not in ("gold", "silver", "bronze"):
            raise ValueError(tier)
        self.dir = Path(staged_dir).expanduser()
        for f in ("READY", "manifest.jsonl", "audio.f16"):
            if not (self.dir / f).exists():
                raise FileNotFoundError(f"staged data incomplete: {self.dir / f} missing")
        self.tier, self.data_cfg, self.aug_cfg, self.train = tier, data_cfg, aug_cfg, train
        items = []
        with (self.dir / "manifest.jsonl").open() as handle:
            for line in handle:
                x = json.loads(line)
                if x["tier"] == tier:
                    items.append(x)
        self.excluded = 0
        if tier == "bronze":
            cap = int(round(data_cfg["bronze_max_seconds"] * SR))
            kept = [x for x in items if 0 < x["samples"] <= cap]
            self.excluded = len(items) - len(kept)
            items = kept
        self.items = [{"offset": int(x["offset"]), "samples": int(x["samples"]),
                       "spans": [tuple(s) for s in x.get("spans", [])], "categories": list(x.get("categories") or [])}
                      for x in items]
        if not self.items:
            raise RuntimeError(f"no {tier} items in {self.dir / 'manifest.jsonl'}")
        self.sigma = float(data_cfg["sigma_frames"].get(tier, 1.0))
        self._blob = None
        self._blob_len = (self.dir / "audio.f16").stat().st_size // 2
        last = max(x["offset"] + x["samples"] for x in self.items)
        if last > self._blob_len:
            raise RuntimeError(f"manifest points beyond the audio blob ({last} > {self._blob_len} samples)")

    def __getstate__(self):          # never pickle a memmap into workers
        state = dict(self.__dict__)
        state["_blob"] = None
        return state

    def _blob_view(self):
        if self._blob is None:
            self._blob = np.memmap(self.dir / "audio.f16", dtype=np.float16, mode="r")
        return self._blob

    def __len__(self):
        return len(self.items)

    def _get(self, index):
        x = self.items[index]
        blob = self._blob_view()
        read = lambda a, b: np.asarray(blob[x["offset"] + a:x["offset"] + b], dtype=np.float32)
        if self.tier == "bronze":
            return bronze_item(read(0, x["samples"]), x["categories"], self.aug_cfg, self.train)
        return strong_item(read, x["samples"], x["spans"], self.tier, self.data_cfg, self.aug_cfg, self.sigma, self.train)


def read_wav(path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if len(wav) == 0:
        raise ValueError(f"{path}: empty audio")
    return np.ascontiguousarray(wav, dtype=np.float32)


class ValNaturalDataset(_RetryMixin, Dataset):
    """Validation clips (Gold quality, test conventions) whose fold is NOT in holdout_folds."""

    def __init__(self, data_cfg, aug_cfg, holdout_folds, include_synthetic=False, train=True):
        val = contract.load_validation()
        fold_of = contract.folds()
        missing = [k for k in val if k not in fold_of]
        if missing:
            raise RuntimeError(f"{len(missing)} validation clips have no fold in folds.json (e.g. {missing[0]})")
        holdout = {int(f) for f in holdout_folds}
        self.items = []
        for cid in sorted(val):
            v = val[cid]
            if fold_of[cid] in holdout or (v["synthetic"] and not include_synthetic):
                continue
            path = contract.val_audio_path(cid, v["synthetic"])
            self.items.append((cid, str(path), v["events"]))
        if not self.items:
            raise RuntimeError("val_natural is empty after removing holdout folds")
        self.ids = [c for c, _, _ in self.items]
        self.data_cfg, self.aug_cfg, self.train = data_cfg, aug_cfg, train
        self.sigma = float(data_cfg["sigma_frames"].get("val_natural", 1.0))

    def __len__(self):
        return len(self.items)

    def _get(self, index):
        cid, path, events = self.items[index]
        wav = read_wav(path)
        return strong_item(lambda a, b: wav[a:b], len(wav), events, "val_natural", self.data_cfg, self.aug_cfg,
                           self.sigma, self.train)


def _import_factory(spec: str):
    module, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"cannot import synthetic factory module {module!r} ({exc})") from exc
    if not hasattr(mod, attr):
        raise AttributeError(f"{module} has no attribute {attr!r}; set synthetic.factory in the config")
    return getattr(mod, attr)


class SyntheticSource(_RetryMixin, Dataset):
    """Thin wrapper around the synthgen Dataset that normalises item types (format checked in collate)."""

    def __init__(self, factory: str, kwargs: dict):
        self.inner = _import_factory(factory)(**kwargs)
        if len(self.inner) == 0:
            raise RuntimeError("synthetic dataset is empty")

    def __len__(self):
        return len(self.inner)

    def _get(self, index):
        item = dict(self.inner[index])
        missing = [k for k in ITEM_KEYS if k not in item]
        if missing:
            raise KeyError(f"synthetic item lacks keys {missing}")
        if item["tier"] != "synthetic":
            raise ValueError(f"synthetic item has tier {item['tier']!r}")
        return item


def build_synthgen(mixer=None, exclude_folds=(4,), size=1_000_000, seed=0, max_seconds=10.0, crop_seconds=None,
                   weight=1.0, sigma_frames=1.0, offline_root=None):
    """Default synthetic factory. mixer: overrides dict for synthgen.build_mixer (see synthgen/README.md);
    exclude_folds: validation folds whose clean speech the mixer must never use (monitor/holdout folds);
    offline_root: use shards from synthgen.write_offline_set instead of on-the-fly mixing."""
    from synthgen.dataset import OfflineSynthDataset, SynthDataset
    item_kw = dict(seed=seed, max_seconds=max_seconds, crop_seconds=crop_seconds, weight=weight,
                   sigma_frames=sigma_frames)
    if offline_root:
        ds = OfflineSynthDataset(str(Path(offline_root).expanduser()), **item_kw)
    else:
        from synthgen import build_mixer, ids_in_folds
        mx = build_mixer(dict(mixer or {}), exclude_ids=ids_in_folds(list(exclude_folds)))
        ds = SynthDataset(mx, int(size), **item_kw)
    # Build lazy source indexes once in the main process (synthgen README) and validate the item format.
    check_item(ds[0])
    return ds


def build_sources(cfg, staged_dir=None, train=True):
    """Ordered {tier: Dataset} for every tier with a positive batch fraction."""
    d, a = cfg["data"], cfg["augment"]
    staged = staged_dir or d["staged_dir"]
    out = {}
    for tier in TIERS:
        if cfg["sources"][tier] <= 0:
            continue
        if tier in ("gold", "silver", "bronze"):
            out[tier] = StagedTierDataset(staged, tier, d, a, train)
        elif tier == "val_natural":
            out[tier] = ValNaturalDataset(d, a, d["val_holdout_folds"], d["val_include_synthetic"], train)
        else:
            out[tier] = SyntheticSource(cfg["synthetic"]["factory"], cfg["synthetic"]["kwargs"])
    return out


# ============================================================================ sampler
class MixtureBatchSampler(Sampler):
    """Infinite batch sampler: each batch holds round(fraction * batch_size) items per source (largest
    remainder with random tie-breaking, so expected fractions are exact). Each source is walked through
    a reshuffled permutation (no repeats within a pass)."""

    def __init__(self, sizes: dict, fractions: dict, batch_size: int, seed: int):
        self.names = list(sizes)
        self.sizes = np.array([sizes[n] for n in self.names], dtype=np.int64)
        if np.any(self.sizes <= 0):
            raise ValueError(f"empty source in mixture: {sizes}")
        self.frac = np.array([float(fractions[n]) for n in self.names])
        if abs(self.frac.sum() - 1) > 1e-6:
            raise ValueError(f"fractions over active sources must sum to 1: {dict(zip(self.names, self.frac))}")
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.batch_size, self.seed = int(batch_size), int(seed)

    def __len__(self):
        return 10 ** 9

    def _counts(self, rng):
        exact = self.frac * self.batch_size
        base = np.floor(exact).astype(np.int64)
        rem = self.batch_size - base.sum()
        if rem > 0:
            p = exact - base
            pick = rng.choice(len(p), size=rem, replace=False, p=p / p.sum())
            base[pick] += 1
        return base

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        perms = [rng.permutation(n) for n in self.sizes]
        ptr = [0] * len(self.sizes)
        while True:
            batch = []
            for s, k in enumerate(self._counts(rng)):
                for _ in range(int(k)):
                    if ptr[s] >= len(perms[s]):
                        perms[s] = rng.permutation(self.sizes[s])
                        ptr[s] = 0
                    batch.append(int(self.offsets[s] + perms[s][ptr[s]]))
                    ptr[s] += 1
            rng.shuffle(batch)
            yield batch


# ============================================================================ collate
def _np(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def check_item(item):
    missing = [k for k in ITEM_KEYS if k not in item]
    if missing:
        raise KeyError(f"item lacks keys {missing}")
    tier = item["tier"]
    if tier not in TIER_ID:
        raise ValueError(f"unknown tier {tier!r}")
    wav = _np(item["wav"]).reshape(-1)
    length = int(item["length"].item() if isinstance(item["length"], torch.Tensor) else item["length"])
    if not 0 < length <= len(wav):
        raise ValueError(f"{tier} item: length {length} vs wav {len(wav)}")
    wav = wav[:length]
    target = _np(item["target"])
    t = frames_for(length)
    if target.ndim != 2 or target.shape[0] != N_CH:
        raise ValueError(f"{tier} item: target shape {target.shape}, expected [{N_CH}, {t}]")
    if target.shape[1] != t:
        if target.shape[1] > t and target.shape[1] == frames_for(len(_np(item["wav"]).reshape(-1))):
            target = target[:, :t]               # padded wav + valid length: drop padded frames
        else:
            raise ValueError(f"{tier} item: target T={target.shape[1]} but ceil(length/320)={t}")
    cats = _np(item["clip_cats"]).reshape(-1)
    if cats.shape != (7,):
        raise ValueError(f"{tier} item: clip_cats shape {cats.shape}")
    return {"wav": wav, "length": length, "target": target, "clip_cats": cats, "tier": tier,
            "weight": float(item["weight"]), "boundary_weight": float(item["boundary_weight"])}


def union_mix(a, b, lam_range):
    """Label-union mixup of two items of the same tier."""
    n = max(a["length"], b["length"])
    t = frames_for(n)
    lam = random.uniform(*lam_range)
    wav = np.zeros(n, np.float32)
    wav[:a["length"]] += lam * a["wav"]
    wav[:b["length"]] += (1 - lam) * b["wav"]
    target = np.zeros((N_CH, t), np.float32)
    target[:, :a["target"].shape[1]] = a["target"]
    target[:, :b["target"].shape[1]] = np.maximum(target[:, :b["target"].shape[1]], b["target"])
    return {"wav": wav, "length": n, "target": target, "clip_cats": np.maximum(a["clip_cats"], b["clip_cats"]),
            "tier": a["tier"], "weight": min(a["weight"], b["weight"]),
            "boundary_weight": min(a["boundary_weight"], b["boundary_weight"])}


class Collator:
    def __init__(self, aug_cfg=None, train=True):
        self.aug = aug_cfg or {}
        self.train = train

    def __call__(self, raw):
        items = [check_item(x) for x in raw]
        p = float(self.aug.get("mixup_prob", 0.0)) if self.train else 0.0
        if p > 0:
            by_tier = {}
            for i, x in enumerate(items):
                if x["tier"] in STRONG_TIERS:
                    by_tier.setdefault(x["tier"], []).append(i)
            mixed = list(items)
            for i, x in enumerate(items):
                group = by_tier.get(x["tier"], [])
                if len(group) > 1 and random.random() < p:
                    j = random.choice([g for g in group if g != i])
                    mixed[i] = union_mix(x, items[j], self.aug.get("mixup_lam", (0.4, 0.6)))
            items = mixed
        b = len(items)
        n = max(x["length"] for x in items)
        n = frames_for(n) * HOP
        t = n // HOP
        wav = torch.zeros(b, n)
        target = torch.zeros(b, N_CH, t)
        mask = torch.zeros(b, t)
        for i, x in enumerate(items):
            wav[i, :x["length"]] = torch.from_numpy(x["wav"])
            target[i, :, :x["target"].shape[1]] = torch.from_numpy(x["target"])
            mask[i, :frames_for(x["length"])] = 1
        return {"wav": wav, "length": torch.tensor([x["length"] for x in items], dtype=torch.long),
                "target": target, "mask": mask,
                "clip_cats": torch.from_numpy(np.stack([x["clip_cats"] for x in items])),
                "tier": torch.tensor([TIER_ID[x["tier"]] for x in items], dtype=torch.long),
                "weight": torch.tensor([x["weight"] for x in items], dtype=torch.float32),
                "boundary_weight": torch.tensor([x["boundary_weight"] for x in items], dtype=torch.float32)}


def _worker_init(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(sources: dict, cfg, seed: int, pin_memory=True):
    fractions = {k: cfg["sources"][k] for k in sources}
    sampler = MixtureBatchSampler({k: len(v) for k, v in sources.items()}, fractions, cfg["optim"]["batch_size"], seed)
    concat = ConcatDataset(list(sources.values()))
    workers = cfg["train"]["workers"]
    kw = dict(persistent_workers=True, prefetch_factor=cfg["train"]["prefetch_factor"]) if workers > 0 else {}
    return DataLoader(concat, batch_sampler=sampler, num_workers=workers, collate_fn=Collator(cfg["augment"], True),
                      pin_memory=pin_memory, worker_init_fn=_worker_init, **kw)


# ============================================================================ GPU waveform augmentations
_MEL_MAX = 2595.0 * math.log10(1 + 8000 / 700)


def filter_augment(wav: torch.Tensor, prob: float, db: float, bands=(3, 6)) -> torch.Tensor:
    """FilterAugment-style random band gains applied to the waveform spectrum (per item, mel-spaced
    knots; half the items get step bands, half piecewise-linear dB curves)."""
    if prob <= 0 or db <= 0:
        return wav
    sel = torch.nonzero(torch.rand(wav.shape[0]) < prob).flatten().tolist()
    if not sel:
        return wav
    n = wav.shape[1]
    spec = torch.fft.rfft(wav[sel].float(), dim=-1)
    f = spec.shape[-1]
    freqs = torch.linspace(0, SR / 2, f, device=wav.device)
    pos = (2595.0 * torch.log10(1 + freqs / 700)) / _MEL_MAX
    gains = []
    for _ in sel:
        k = random.randint(int(bands[0]), int(bands[1]))
        edges = torch.sort(torch.rand(k - 1, device=wav.device)).values
        edges = torch.cat([edges.new_zeros(1), edges, edges.new_ones(1)])
        g = (torch.rand(k + 1, device=wav.device) * 2 - 1) * db
        if random.random() < 0.5:     # step: constant gain per band
            idx = torch.bucketize(pos, edges[1:-1])
            curve = g[:k][idx]
        else:                         # linear: interpolate dB between knots
            idx = torch.bucketize(pos, edges).clamp(1, k)
            x0, x1 = edges[idx - 1], edges[idx]
            w = ((pos - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0, 1)
            curve = g[idx - 1] * (1 - w) + g[idx] * w
        gains.append(curve)
    gain = torch.pow(10.0, torch.stack(gains) / 20)
    out = wav.clone()
    out[sel] = torch.fft.irfft(spec * gain, n=n, dim=-1).to(wav.dtype)
    return out


def freq_mixstyle(wav: torch.Tensor, lengths: torch.Tensor, prob: float, alpha=0.3, n_fft=512, hop=160) -> torch.Tensor:
    """Freq-MixStyle on the log-magnitude STFT: per-frequency (over time) mean/std are mixed with a
    shuffled batch partner; phase is kept; padded tails are re-zeroed."""
    if prob <= 0 or wav.shape[0] < 2 or random.random() >= prob:
        return wav
    n = wav.shape[1]
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav.float(), n_fft, hop, window=window, return_complex=True)
    mag = spec.abs().clamp_min(1e-5)
    logm = mag.log()
    frames = spec.shape[-1]
    valid = (torch.arange(frames, device=wav.device)[None] <= (lengths.to(wav.device) // hop)[:, None]).float()[:, None]
    cnt = valid.sum(-1, keepdim=True).clamp_min(1)
    mu = (logm * valid).sum(-1, keepdim=True) / cnt
    sig = (((logm - mu) ** 2 * valid).sum(-1, keepdim=True) / cnt).add(1e-6).sqrt()
    perm = torch.randperm(wav.shape[0], device=wav.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((wav.shape[0], 1, 1)).to(wav.device)
    mu2, sig2 = lam * mu + (1 - lam) * mu[perm], lam * sig + (1 - lam) * sig[perm]
    new = ((logm - mu) / sig * sig2 + mu2).exp()
    out = torch.istft(new * (spec / mag), n_fft, hop, window=window, length=n)
    keep = (torch.arange(n, device=wav.device)[None] < lengths.to(wav.device)[:, None]).float()
    return (out * keep).to(wav.dtype)
