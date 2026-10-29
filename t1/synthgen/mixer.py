"""Synthetic mixer reproducing the organisers' construction: noisy = clean + sum_i g_i * snippet_i.

Per-event gain g_i is set so that  mean(clean^2 over the whole clip) / mean((g_i*snippet_i)^2) = 10^(snr_i/10).
Events are placed with the measured count / first-onset / tail / min-gap distributions (config_default.yaml),
labels are sample-exact. Every random draw goes through the caller's np.random.Generator.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from common.contract import ROOT, SR

DEFAULT_CONFIG = Path(__file__).with_name("config_default.yaml")
DIST_KEYS = {"probs", "weights", "uniform", "const"}


# --------------------------------------------------------------------------------------------------- config

def deep_merge(base: dict, over: dict | None) -> dict:
    """Recursive dict merge; a distribution spec (dict with probs/weights/uniform/const) replaces wholesale."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not DIST_KEYS & v.keys():
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _scalar(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return s.strip("'")


def mini_yaml(text: str) -> dict:
    """YAML subset used by config_default.yaml (block mappings, JSON-literal values) for hosts without PyYAML."""
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        key = json.loads(key) if key.startswith('"') else key
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if val.strip():
            parent[key] = _scalar(val.strip())
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return root


def read_config_file(path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError:
        return mini_yaml(text)
    return yaml.safe_load(text) or {}


def parse_overrides(pairs) -> dict:
    """['widen_prob=0.3', 'sources.noise.gold.weight=0.5'] -> nested dict (values parsed as JSON when possible)."""
    out = {}
    for pair in pairs or ():
        key, _, val = pair.partition("=")
        node = out
        *path, leaf = key.strip().split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = _scalar(val.strip())
    return out


def load_config(path=None, overrides: dict | None = None) -> dict:
    """config_default.yaml, deep-merged with an optional YAML/JSON file and then an overrides dict."""
    cfg = read_config_file(DEFAULT_CONFIG)
    if path:
        cfg = deep_merge(cfg, read_config_file(path))
    return deep_merge(cfg, overrides)


class Dist:
    """Scalar sampler from a config spec:
    number | [lo, hi] (uniform) | {uniform: [lo, hi]} | {const: x}
    | {probs, values}   inverse CDF, linear between quantile knots
    | {values, weights} discrete choice.
    """

    def __init__(self, spec):
        self.spec = spec
        if isinstance(spec, (int, float)):
            spec = {"const": spec}
        elif isinstance(spec, (list, tuple)):
            spec = {"uniform": spec}
        if "probs" in spec:
            self.kind, self.p, self.v = "q", np.asarray(spec["probs"], float), np.asarray(spec["values"], float)
            if len(self.p) != len(self.v) or np.any(np.diff(self.p) < 0) or np.any(np.diff(self.v) < 0):
                raise ValueError(f"bad quantile spec {spec}")
        elif "weights" in spec:
            w = np.asarray(spec["weights"], float)
            self.kind, self.v, self.cdf = "d", list(spec["values"]), np.cumsum(w / w.sum())
        elif "uniform" in spec:
            self.kind, (self.lo, self.hi) = "u", map(float, spec["uniform"])
        elif "const" in spec:
            self.kind, self.c = "c", float(spec["const"])
        else:
            raise ValueError(f"unknown distribution spec {spec}")

    def __call__(self, rng: np.random.Generator) -> float:
        if self.kind == "q":
            return float(np.interp(rng.random(), self.p, self.v))
        if self.kind == "d":
            return self.v[min(int(np.searchsorted(self.cdf, rng.random(), side="right")), len(self.v) - 1)]
        if self.kind == "u":
            return float(rng.uniform(self.lo, self.hi))
        return self.c


class _Plan:
    """One resolved config (measured or widened) with its samplers."""

    def __init__(self, c: dict):
        self.c = c
        for name in ("n_events", "clip_duration", "first_onset", "last_tail", "event_duration", "snr_db",
                     "fade_ms", "speech_gain_db"):
            setattr(self, name, Dist(c[name]))


class _Weighted:
    def __init__(self, pairs, what: str):
        pairs = [(s, float(w)) for s, w in pairs if float(w) > 0]
        if not pairs:
            raise ValueError(f"SynthMixer needs at least one {what} source with weight > 0")
        self.items = [s for s, _ in pairs]
        w = np.array([w for _, w in pairs])
        self.cdf = np.cumsum(w / w.sum())

    def pick(self, rng):
        return self.items[min(int(np.searchsorted(self.cdf, rng.random(), side="right")), len(self.items) - 1)]


# ---------------------------------------------------------------------------------------------------- mixer

def _fade(seg: np.ndarray, k: int):
    k = min(k, len(seg) // 2)
    if k > 0:
        r = (0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)).astype(np.float32)
        seg[:k] *= r
        seg[-k:] *= r[::-1]


class SynthMixer:
    """noise_sources / speech_sources: [(source, weight)]; cfg: overrides deep-merged over config_default.yaml.

    sample(rng) -> dict(wav=float32[N], clean=float32[N], events=[(on, off, category, tag)] sorted by onset,
                        duration=N/SR, info=[dict(sid, snr_db, gain)] aligned with events)
    """

    def __init__(self, noise_sources, speech_sources, cfg: dict | None = None):
        self.cfg = deep_merge(load_config(), cfg)
        self.noise, self.speech = _Weighted(noise_sources, "noise"), _Weighted(speech_sources, "speech")
        base = {k: v for k, v in self.cfg.items() if k != "widen"}
        self.plans = (_Plan(base), _Plan(deep_merge(base, self.cfg.get("widen"))))

    # -- event drawing / layout -------------------------------------------------------------------------
    def _draw_event(self, rng, plan):
        """-> [source, item, start_in_snippet, length] with optional crop to a sampled event duration."""
        c = plan.c
        src = self.noise.pick(rng)
        item = src.pick(rng)
        n, start = item.n, 0
        prob = c["crop"]["prob"] if getattr(src, "crop_prob", None) is None else src.crop_prob
        cap = int(c["event_max_s"] * SR)
        if n > cap or (n > c["crop"]["min_snippet_s"] * SR and rng.random() < prob):
            target = min(n, cap, max(int(c["event_min_s"] * SR), int(plan.event_duration(rng) * SR)))
            start, n = int(rng.integers(0, item.n - target + 1)), target
        return [src, item, start, n]

    @staticmethod
    def _chain(rng, lens, n, gap, lead, tail, edge_w):
        """Starts for events laid out in order with >= gap between them; leftover time is spread by a
        Dirichlet over (lead, gaps..., tail) with weight edge_w on the edges (edges only when one event)."""
        m = len(lens)
        slack = n - sum(lens) - gap * (m - 1) - lead - tail
        w = np.r_[edge_w, np.ones(m - 1), edge_w] if m > 1 else np.ones(2)
        part = np.zeros(m + 1)
        if w.sum() > 0:
            part[w > 0] = rng.dirichlet(w[w > 0])
        extra = np.floor(part * max(slack, 0)).astype(np.int64)
        starts, t = [], lead + int(extra[0])
        for j, L in enumerate(lens):
            starts.append(t)
            t += L + gap + int(extra[j + 1])
        return starts

    def _layout(self, rng, plan, n, lens):
        """Start sample per event, or None if this draw does not fit."""
        c = plan.c
        k = len(lens)
        loose = [i for i in range(k) if k > 1 and rng.random() < c["overlap_prob"]]
        chain = [i for i in range(k) if i not in loose]
        starts = [0] * k
        for i in loose:  # widen mode only: placed anywhere, may overlap other events
            if lens[i] > n:
                return None
            starts[i] = int(rng.integers(0, n - lens[i] + 1))
        if chain:
            gap, need = int(c["gap_min_s"] * SR), sum(lens[i] for i in chain)
            for _ in range(c["edge_tries"]):
                lead, tail = int(plan.first_onset(rng) * SR), int(plan.last_tail(rng) * SR)
                if need + gap * (len(chain) - 1) + lead + tail <= n:
                    break
            else:
                return None
            for i, s in zip(chain, self._chain(rng, [lens[i] for i in chain], n, gap, lead, tail, c["edge_slack_weight"])):
                starts[i] = s
        return starts

    def _degrade(self, rng, plan, n, picks):
        """Rejection failed: no edge margins, crop events to fit and drop the longest ones only if needed."""
        c = plan.c
        gap, lo = int(c["gap_min_s"] * SR), int(c["event_min_s"] * SR)
        picks = sorted(picks, key=lambda p: p[3])
        while picks and (n - gap * (len(picks) - 1)) // len(picks) < lo:
            picks.pop()
        if not picks:
            return [], []
        cap = (n - gap * (len(picks) - 1)) // len(picks)
        for p in picks:
            if p[3] > cap:
                p[2] += int(rng.integers(0, p[3] - cap + 1))
                p[3] = cap
        order = rng.permutation(len(picks))
        picks = [picks[i] for i in order]
        return picks, self._chain(rng, [p[3] for p in picks], n, gap, 0, 0, 1.0)

    # -- render -------------------------------------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> dict:
        plan = self.plans[int(rng.random() < self.cfg["widen_prob"])]
        c = plan.c
        speech = self.speech.pick(rng)
        max_n = int(c["clip_max_s"] * SR)
        clean = speech.sample(rng, None)[:max_n] if c["clip_duration_mode"] == "speech" else None
        k = int(plan.n_events(rng))
        n, picks, starts = 0, [], None
        for _ in range(c["max_tries"]):
            n = len(clean) if clean is not None else min(max_n, int(round(plan.clip_duration(rng) * SR)))
            picks = [self._draw_event(rng, plan) for _ in range(k)]
            starts = self._layout(rng, plan, n, [p[3] for p in picks])
            if starts is not None:
                break
        else:
            picks, starts = self._degrade(rng, plan, n, picks)
        if clean is None:
            clean = speech.sample(rng, n)
        clean = np.asarray(clean, dtype=np.float32) * np.float32(10 ** (plan.speech_gain_db(rng) / 20))

        p_clean = max(float(np.mean(clean.astype(np.float64) ** 2)), c["min_clean_power"])
        clip_snr = plan.snr_db(rng)
        noise = np.zeros(n, dtype=np.float32)
        events, info = [], []
        for (src, item, s0, L), t0 in zip(picks, starts):
            seg = np.array(src.load(item)[s0:s0 + L], dtype=np.float32)
            if len(seg) < L:
                seg = np.pad(seg, (0, L - len(seg)))
            _fade(seg, int(plan.fade_ms(rng) * SR / 1000))
            snr = clip_snr if c["snr_per_clip"] else plan.snr_db(rng)
            p_noise = float(np.mean(seg.astype(np.float64) ** 2))
            if p_noise < c["min_noise_power"]:  # silent snippet: no audible event, so no label
                continue
            g = float(np.sqrt(p_clean / (p_noise * 10 ** (snr / 10))))
            noise[t0:t0 + L] += np.float32(g) * seg
            events.append((t0 / SR, (t0 + L) / SR, item.category, item.tag))
            info.append(dict(sid=item.sid, snr_db=snr, gain=g))

        wav = clean + noise
        peak = max(float(np.abs(wav).max(initial=0.0)), float(np.abs(clean).max(initial=0.0)))
        if peak > c["peak_limit"]:
            f = np.float32(c["peak_limit"] / peak)
            wav, clean = wav * f, clean * f
        order = sorted(range(len(events)), key=lambda i: events[i][:2])
        return dict(wav=wav, clean=clean, events=[events[i] for i in order], info=[info[i] for i in order],
                    duration=n / SR)


# ---------------------------------------------------------------------------------------------- factory

def _path(p) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else ROOT / p


def build_mixer(cfg: dict | None = None, exclude_ids=()) -> SynthMixer:
    """SynthMixer with sources built from cfg['sources'] (only entries with weight > 0).

    exclude_ids: validation clip ids never used as clean speech (added to sources.speech.val_clean.exclude_folds),
    and dropped from the bank too when sources.noise.bank.exclude_heldout is true.
    """
    from synthgen import sources as S
    cfg = deep_merge(load_config(), cfg)
    noise_cfg, speech_cfg, prior = cfg["sources"]["noise"], cfg["sources"]["speech"], cfg.get("tag_freq")
    heldout = set(exclude_ids) | S.ids_in_folds(speech_cfg["val_clean"].get("exclude_folds") or [])
    noise, speech = [], []
    if (b := noise_cfg["bank"])["weight"] > 0:
        noise.append((S.BankSource(_path(b["dir"]), uniform=b["uniform"],
                                   exclude_source_clips=heldout if b.get("exclude_heldout") else ()), b["weight"]))
    if (g := noise_cfg["gold"])["weight"] > 0:
        noise.append((S.GoldEventSource(_path(g["data_dir"]), tiers=g["tiers"], min_dur=g["min_dur"], max_dur=g["max_dur"],
                                        isolation=g["isolation"], tag_prior=prior if g["tag_prior"] else None,
                                        crop_prob=g["crop_prob"]), g["weight"]))
    if (f := noise_cfg["folder"])["weight"] > 0:
        noise.append((S.FolderSource(_path(f["dir"]), _path(f["csv"]), trim_db=f["trim_db"],
                                     tag_prior=prior if f["tag_prior"] else None, crop_prob=f["crop_prob"]), f["weight"]))
    if (v := speech_cfg["val_clean"])["weight"] > 0:
        speech.append((S.ValCleanSource(heldout, splice_prob=v["splice_prob"]), v["weight"]))
    if (s := speech_cfg["gold_speech"])["weight"] > 0:
        speech.append((S.GoldSpeechSource(_path(s["data_dir"]), tiers=s["tiers"], margin=s["margin_s"],
                                          min_len=s["min_len_s"], splice_prob=s["splice_prob"]), s["weight"]))
    return SynthMixer(noise, speech, cfg)
