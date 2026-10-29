"""Cache loading and per-clip weighted fusion on the 20 ms grid.

Cache spec (as stored in tuned configs):
  {"name": "s9000", "format": "legacy" | "contract" | "auto", "val": path, "blind": path}
legacy = ai2 74-channel npz (converted with common.contract.from_legacy); contract = 24-channel
work/caches/<model>/{val,blind}.npz. Keys are clip ids (a trailing ".wav" is stripped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.contract import N_CH, WORK, clip_id, from_legacy

LEGACY_DIR = Path("/Users/dwivediishivam/code/indoml/ai2/work")
LEGACY_DEFAULT = {"s9000": 0.8, "s1600": 0.2}


def legacy_spec(name, val=None, blind=None):
    """name 's9000' -> probs_s9000.npz / probs_blind_s9000.npz in the ai2 work dir."""
    tag = name[1:] if name.startswith("s") and name[1:].isdigit() else name
    return {"name": name, "format": "legacy",
            "val": str(val or LEGACY_DIR / f"probs_s{tag}.npz"),
            "blind": str(blind or LEGACY_DIR / f"probs_blind_s{tag}.npz")}


def model_spec(name, val=None, blind=None):
    return {"name": name, "format": "contract",
            "val": str(val or WORK / "caches" / name / "val.npz"),
            "blind": str(blind or WORK / "caches" / name / "blind.npz")}


def discover_models():
    root = WORK / "caches"
    return sorted(p.parent.name for p in root.glob("*/val.npz")) if root.exists() else []


def load_npz(path, fmt="auto", ids=None, dtype=np.float32):
    """-> {clip_id: float32 [24, T]}; ids restricts which keys are converted."""
    try:
        return _load_npz(path, fmt, ids, dtype, allow_pickle=False)
    except ValueError as e:
        if "pickle" not in str(e).lower():
            raise
        return _load_npz(path, fmt, ids, dtype, allow_pickle=True)  # trusted local legacy files only


def _load_npz(path, fmt, ids, dtype, allow_pickle):
    out = {}
    want = None if ids is None else set(ids)
    with np.load(path, allow_pickle=allow_pickle) as z:
        for k in z.files:
            cid = clip_id(k)
            if want is not None and cid not in want:
                continue
            a = np.asarray(z[k], dtype=np.float32)
            if fmt == "legacy" or (fmt == "auto" and a.shape[0] != N_CH):
                a = from_legacy(a)
            if a.ndim != 2 or a.shape[0] != N_CH:
                raise ValueError(f"{path}:{k}: expected [{N_CH}, T], got {a.shape}")
            out[cid] = a.astype(dtype, copy=False)
    return out


def load_caches(specs, split, ids=None, overrides=None, log=print):
    """specs: list of cache specs; split 'val' | 'blind'. Returns {name: {cid: array}}."""
    overrides = overrides or {}
    caches = {}
    for s in specs:
        path = overrides.get(s["name"], s[split])
        log(f"[fusion] loading {s['name']} ({s.get('format', 'auto')}) {split}: {path}")
        caches[s["name"]] = load_npz(path, s.get("format", "auto"), ids)
        if ids is not None:
            miss = len(set(ids) - set(caches[s["name"]]))
            if miss:
                log(f"[fusion] WARNING {s['name']}: {miss} clips missing (excluded from its average)")
    return caches


def align(x, n):
    """Truncate or edge-pad [C, T] to [C, n]."""
    T = x.shape[1]
    if T == n:
        return x
    if T > n:
        return x[:, :n]
    return np.concatenate([x, np.repeat(x[:, -1:], n - T, axis=1)], axis=1)


def fuse(arrays, weights, n, mode="prob", eps=1e-4):
    """Weighted average of [24, T_i] arrays on an n-frame grid.

    mode 'prob' averages probabilities, 'logit' averages logits (then sigmoid). NaN entries
    (missing channels) are excluded from that channel's average; all-missing stays NaN.
    """
    pairs = [(a, float(w)) for a, w in zip(arrays, weights) if a is not None and w > 0]
    if not pairs:
        return np.full((N_CH, n), np.nan, np.float32)
    if len(pairs) == 1:
        return align(pairs[0][0], n)
    X = np.stack([align(a, n) for a, _ in pairs])
    w = np.array([w for _, w in pairs], np.float32)[:, None, None]
    valid = np.isfinite(X)
    if mode == "logit":
        Xc = np.clip(np.where(valid, X, 0.5), eps, 1 - eps)
        Z = np.log(Xc) - np.log1p(-Xc)
    else:
        Z = np.where(valid, X, 0.0)
    W = np.where(valid, w, 0.0)
    den = W.sum(0)
    out = (Z * W).sum(0) / np.where(den > 0, den, 1.0)
    if mode == "logit":
        out = 1.0 / (1.0 + np.exp(-out))
    out = out.astype(np.float32)
    out[den <= 0] = np.nan
    return out


def fused_clip(caches, cid, fusion, n):
    """fusion = {"weights": {name: w}, "mode": "prob"|"logit"}; returns [24, n] or None."""
    weights = fusion.get("weights", {})
    names = [k for k, w in weights.items() if w > 0 and k in caches]
    arrays = [caches[k].get(cid) for k in names]
    if all(a is None for a in arrays):
        return None
    return fuse(arrays, [weights[k] for k in names], n, fusion.get("mode", "prob"))
