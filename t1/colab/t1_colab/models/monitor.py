"""Fast fixed-decoder monitor on a fixed validation subset, scored with common.metric.score.

The decoder is deliberately simple and frozen so checkpoint rankings are comparable across runs;
real decoding is the decode/ workstream's job.
"""
from __future__ import annotations

import hashlib

import numpy as np

from common import contract
from common.contract import FRAME
from common.metric import score
from models.data import read_wav
from models.predict import predict_clips


def monitor_ids(folds, max_clips=0, exclude=()):
    """Deterministic monitor subset from the given folds (natural/synthetic proportional when capped)."""
    val = contract.load_validation()
    fold_of = contract.folds()
    folds = {int(f) for f in folds}
    ids = sorted(k for k in val if fold_of.get(k) in folds and k not in set(exclude))
    if not ids:
        raise RuntimeError(f"no monitor clips in folds {sorted(folds)}")
    if max_clips and max_clips < len(ids):
        key = lambda x: hashlib.sha256(x.encode()).hexdigest()
        nat = sorted([k for k in ids if not val[k]["synthetic"]], key=key)
        syn = sorted([k for k in ids if val[k]["synthetic"]], key=key)
        n_nat = round(max_clips * len(nat) / len(ids))
        ids = sorted(nat[:n_nat] + syn[:max_clips - n_nat])
    return ids, val


def load_monitor_audio(ids, val):
    return {k: read_wav(contract.val_audio_path(k, val[k]["synthetic"])) for k in ids}


def _runs(x: np.ndarray):
    if not x.any():
        return []
    d = np.diff(np.concatenate([[0], x.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close_gaps(x: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 0:
        return x
    x = x.copy()
    runs = _runs(x)
    for (a0, b0), (a1, _) in zip(runs[:-1], runs[1:]):
        if a1 - b0 <= gap:
            x[b0:a1] = True
    return x


def simple_decode(prob: np.ndarray, duration: float, threshold=0.5, min_frames=3, gap_frames=3, dup_iou=0.7):
    """Per-category tracks (channels 1..7) + union-only fallback (channel 0), duplicate suppression."""
    if not np.isfinite(prob[:8]).all():
        raise ValueError("monitor decoder received non-finite presence probabilities")
    cand, covered = [], np.zeros(prob.shape[1], bool)
    for c in range(1, 8):
        for a, b in _runs(_close_gaps(prob[c] >= threshold, gap_frames)):
            if b - a >= min_frames:
                cand.append((float(prob[c, a:b].mean()), a, b))
                covered[a:b] = True
    for a, b in _runs(_close_gaps(prob[0] >= threshold, gap_frames)):
        if b - a >= min_frames and covered[a:b].mean() < 0.5:
            cand.append((float(prob[0, a:b].mean()), a, b))
    kept = []
    for conf, a, b in sorted(cand, reverse=True):
        dup = False
        for _, a2, b2 in kept:
            inter = max(0, min(b, b2) - max(a, a2))
            if inter / (max(b, b2) - min(a, a2)) >= dup_iou:
                dup = True
                break
        if not dup:
            kept.append((conf, a, b))
    return sorted((a * FRAME, min(duration, b * FRAME)) for _, a, b in kept if a * FRAME < duration)


def evaluate(model, wavs, val, device, mcfg):
    probs = predict_clips(model, wavs, device, mcfg["window_seconds"], mcfg["hop_seconds"], 0.3, (0,), mcfg["batch_size"])
    pred = {k: simple_decode(p, val[k]["duration"], mcfg["threshold"], mcfg["min_frames"], mcfg["gap_frames"],
                             mcfg["dup_iou"]) for k, p in probs.items()}
    refs = {k: [(a, b) for a, b, *_ in val[k]["events"]] for k in wavs}
    out = score(refs, pred)
    nat = [k for k in wavs if not val[k]["synthetic"]]
    syn = [k for k in wavs if val[k]["synthetic"]]
    if nat:
        out["natural"] = score({k: refs[k] for k in nat}, {k: pred[k] for k in nat})["combined"]
    if syn:
        out["synthetic"] = score({k: refs[k] for k in syn}, {k: pred[k] for k in syn})["combined"]
    out["clips"] = len(wavs)
    return out
