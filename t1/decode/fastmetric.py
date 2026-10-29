"""Exact per-clip re-implementation of common/metric.py for fast tuning.

clip_stats() returns (tp, n_pred, n_ref, dice) with the same float arithmetic and the same
greedy tie-breaking (sorted (cost, ref_idx, pred_idx)) as metric.match_events / metric.dice.
aggregate() pools them exactly like metric.score.
"""
from __future__ import annotations

import numpy as np


def match_count(ref, pred):
    if not ref or not pred:
        return 0
    r = np.asarray(ref, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    rs, re = r[:, 0:1], r[:, 1:2]
    ps, pe = p[None, :, 0], p[None, :, 1]
    tol = np.maximum(.20 * (re - rs), .05)
    ds, de = np.abs(ps - rs), np.abs(pe - re)
    ri, pi = np.nonzero((ds <= tol) & (de <= tol))
    if len(ri) <= 1:
        return len(ri)
    cost = ds[ri, pi] + de[ri, pi]
    used_r = np.zeros(len(ref), bool)
    used_p = np.zeros(len(pred), bool)
    tp = 0
    for k in np.lexsort((pi, ri, cost)):
        a, b = ri[k], pi[k]
        if not used_r[a] and not used_p[b]:
            used_r[a] = used_p[b] = True
            tp += 1
    return tp


def dice(ref, pred, frame=.01):
    events = list(ref) + list(pred)
    if not events:
        return 1.
    n = int((max(end for _, end in events) + .5) / frame) + 1
    masks = []
    for spans in (ref, pred):
        d = np.zeros(n + 1, np.int32)
        for start, end in spans:
            lo, hi = max(0, int(start / frame)), min(n, int(end / frame) + 1)
            if hi > lo:
                d[lo] += 1; d[hi] -= 1
        masks.append(np.cumsum(d[:n]) > 0)
    total = int(masks[0].sum()) + int(masks[1].sum())
    return 1. if not total else 2 * int((masks[0] & masks[1]).sum()) / total


def clip_stats(ref, pred):
    ref = [(float(a), float(b)) for a, b in ref]
    pred = [(float(a), float(b)) for a, b in pred]
    return match_count(ref, pred), len(pred), len(ref), dice(ref, pred)


def aggregate(tp, fp, fn, dice_sum, n_clips):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    seg = dice_sum / n_clips if n_clips else 0.
    return {"event_f1": f1, "segment_dice": seg, "combined": f1 + seg, "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "clips": int(n_clips)}


def score_arrays(tp, npred, nref, dsc, mask=None):
    """Pool per-clip stat arrays (optionally restricted to a boolean mask)."""
    tp, npred, nref, dsc = (np.asarray(x) for x in (tp, npred, nref, dsc))
    if mask is not None:
        tp, npred, nref, dsc = tp[mask], npred[mask], nref[mask], dsc[mask]
    t = int(tp.sum())
    return aggregate(t, int(npred.sum()) - t, int(nref.sum()) - t, float(np.sum(dsc, dtype=np.float64)), len(dsc))
