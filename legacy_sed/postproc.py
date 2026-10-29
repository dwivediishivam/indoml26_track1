"""Event decoding from frame probabilities, plus the official metric.

Decoder features beyond the original `intervals()`:
  * hysteresis - a high threshold triggers an event, a low threshold sustains it
  * valley splitting - one blob is cut at internal minima so adjacent events
    are not merged into a single long span (event F1 punishes merging hard)
  * asymmetric onset/offset padding
  * per-clip probability normalisation (optional)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FRAME = 0.02

DEFAULT = dict(hi=0.55, lo=0.35, smooth=1, median=1, min_dur=0.10,
               merge_gap=0.00, pad_on=0.0, pad_off=0.0, split_drop=0.0,
               split_min=0.30, norm=0.0, max_dur=0.0)


def decode(prob, duration, hi=0.55, lo=0.35, smooth=1, median=1, min_dur=0.10,
           merge_gap=0.0, pad_on=0.0, pad_off=0.0, split_drop=0.0,
           split_min=0.30, norm=0.0, max_dur=0.0):
    p = np.asarray(prob, dtype=np.float32)
    if norm > 0:
        mx = float(p.max()) if p.size else 0.0
        if mx > 1e-6:
            p = (1.0 - norm) * p + norm * (p / mx)
    if smooth > 1:
        p = uniform_filter1d(p, size=int(smooth), mode="nearest")
    if median > 1:
        p = median_filter(p, size=int(median), mode="nearest")

    n = p.size
    above_lo = p >= lo
    # connected components of above_lo that contain at least one frame >= hi
    edges = np.diff(np.r_[False, above_lo, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    segs = []
    for a, b in zip(starts, ends):
        if p[a:b].max() >= hi:
            segs.append([a, b])

    if merge_gap > 0:
        gap = int(round(merge_gap / FRAME))
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] <= gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        segs = merged

    if split_drop > 0:
        split_len = max(1, int(round(split_min / FRAME)))
        out = []
        for a, b in segs:
            cur = a
            i = a + split_len
            while i < b - split_len:
                lo_i = i
                # local minimum inside the blob deep enough below both peaks
                left_peak = p[cur:i].max()
                right_peak = p[i:b].max()
                if p[lo_i] <= min(left_peak, right_peak) - split_drop and \
                        p[lo_i] <= p[max(a, lo_i - 1)] and p[lo_i] <= p[min(b - 1, lo_i + 1)]:
                    out.append([cur, lo_i])
                    cur = lo_i
                    i = lo_i + split_len
                else:
                    i += 1
            out.append([cur, b])
        segs = out

    result = []
    for a, b in segs:
        s, e = a * FRAME, b * FRAME
        if e - s < min_dur:
            continue
        if max_dur > 0 and e - s > max_dur:
            e = s + max_dur
        s = max(0.0, s - pad_on)
        e = e + pad_off
        if duration is not None:
            e = min(float(duration), e)
            s = min(s, max(0.0, float(duration) - 0.01))
        if e > s:
            result.append((round(float(s), 3), round(float(e), 3)))
    return result


# ---------------------------------------------------------------- metric
def event_counts(reference, predicted):
    candidates = []
    for ri, (ra, rb) in enumerate(reference):
        tol = max(0.2 * (rb - ra), 0.05)
        for pi, (pa, pb) in enumerate(predicted):
            if abs(pa - ra) <= tol and abs(pb - rb) <= tol:
                candidates.append((abs(pa - ra) + abs(pb - rb), ri, pi))
    used_ref, used_pred = set(), set()
    for _, ri, pi in sorted(candidates):
        if ri not in used_ref and pi not in used_pred:
            used_ref.add(ri); used_pred.add(pi)
    tp = len(used_ref)
    return tp, len(predicted) - tp, len(reference) - tp


def _mask(events, n):
    m = np.zeros(n, dtype=bool)
    for s, e in events:
        m[max(0, int(s / 0.01)): min(n, int(e / 0.01) + 1)] = True
    return m


def dice(reference, predicted):
    both = list(reference) + list(predicted)
    if not both:
        return 1.0
    n = int((max(e for _, e in both) + 0.5) / 0.01) + 1
    r, p = _mask(reference, n), _mask(predicted, n)
    denom = int(r.sum()) + int(p.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(r, p).sum()) / denom


def score(references, predictions):
    tp = fp = fn = 0
    dices = []
    for cid, ref in references.items():
        pred = predictions.get(cid, [])
        a, b, c = event_counts(ref, pred)
        tp += a; fp += b; fn += c
        dices.append(dice(ref, pred))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    sd = float(np.mean(dices)) if dices else 0.0
    return {"combined": f1 + sd, "f1": f1, "dice": sd, "tp": tp, "fp": fp, "fn": fn,
            "clips": len(references)}
