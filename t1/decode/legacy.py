"""Proven ai2 multi-track legacy decoder, adapted to the 24-channel contract layout.

Port of Phase 2/phase2/recovery/legacy_decoder.py. Semantics are unchanged, except that:
  * it reads union / category / union-onset / union-offset from the contract channels
    (per-tag tracks do not exist in the contract, so `use_tag` is accepted and ignored);
  * NaN (missing) channels are skipped (no categories -> no track split; no boundaries -> no bref);
  * it returns (onset, offset, category_or_None, score) with score = mean track posterior.

Proven setting (cache_audit_9016 decoder_search.json best.decoder, 0.8/0.2 s9000/s1600 blend,
combined 1.1413 on 2,501 validation clips) is BEST below.
"""
from __future__ import annotations

import numpy as np

from common.contract import CATEGORIES, CH_CAT, CH_OFF, CH_ON, CH_UNION, FRAME

try:  # scipy is optional; numpy fallbacks reproduce mode="nearest" for odd sizes
    from scipy.ndimage import median_filter as _sp_median, uniform_filter1d as _sp_uniform
except Exception:  # pragma: no cover
    _sp_median = _sp_uniform = None

BEST = dict(hi=0.9, lo=0.9, smooth=1, median=3, min_dur=0.0, merge_gap=0.08, pad_on=-0.02, pad_off=-0.02,
            track_thr=0.25, track_min=0.24, track_cover=0.45, track_iou=0.45, use_cat=1.0, use_tag=0.0,
            bref=5, bref_thr=0.15)

DEFAULT = dict(
    hi=0.55, lo=0.40, smooth=1, median=1, min_dur=0.10, merge_gap=0.0,
    pad_on=0.0, pad_off=0.0, norm=0.0, split_drop=0.0, split_min=0.30,
    track_thr=1.1, track_min=0.10, track_cover=0.55, track_iou=0.6, use_cat=1.0, use_tag=1.0,
    bref=0, bref_thr=0.30,
)
PARAMS = tuple(DEFAULT)


def _median1d(x, k):
    k = int(k)
    if k <= 1:
        return x
    if _sp_median is not None:
        return _sp_median(x, size=k, mode="nearest")
    if k % 2 == 0:  # scipy picks the upper median for even sizes; emulate on an odd window
        k += 1
    h = k // 2
    xp = np.pad(x, (h, h), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=-1).astype(np.float32)


def _uniform1d(x, k):
    k = int(k)
    if k <= 1:
        return x
    if _sp_uniform is not None:
        return _sp_uniform(x, size=k, mode="nearest")
    left = k // 2
    right = k - 1 - left
    xp = np.pad(x.astype(np.float64), (left, right), mode="edge")
    cs = np.concatenate([[0.0], np.cumsum(xp)])
    return ((cs[k:] - cs[:-k]) / k).astype(np.float32)


def _blobs(p, hi, lo):
    above = p >= lo
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    out = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if p[a:b].max() >= hi:
            out.append([int(a), int(b), -1])
    return out


def _valley_split(p, segs, drop, min_len):
    if drop <= 0:
        return segs
    out = []
    for a, b, k in segs:
        cur, i = a, a + min_len
        while i < b - min_len:
            if p[i] <= min(p[cur:i].max(), p[i:b].max()) - drop and p[i] <= p[i - 1] and p[i] <= p[i + 1]:
                out.append([cur, i, k]); cur = i; i = cur + min_len
            else:
                i += 1
        out.append([cur, b, k])
    return out


def _iou(x, y):
    inter = max(0, min(x[1], y[1]) - max(x[0], y[0]))
    union = max(x[1], y[1]) - min(x[0], y[0])
    return inter / union if union else 0.0


def decode(P, duration, hi=0.55, lo=0.40, smooth=1, median=1, min_dur=0.10,
           merge_gap=0.0, pad_on=0.0, pad_off=0.0, norm=0.0, split_drop=0.0,
           split_min=0.30, track_thr=1.1, track_min=0.10, track_cover=0.55,
           track_iou=0.6, use_cat=1.0, use_tag=1.0, bref=0, bref_thr=0.30):
    """P: contract array [24, T]. Returns sorted [(onset, offset, category_or_None, score)]."""
    P = np.asarray(P, dtype=np.float32)
    base = P[CH_UNION].copy()
    cats = P[CH_CAT]
    cat_ok = np.isfinite(cats).all(axis=1)
    if not np.isfinite(base).all():
        base = np.nanmax(cats, axis=0) if cat_ok.any() else np.zeros(P.shape[1], np.float32)
        base = np.nan_to_num(base).astype(np.float32)
    on, off = P[CH_ON], P[CH_OFF]
    have_bounds = bool(np.isfinite(on).all() and np.isfinite(off).all())
    if use_cat >= 0.5 and cat_ok.any():
        aux, aux_ids = cats[cat_ok], np.flatnonzero(cat_ok)
    else:
        aux, aux_ids = np.zeros((0, base.size), np.float32), np.zeros(0, int)

    if norm > 0:
        mx = float(base.max())
        if mx > 1e-6:
            base = (1 - norm) * base + norm * (base / mx)
    if smooth > 1:
        base = _uniform1d(base, smooth)
    if median > 1:
        base = _median1d(base, median)

    segs = _blobs(base, hi, lo)
    if merge_gap > 0:
        gap = int(round(merge_gap / FRAME))
        merged = []
        for a, b, k in segs:
            if merged and a - merged[-1][1] <= gap:
                merged[-1][1] = b
            else:
                merged.append([a, b, k])
        segs = merged
    segs = _valley_split(base, segs, split_drop, max(1, int(round(split_min / FRAME))))

    if track_thr <= 1.0 and aux.shape[0]:
        tmin = max(1, int(round(track_min / FRAME)))
        out = []
        for a, b, k0 in segs:
            sub = []
            window = aux[:, a:b]
            for k in range(window.shape[0]):
                pk = window[k]
                if pk.max() < track_thr:
                    continue
                m = pk >= track_thr
                e = np.diff(np.r_[False, m, False].astype(np.int8))
                for s0, s1 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
                    if s1 - s0 >= tmin:
                        sub.append([a + int(s0), a + int(s1), float(pk[s0:s1].mean()), int(aux_ids[k])])
            if len(sub) < 2:
                out.append([a, b, k0]); continue
            sub.sort(key=lambda s: -s[2])
            keep = []
            for s in sub:
                if all(_iou(s, kk) < track_iou for kk in keep):
                    keep.append(s)
            keep.sort()
            covered = np.zeros(b - a, bool)
            for s in keep:
                covered[s[0] - a: s[1] - a] = True
            if len(keep) >= 2 and covered.mean() >= track_cover:
                out.extend([[s[0], s[1], s[3]] for s in keep])
            else:
                out.append([a, b, k0])
        segs = out

    if bref and have_bounds:
        r = int(bref)
        n = base.size
        ref = []
        for a, b, k in segs:
            la, lb = max(0, a - r), min(n, a + r + 1)
            if lb > la and on[la:lb].max() >= bref_thr:
                a = la + int(np.argmax(on[la:lb]))
            ra, rb = max(0, b - 1 - r), min(n, b + r)
            if rb > ra and off[ra:rb].max() >= bref_thr:
                b = ra + int(np.argmax(off[ra:rb])) + 1
            if b > a:
                ref.append([a, b, k])
        segs = ref

    result = []
    for a, b, k in segs:
        s, e = a * FRAME, b * FRAME
        if e - s < min_dur:
            continue
        track = base if k < 0 else cats[k]
        score = float(track[a:max(b, a + 1)].mean())
        s = max(0.0, s - pad_on)
        e = e + pad_off
        if duration is not None:
            e = min(float(duration), e)
            s = min(s, max(0.0, float(duration) - 0.01))
        if e > s:
            result.append((round(float(s), 3), round(float(e), 3), None if k < 0 else CATEGORIES[k], score))
    result.sort(key=lambda x: (x[0], x[1]))
    return result
