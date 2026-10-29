"""Multi-track event decoder for the SED probability stack.

Input array P has shape (1 + 7 + n_tags + 2, frames):
  P[0]      binary "any noise" track
  P[1:8]    the seven noise categories
  P[8:-2]   per-tag tracks
  P[-2:]    onset / offset heatmaps

Why tracks matter: in the natural validation subset every pair of adjacent
reference events (gap < 50 ms, 13% of all events) carries a *different* tag.
A single binary track fuses them into one blob and loses both events under the
tolerance-based F1, so blobs are re-split using the auxiliary tracks.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FRAME = 0.02

DEFAULT = dict(
    hi=0.55, lo=0.40, smooth=1, median=1, min_dur=0.10, merge_gap=0.0,
    pad_on=0.0, pad_off=0.0, norm=0.0, split_drop=0.0, split_min=0.30,
    # multi-track split
    track_thr=1.1,        # >1 disables track splitting
    track_min=0.10,       # shortest track sub-segment kept, seconds
    track_cover=0.55,     # sub-segments must cover this much of the blob
    track_iou=0.6,        # dedup threshold between sub-segments
    use_cat=1.0,          # 1 = include the 7 category tracks
    use_tag=1.0,          # 1 = include the per-tag tracks
    # boundary refinement from the onset/offset heatmaps
    bref=0,               # search radius in frames, 0 = off
    bref_thr=0.30,
)


def _blobs(p, hi, lo):
    above = p >= lo
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    out = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if p[a:b].max() >= hi:
            out.append([int(a), int(b)])
    return out


def _valley_split(p, segs, drop, min_len):
    if drop <= 0:
        return segs
    out = []
    for a, b in segs:
        cur, i = a, a + min_len
        while i < b - min_len:
            if p[i] <= min(p[cur:i].max(), p[i:b].max()) - drop and \
               p[i] <= p[i - 1] and p[i] <= p[i + 1]:
                out.append([cur, i]); cur = i; i = cur + min_len
            else:
                i += 1
        out.append([cur, b])
    return out


def _iou(x, y):
    inter = max(0, min(x[1], y[1]) - max(x[0], y[0]))
    union = max(x[1], y[1]) - min(x[0], y[0])
    return inter / union if union else 0.0


def decode(P, duration, hi=0.55, lo=0.40, smooth=1, median=1, min_dur=0.10,
           merge_gap=0.0, pad_on=0.0, pad_off=0.0, norm=0.0, split_drop=0.0,
           split_min=0.30, track_thr=1.1, track_min=0.10, track_cover=0.55,
           track_iou=0.6, use_cat=1.0, use_tag=1.0, bref=0, bref_thr=0.30):
    P = np.asarray(P, dtype=np.float32)
    base = P[0].copy()
    on, off = P[-2], P[-1]
    lo_aux, hi_aux = 1, P.shape[0] - 2
    tracks = []
    if use_cat >= 0.5:
        tracks.append(P[1:8])
    if use_tag >= 0.5 and hi_aux > 8:
        tracks.append(P[8:hi_aux])
    aux = np.concatenate(tracks, 0) if tracks else np.zeros((0, base.size), np.float32)

    if norm > 0:
        mx = float(base.max())
        if mx > 1e-6:
            base = (1 - norm) * base + norm * (base / mx)
    if smooth > 1:
        base = uniform_filter1d(base, size=int(smooth), mode="nearest")
    if median > 1:
        base = median_filter(base, size=int(median), mode="nearest")

    segs = _blobs(base, hi, lo)
    if merge_gap > 0:
        gap = int(round(merge_gap / FRAME))
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] <= gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        segs = merged
    segs = _valley_split(base, segs, split_drop, max(1, int(round(split_min / FRAME))))

    if track_thr <= 1.0 and aux.shape[0]:
        tmin = max(1, int(round(track_min / FRAME)))
        out = []
        for a, b in segs:
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
                        sub.append([a + int(s0), a + int(s1), float(pk[s0:s1].mean())])
            if len(sub) < 2:
                out.append([a, b]); continue
            sub.sort(key=lambda s: -s[2])
            keep = []
            for s in sub:
                if all(_iou(s, k) < track_iou for k in keep):
                    keep.append(s)
            keep.sort()
            covered = np.zeros(b - a, bool)
            for s in keep:
                covered[s[0] - a: s[1] - a] = True
            if len(keep) >= 2 and covered.mean() >= track_cover:
                out.extend([[s[0], s[1]] for s in keep])
            else:
                out.append([a, b])
        segs = out

    if bref:
        r = int(bref)
        n = base.size
        ref = []
        for a, b in segs:
            la, lb = max(0, a - r), min(n, a + r + 1)
            if lb > la and on[la:lb].max() >= bref_thr:
                a = la + int(np.argmax(on[la:lb]))
            ra, rb = max(0, b - 1 - r), min(n, b + r)
            if rb > ra and off[ra:rb].max() >= bref_thr:
                b = ra + int(np.argmax(off[ra:rb])) + 1
            if b > a:
                ref.append([a, b])
        segs = ref

    result = []
    for a, b in segs:
        s, e = a * FRAME, b * FRAME
        if e - s < min_dur:
            continue
        s = max(0.0, s - pad_on)
        e = e + pad_off
        if duration is not None:
            e = min(float(duration), e)
            s = min(s, max(0.0, float(duration) - 0.01))
        if e > s:
            result.append((round(float(s), 3), round(float(e), 3)))
    result.sort()
    return result
