"""Event decoders over the 24-channel contract cache (pure numpy, scipy optional).

Every decoder is ``f(probs[24, T], duration, params) -> [(onset, offset, category_or_None, score)]``.
``run(probs, duration, dcfg)`` = decoder + common post-steps, with
``dcfg = {"family": str, "params": {...}, "post": {...}}``.

Families
  legacy                   proven ai2 decoder (decode/legacy.py)
  hysteresis_per_category  per-track median + hi/lo hysteresis + gap merge + min-dur + biases,
                           union over category tracks and the union track, cross-track dedup
  csebb                    clean-room change-detection bounding boxes (step filter change points,
                           dip-significance merging, mean-posterior confidence, detection threshold)
  peaks                    onset/offset peak pairing from the boundary channels, validated by
                           the presence integral, parabolic sub-frame refinement

Post (all optional): edge_tol, min_score, min_len, fallback_len, prior (bool), prior_gap,
prior_max, prior_mode ("drop" | "merge").
"""
from __future__ import annotations

import numpy as np

from common.contract import CATEGORIES, CH_CAT, CH_CAT_OFF, CH_CAT_ON, CH_OFF, CH_ON, CH_UNION, FRAME
from decode import legacy as _legacy

try:
    from scipy.ndimage import median_filter as _sp_median
except Exception:  # pragma: no cover
    _sp_median = None

N_CAT = len(CATEGORIES)
TRACK_NAMES = (None,) + tuple(CATEGORIES)  # track id 0 = union, 1..7 = categories


# ----------------------------------------------------------------------------- primitives
def median_smooth(x, k):
    """Median filter along the last axis, mode='nearest'. Even k is rounded up to odd."""
    k = int(k)
    if k <= 1 or x.shape[-1] < 2:
        return x
    if k % 2 == 0:
        k += 1
    if _sp_median is not None:
        return _sp_median(x, size=(1,) * (x.ndim - 1) + (k,), mode="nearest")
    h = k // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(h, h)], mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=-1)
    return np.median(win, axis=-1).astype(x.dtype, copy=False)


def channel_ok(x):
    """True per row if the channel is present (no NaN)."""
    return np.isfinite(x).all(axis=-1)


def union_track(probs):
    u = probs[CH_UNION]
    if np.isfinite(u).all():
        return u
    c = probs[CH_CAT]
    ok = channel_ok(c)
    if ok.any():
        return c[ok].max(axis=0)
    return np.zeros(probs.shape[1], np.float32)


def runs(mask):
    """mask [K, T] bool -> (rows, starts, ends) of True runs, row-major order, end exclusive."""
    K, T = mask.shape
    m = np.zeros((K, T + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1)
    r, s = np.nonzero(d == 1)
    _, e = np.nonzero(d == -1)
    return r, s, e


def seg_max(x, rows, s, e):
    if len(s) == 0:
        return np.zeros(0, np.float32)
    K, T = x.shape
    flat = np.concatenate([x.ravel(), np.array([-np.inf], x.dtype)])
    idx = np.empty(2 * len(s), np.int64)
    idx[0::2] = rows * T + s
    idx[1::2] = rows * T + e
    return np.maximum.reduceat(flat, idx)[0::2]


def cumsum0(x):
    """[K, T] -> [K, T+1] with a leading zero column (float64 for exact-ish means)."""
    out = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,), np.float64)
    np.cumsum(x, axis=-1, out=out[..., 1:])
    return out


def seg_mean(cs, rows, s, e):
    if len(s) == 0:
        return np.zeros(0, np.float64)
    return (cs[rows, e] - cs[rows, s]) / np.maximum(e - s, 1)


def hysteresis(x, hi, lo):
    K = x.shape[0]
    hi = np.broadcast_to(np.asarray(hi, np.float32), (K,))
    lo = np.minimum(np.broadcast_to(np.asarray(lo, np.float32), (K,)), hi)
    r, s, e = runs(x >= lo[:, None])
    if len(s) == 0:
        return r, s, e
    keep = seg_max(x, r, s, e) >= hi[r]
    return r[keep], s[keep], e[keep]


def merge_gaps(r, s, e, gap_frames):
    """Merge consecutive runs of the same row separated by fewer than gap_frames frames."""
    if len(s) < 2 or gap_frames <= 0:
        return r, s, e
    new = np.ones(len(s), bool)
    new[1:] = (r[1:] != r[:-1]) | ((s[1:] - e[:-1]) >= gap_frames)
    first = np.flatnonzero(new)
    last = np.r_[first[1:], len(s)] - 1
    return r[first], s[first], e[last]


def iou_matrix(on, off):
    inter = np.clip(np.minimum(off[:, None], off[None]) - np.maximum(on[:, None], on[None]), 0, None)
    union = np.maximum(off[:, None], off[None]) - np.minimum(on[:, None], on[None])
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(on, off, score, thr):
    """Greedy suppression by score; returns kept indices. Pairs with IoU > thr are duplicates."""
    n = len(on)
    order = np.argsort(-np.asarray(score), kind="stable")
    if n <= 1:
        return order
    iou = iou_matrix(np.asarray(on, float), np.asarray(off, float))
    sup = np.zeros(n, bool)
    keep = []
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        sup |= iou[i] > thr
    return np.asarray(keep, int)


def parabolic(y, idx):
    """Sub-sample offset in [-0.5, 0.5] of the extremum at y[..., idx] (rows given by idx tuple)."""
    r, i = idx
    T = y.shape[1]
    il, ir = np.clip(i - 1, 0, T - 1), np.clip(i + 1, 0, T - 1)
    a, b, c = y[r, il], y[r, i], y[r, ir]
    den = a - 2 * b + c
    d = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    d = np.where((i == 0) | (i == T - 1), 0.0, d)
    return np.clip(d, -0.5, 0.5)


def local_max(y, thr):
    """[K, T] -> (rows, cols) of local maxima (>= left, > right) with value >= thr."""
    K, T = y.shape
    yp = np.full((K, T + 2), -np.inf, y.dtype)
    yp[:, 1:-1] = y
    m = (y >= yp[:, :-2]) & (y > yp[:, 2:]) & (y >= thr)
    return np.nonzero(m)


def _bias_arrays(p):
    """Per-track (union + 7 categories) onset/offset biases in seconds."""
    on = np.full(N_CAT + 1, float(p.get("on_bias", 0.0)))
    off = np.full(N_CAT + 1, float(p.get("off_bias", 0.0)))
    on[0] += float(p.get("union_on_bias", 0.0))
    off[0] += float(p.get("union_off_bias", 0.0))
    if p.get("cat_on_bias") is not None:
        on[1:] += np.asarray(p["cat_on_bias"], float)
    if p.get("cat_off_bias") is not None:
        off[1:] += np.asarray(p["cat_off_bias"], float)
    return on, off


def _tracks(probs, p, need=None):
    """Stack presence tracks: union (id 0) and present categories (ids 1..7)."""
    ids = []
    rows = []
    cat = probs[CH_CAT]
    cok = channel_ok(cat)
    if need is not None:
        cok = cok & need
    use_cat = bool(p.get("use_cat", True)) and cok.any()
    use_union = bool(p.get("use_union", True)) or not use_cat
    if use_union:
        ids.append(0); rows.append(union_track(probs))
    if use_cat:
        for k in np.flatnonzero(cok):
            ids.append(int(k) + 1); rows.append(cat[k])
    return np.asarray(ids, int), np.stack(rows).astype(np.float32, copy=False)


def _finalize(tid, on, off, score, T, p):
    """Union-cover filter for union-track events, then cross-track dedup (IoU > dedup_iou)."""
    n = len(on)
    if n == 0:
        return []
    tid = np.asarray(tid, int); on = np.asarray(on, float); off = np.asarray(off, float)
    score = np.asarray(score, float)
    keep = off > on
    uc = float(p.get("union_cover", 1.01))
    is_u = (tid == 0) & keep
    is_c = (tid > 0) & keep
    if uc <= 1.0 and is_u.any() and is_c.any():
        cov = np.zeros(T + 1, np.int32)
        a = np.clip((on[is_c] / FRAME).astype(int), 0, T)
        b = np.clip(np.ceil(off[is_c] / FRAME - 1e-6).astype(int), 0, T)
        np.add.at(cov, a, 1); np.add.at(cov, b, -1)
        cs = np.concatenate([[0], np.cumsum(np.cumsum(cov)[:T] > 0)])
        ui = np.flatnonzero(is_u)
        ua = np.clip((on[ui] / FRAME).astype(int), 0, T)
        ub = np.clip(np.ceil(off[ui] / FRAME - 1e-6).astype(int), 0, T)
        frac = (cs[ub] - cs[ua]) / np.maximum(ub - ua, 1)
        keep[ui[frac >= uc]] = False
    idx = np.flatnonzero(keep)
    di = float(p.get("dedup_iou", 0.8))
    if di < 1.0 and len(idx) > 1:
        idx = idx[nms(on[idx], off[idx], score[idx], di)]
    idx = sorted(idx, key=lambda i: (on[i], off[i]))
    return [(float(on[i]), float(off[i]), TRACK_NAMES[tid[i]], float(score[i])) for i in idx]


# ----------------------------------------------------------------------------- decoders
def legacy(probs, duration, params):
    kw = {k: v for k, v in params.items() if k in _legacy.PARAMS}
    return _legacy.decode(probs, duration, **kw)


def hysteresis_per_category(probs, duration, p):
    """params: median, hi, lo_delta, cat_hi, cat_lo_delta, cat_hi_delta[7], gate, gap, min_dur,
    on_bias, off_bias, union_on_bias, union_off_bias, cat_on_bias[7], cat_off_bias[7],
    use_union, use_cat, union_cover, dedup_iou."""
    tid, X = _tracks(probs, p)
    T = X.shape[1]
    X = median_smooth(X, p.get("median", 1))
    gate = float(p.get("gate", 0.0))
    if gate > 0 and (tid > 0).any():
        u = X[0] if tid[0] == 0 else median_smooth(union_track(probs)[None], p.get("median", 1))[0]
        X = X.copy()
        X[tid > 0] *= (u >= gate)
    hi_u = float(p.get("hi", 0.5)); lo_u = max(0.01, hi_u - float(p.get("lo_delta", 0.1)))
    hi_c = np.full(N_CAT, float(p.get("cat_hi", hi_u)))
    if p.get("cat_hi_delta") is not None:
        hi_c = hi_c + np.asarray(p["cat_hi_delta"], float)
    hi_c = np.clip(hi_c, 0.02, 1.0)
    lo_c = np.maximum(0.01, hi_c - float(p.get("cat_lo_delta", p.get("lo_delta", 0.1))))
    hi = np.where(tid == 0, hi_u, hi_c[np.maximum(tid - 1, 0)])
    lo = np.where(tid == 0, lo_u, lo_c[np.maximum(tid - 1, 0)])
    r, s, e = hysteresis(X, hi, lo)
    r, s, e = merge_gaps(r, s, e, float(p.get("gap", 0.0)) / FRAME)
    if len(s) == 0:
        return []
    sc = seg_mean(cumsum0(X), r, s, e)
    bon, boff = _bias_arrays(p)
    t = tid[r]
    on = np.where(s == 0, 0.0, s * FRAME + bon[t])
    off = np.where(e >= T, duration, e * FRAME + boff[t])
    keep = (off - on) >= float(p.get("min_dur", 0.0))
    return _finalize(t[keep], on[keep], off[keep], sc[keep], T, p)


def csebb(probs, duration, p):
    """Clean-room change-detection sound event bounding boxes.

    params: tau (s), median, pad ("zero"|"edge"), min_step, seg_thr, gamma_abs, gamma_rel,
    thr (union confidence), cat_thr (category confidence), min_dur, refine, on_bias, off_bias,
    union_on_bias, union_off_bias, cat_on_bias, cat_off_bias, use_union, use_cat, union_cover,
    dedup_iou.

    1. Step filter: delta[b] = mean(x[b:b+h]) - mean(x[b-h:b]) at every frame boundary b in
       [0, T] (h = tau/2 in frames; outside the clip is zero or edge-padded).
    2. Change points: local maxima of delta >= min_step (onsets), local minima <= -min_step
       (offsets), plus the clip edges; optional parabolic sub-frame refinement.
    3. Elementary segments between consecutive change points, level = mean posterior.
       Segments below seg_thr are gaps. An interior local-minimum segment is a separating dip
       only if it lies >= gamma_abs below both neighbouring maxima AND <= (1 - gamma_rel) of
       them; otherwise it is merged into its neighbours.
    4. Boxes = maximal runs of non-gap segments; confidence = mean posterior inside; keep boxes
       with confidence >= threshold (thr for the union track, cat_thr for categories).
    """
    tid, X = _tracks(probs, p)
    X = median_smooth(X, p.get("median", 1)).astype(np.float64)
    K, T = X.shape
    h = max(1, int(round(float(p.get("tau", 0.12)) / (2 * FRAME))))
    mode = "edge" if p.get("pad", "zero") == "edge" else "constant"
    xp = np.pad(X, ((0, 0), (h, h)), mode=mode)
    cs = cumsum0(xp)
    j = np.arange(T + 1) + h
    delta = (cs[:, j + h] - 2 * cs[:, j] + cs[:, j - h]) / h  # mean(next h) - mean(prev h)
    ms = float(p.get("min_step", 0.05))
    on_r, on_c = local_max(delta, ms)
    of_r, of_c = local_max(-delta, ms)
    refine = bool(p.get("refine", True))
    don = parabolic(delta, (on_r, on_c)) if refine else np.zeros(len(on_c))
    dof = parabolic(-delta, (of_r, of_c)) if refine else np.zeros(len(of_c))
    csx = cumsum0(X)
    seg_thr = float(p.get("seg_thr", 0.3))
    g_abs = float(p.get("gamma_abs", 0.2)); g_rel = float(p.get("gamma_rel", 0.0))
    thr_u = float(p.get("thr", 0.5)); thr_c = float(p.get("cat_thr", thr_u))
    min_dur = float(p.get("min_dur", 0.0))
    bon, boff = _bias_arrays(p)
    o_tid, o_on, o_off, o_sc = [], [], [], []
    for k in range(K):
        # change points (frame boundary index, refined time)
        cpi = np.concatenate([[0, T], on_c[on_r == k], of_c[of_r == k]])
        cpt = np.concatenate([[0.0, T], on_c[on_r == k] + don[on_r == k], of_c[of_r == k] + dof[of_r == k]])
        order = np.argsort(cpi, kind="stable")
        cpi, cpt = cpi[order], cpt[order]
        uniq = np.r_[True, np.diff(cpi) > 0]
        cpi, cpt = cpi[uniq], cpt[uniq]
        a, b = cpi[:-1], cpi[1:]
        if len(a) == 0:
            continue
        lev = (csx[k, b] - csx[k, a]) / np.maximum(b - a, 1)
        gap = lev < seg_thr
        n = len(lev)
        if n >= 3:
            interior = np.zeros(n, bool)
            interior[1:-1] = (lev[1:-1] < lev[:-2]) & (lev[1:-1] < lev[2:]) & ~gap[1:-1] & ~gap[:-2] & ~gap[2:]
            for i in np.flatnonzero(interior):
                lft = i - 1
                while lft - 1 >= 0 and not gap[lft - 1] and lev[lft - 1] >= lev[lft]:
                    lft -= 1
                rgt = i + 1
                while rgt + 1 < n and not gap[rgt + 1] and lev[rgt + 1] >= lev[rgt]:
                    rgt += 1
                peak = min(lev[lft:i].max(), lev[i + 1:rgt + 1].max())  # neighbouring maxima
                if peak - lev[i] >= g_abs and lev[i] <= (1.0 - g_rel) * peak:
                    gap[i] = True
        rr, rs, re = runs(~gap[None])
        if len(rs) == 0:
            continue
        fa, fb = a[rs], b[re - 1]
        conf = (csx[k, fb] - csx[k, fa]) / np.maximum(fb - fa, 1)
        thr = thr_u if tid[k] == 0 else thr_c
        ok = conf >= thr
        if not ok.any():
            continue
        t_on, t_off = cpt[rs][ok], cpt[re][ok]
        fa, fb, conf = fa[ok], fb[ok], conf[ok]
        on = np.where(fa == 0, 0.0, t_on * FRAME + bon[tid[k]])
        off = np.where(fb >= T, duration, t_off * FRAME + boff[tid[k]])
        good = (off - on) >= min_dur
        o_tid.append(np.full(good.sum(), tid[k])); o_on.append(on[good]); o_off.append(off[good])
        o_sc.append(conf[good])
    if not o_on:
        return []
    return _finalize(np.concatenate(o_tid), np.concatenate(o_on), np.concatenate(o_off),
                     np.concatenate(o_sc), T, p)


def peaks(probs, duration, p):
    """Onset/offset peak pairing.

    params: on_thr, off_thr, pres_thr, score_thr, alpha, max_next, min_dur, max_dur, nms_iou,
    edge_pres, on_shift, off_shift, bmedian, pmedian, refine, use_union, use_cat,
    fill_hi, fill_lo_delta, fill_cover, union_cover, dedup_iou.

    Per track (category tracks if per-category boundary channels exist, else the union):
    onset/offset candidates are local maxima of the boundary channels (sub-frame refined by
    parabolic interpolation; time = (i + 0.5 + d) * FRAME + shift); clip-edge onsets/offsets are
    implicit when presence at the edge >= edge_pres. Pairs (onset, one of the next max_next
    offsets) with presence mean >= pres_thr are scored (1-alpha)*presence + alpha*sqrt(on*off)
    and selected greedily one-to-one with in-track NMS. Optional fill: union hysteresis blobs not
    covered by any pair (coverage < fill_cover) are added.
    """
    T = probs.shape[1]
    cat_b = channel_ok(probs[CH_CAT_ON]) & channel_ok(probs[CH_CAT_OFF]) & channel_ok(probs[CH_CAT])
    union_b = bool(np.isfinite(probs[CH_ON]).all() and np.isfinite(probs[CH_OFF]).all())
    tracks = []  # (tid, presence, onset, offset)
    if bool(p.get("use_cat", True)) and cat_b.any():
        for k in np.flatnonzero(cat_b):
            tracks.append((int(k) + 1, probs[1 + k], probs[CH_CAT_ON][k], probs[CH_CAT_OFF][k]))
    if union_b and (bool(p.get("use_union", False)) or not tracks):
        tracks.insert(0, (0, union_track(probs), probs[CH_ON], probs[CH_OFF]))
    if not tracks:
        return []
    tid = np.array([t[0] for t in tracks])
    PR = median_smooth(np.stack([t[1] for t in tracks]).astype(np.float32), p.get("pmedian", 1))
    ON = median_smooth(np.stack([t[2] for t in tracks]).astype(np.float32), p.get("bmedian", 1))
    OF = median_smooth(np.stack([t[3] for t in tracks]).astype(np.float32), p.get("bmedian", 1))
    csp = cumsum0(PR)
    refine = bool(p.get("refine", True))
    or_, oc = local_max(ON, float(p.get("on_thr", 0.2)))
    fr_, fc = local_max(OF, float(p.get("off_thr", 0.2)))
    od = parabolic(ON, (or_, oc)) if refine else np.zeros(len(oc))
    fd = parabolic(OF, (fr_, fc)) if refine else np.zeros(len(fc))
    ot = (oc + 0.5 + od) * FRAME + float(p.get("on_shift", 0.0))
    ft = (fc + 0.5 + fd) * FRAME + float(p.get("off_shift", 0.0))
    ov, fv = ON[or_, oc], OF[fr_, fc]
    edge = float(p.get("edge_pres", 0.6))
    pres_thr = float(p.get("pres_thr", 0.4)); score_thr = float(p.get("score_thr", 0.0))
    alpha = float(p.get("alpha", 0.5)); max_next = int(p.get("max_next", 2))
    min_dur = float(p.get("min_dur", 0.04)); max_dur = float(p.get("max_dur", 30.0))
    nms_iou = float(p.get("nms_iou", 0.5))
    o_tid, o_on, o_off, o_sc = [], [], [], []
    for k in range(len(tracks)):
        t_on, v_on = ot[or_ == k], ov[or_ == k]
        t_of, v_of = ft[fr_ == k], fv[fr_ == k]
        if PR[k, 0] >= edge:
            t_on, v_on = np.r_[0.0, t_on], np.r_[PR[k, 0], v_on]
        if PR[k, -1] >= edge:
            t_of, v_of = np.r_[t_of, duration], np.r_[v_of, PR[k, -1]]
        if len(t_on) == 0 or len(t_of) == 0:
            continue
        o_order, f_order = np.argsort(t_on), np.argsort(t_of)
        t_on, v_on, t_of, v_of = t_on[o_order], v_on[o_order], t_of[f_order], v_of[f_order]
        D = t_of[None, :] - t_on[:, None]
        valid = (D >= max(min_dur, 1e-3)) & (D <= max_dur)
        valid &= np.cumsum(valid, axis=1) <= max_next
        oi, fi = np.nonzero(valid)
        if len(oi) == 0:
            continue
        fa = np.clip((t_on[oi] / FRAME).astype(int), 0, T - 1)
        fb = np.clip(np.ceil(t_of[fi] / FRAME - 1e-6).astype(int), fa + 1, T)
        pm = (csp[k, fb] - csp[k, fa]) / (fb - fa)
        sc = (1 - alpha) * pm + alpha * np.sqrt(v_on[oi] * v_of[fi])
        ok = (pm >= pres_thr) & (sc >= score_thr)
        oi, fi, sc = oi[ok], fi[ok], sc[ok]
        if len(oi) == 0:
            continue
        order = np.argsort(-sc, kind="stable")
        uo, uf = set(), set()
        acc = []
        for q in order:
            if oi[q] in uo or fi[q] in uf:
                continue
            a, b = t_on[oi[q]], t_of[fi[q]]
            if any(min(b, y) - max(a, x) > nms_iou * (max(b, y) - min(a, x)) for x, y, _ in acc):
                continue
            uo.add(oi[q]); uf.add(fi[q]); acc.append((a, b, sc[q]))
        if acc:
            o_tid.append(np.full(len(acc), tid[k])); o_on.append(np.array([x[0] for x in acc]))
            o_off.append(np.array([x[1] for x in acc])); o_sc.append(np.array([x[2] for x in acc]))
    fill_hi = float(p.get("fill_hi", 1.01))
    if fill_hi <= 1.0:
        u = median_smooth(union_track(probs)[None].astype(np.float32), p.get("pmedian", 1))
        r, s, e = hysteresis(u, fill_hi, max(0.01, fill_hi - float(p.get("fill_lo_delta", 0.1))))
        if len(s):
            cov = np.zeros(T + 1, np.int32)
            for arr_on, arr_off in zip(o_on, o_off):
                a = np.clip((arr_on / FRAME).astype(int), 0, T)
                b = np.clip(np.ceil(arr_off / FRAME - 1e-6).astype(int), 0, T)
                np.add.at(cov, a, 1); np.add.at(cov, b, -1)
            cc = np.concatenate([[0], np.cumsum(np.cumsum(cov)[:T] > 0)])
            frac = (cc[e] - cc[s]) / np.maximum(e - s, 1)
            m = frac < float(p.get("fill_cover", 0.5))
            if m.any():
                s, e = s[m], e[m]
                o_tid.append(np.zeros(len(s), int)); o_on.append(s * FRAME)
                o_off.append(np.where(e >= T, duration, e * FRAME))
                o_sc.append(seg_mean(cumsum0(u), np.zeros(len(s), int), s, e))
    if not o_on:
        return []
    return _finalize(np.concatenate(o_tid), np.concatenate(o_on), np.concatenate(o_off),
                     np.concatenate(o_sc), T, p)


FAMILIES = {
    "legacy": legacy,
    "hysteresis_per_category": hysteresis_per_category,
    "csebb": csebb,
    "peaks": peaks,
}


# ----------------------------------------------------------------------------- post-steps
def fallback_event(probs, duration, length=1.0):
    """Max-mean window of the union track (length <= 0 or >= clip -> the whole clip)."""
    u = np.nan_to_num(union_track(probs).astype(np.float64))
    T = len(u)
    w = T if length <= 0 else int(max(1, min(T, round(length / FRAME))))
    cs = np.concatenate([[0.0], np.cumsum(u)])
    means = (cs[w:] - cs[:-w]) / w
    i = int(np.argmax(means))
    return (i * FRAME, min(float(duration), (i + w) * FRAME), None, float(means[i]))


def synthetic_prior(events, gap=0.2, max_events=5, exclude=(), margin=0.0, mode="drop"):
    """No overlaps, gaps >= gap, at most max_events, nothing within `margin` of `exclude` spans.

    mode="drop": greedy by score, conflicting events are dropped.
    mode="merge": time-ordered events closer than `gap` are first merged (score = max).
    """
    ev = [tuple(e) for e in events]
    if max_events <= 0 or not ev:
        return []
    if mode == "merge":
        ev.sort(key=lambda e: (e[0], e[1]))
        merged = [list(ev[0])]
        for e in ev[1:]:
            cur = merged[-1]
            if e[0] < cur[1] + gap:
                cur[1] = max(cur[1], e[1])
                if e[2] != cur[2]:
                    cur[2] = None
                cur[3] = max(cur[3], e[3])
            else:
                merged.append(list(e))
        ev = [tuple(m) for m in merged]
    blocks = [(float(a) - margin, float(b) + margin) for a, b, *_ in exclude]
    acc = []
    for e in sorted(ev, key=lambda e: -e[3]):
        if len(acc) >= max_events:
            break
        if any(e[0] < b and e[1] > a for a, b in blocks):
            continue
        if any(e[0] < x[1] + gap and e[1] > x[0] - gap for x in acc):
            continue
        acc.append(e)
    return sorted(acc, key=lambda e: (e[0], e[1]))


def postprocess(events, probs, duration, post=None, fallback=True):
    post = post or {}
    tol = float(post.get("edge_tol", 0.0))
    min_score = float(post.get("min_score", 0.0))
    min_len = float(post.get("min_len", 0.005))
    out = []
    for on, off, cat, sc in events:
        if sc < min_score:
            continue
        if on < tol:
            on = 0.0
        if off > duration - tol:
            off = duration
        on, off = max(0.0, float(on)), min(float(duration), float(off))
        if off - on >= min_len:
            out.append((on, off, cat, float(sc)))
    if post.get("prior", False):
        out = synthetic_prior(out, float(post.get("prior_gap", 0.2)), int(post.get("prior_max", 5)),
                              mode=post.get("prior_mode", "drop"))
    if not out and fallback:
        out = [fallback_event(probs, duration, float(post.get("fallback_len", 1.0)))]
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def run(probs, duration, dcfg, fallback=True):
    fam = dcfg.get("family", "legacy")
    if fam not in FAMILIES:
        raise ValueError(f"unknown decoder family {fam!r}; choose from {sorted(FAMILIES)}")
    events = FAMILIES[fam](probs, float(duration), dcfg.get("params", {}))
    return postprocess(events, probs, float(duration), dcfg.get("post", {}), fallback=fallback)
