"""Frame targets in the shared 24-channel contract layout (see common/contract.py).

Used by both real-data loaders and the synthetic mixer so labels are identical everywhere.
"""
from __future__ import annotations

import numpy as np

from common.contract import CATEGORIES, CH_CAT, CH_CAT_OFF, CH_CAT_ON, CH_OFF, CH_ON, CH_UNION, FRAME, N_CH, n_frames

CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


def _bump(n: int, center: float, sigma: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) + 0.5
    return np.exp(-0.5 * ((t - center) / sigma) ** 2).astype(np.float32)


def frame_targets(events, duration: float, sigma_frames: float = 1.0, n: int | None = None,
                  clip_edge_boundaries: bool = False) -> np.ndarray:
    """events: iterable of (onset_s, offset_s, category, *rest). Returns float32 [24, T].

    Presence: frame i is active if it overlaps [onset, offset).
    Boundaries: Gaussian bump (sigma in frames) centred on the exact boundary time.
    Boundaries that coincide with the clip edges (onset <= 0 or offset >= duration) are
    usually truncation artefacts of the segmenter, so they get no boundary target unless
    clip_edge_boundaries=True.
    """
    n = n or n_frames(duration)
    y = np.zeros((N_CH, n), dtype=np.float32)
    cat_on, cat_off = y[CH_CAT_ON], y[CH_CAT_OFF]
    for ev in events:
        on, off, cat = float(ev[0]), float(ev[1]), ev[2]
        if off <= on:
            continue
        lo, hi = max(0, int(np.floor(on / FRAME))), min(n, int(np.ceil(off / FRAME)))
        if hi <= lo:
            continue
        ci = CAT_INDEX.get(cat)
        y[CH_UNION, lo:hi] = 1.0
        if ci is not None:
            y[CH_CAT][ci, lo:hi] = 1.0
        if clip_edge_boundaries or on > FRAME / 2:
            b = _bump(n, on / FRAME, sigma_frames)
            y[CH_ON] = np.maximum(y[CH_ON], b)
            if ci is not None:
                cat_on[ci] = np.maximum(cat_on[ci], b)
        if clip_edge_boundaries or off < duration - FRAME / 2:
            b = _bump(n, off / FRAME, sigma_frames)
            y[CH_OFF] = np.maximum(y[CH_OFF], b)
            if ci is not None:
                cat_off[ci] = np.maximum(cat_off[ci], b)
    return y


def clip_categories(events) -> np.ndarray:
    """Multi-hot [7] clip-level category vector (for Bronze MIL supervision)."""
    v = np.zeros(len(CATEGORIES), dtype=np.float32)
    for ev in events:
        ci = CAT_INDEX.get(ev[2] if not isinstance(ev, str) else ev)
        if ci is not None:
            v[ci] = 1.0
    return v
