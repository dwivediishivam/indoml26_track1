"""Matched-filter detection of bank snippets inside a clip.

A synthetic clip is clean speech plus sample-exact copies of bank snippets, so a true match has:
  * a high normalised cross-correlation (NCC) over the whole snippet, reduced only by speech overlap;
  * exact lag agreement of every 50 ms sub-chunk with the whole-snippet lag (chunk consistency), which
    random look-alikes in natural audio do not have;
  * a mixing gain inside the organizers' SNR range.
Candidates carry these features; a logistic model fitted on validation (bank/calibrate.py) turns
them into a match probability p. Events truncated by the clip edges are found with head/tail
sub-templates.
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, find_peaks

from common.contract import SR, WORK

BANK_DIR = WORK / "bank"
CHUNK = 800            # 50 ms verification chunks
LAG = 16               # +-1 ms lag search per chunk
EDGE = int(0.25 * SR)  # head/tail sub-template length for clip-edge truncation
MIN_NCC = 0.35       # worst-case speech overlap (SNR +4 dB) still gives NCC ~0.5
TOP_K = 3            # a snippet can recur within one clip
RMS_FLOOR = 1e-4       # -80 dBFS: silence must not produce spurious NCC
FEATURES = ("ncc", "cons", "ncc_x_cons", "snr", "log_dur", "edge")


def read_wav(path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr}, expected {SR}")
    return (x if x.ndim == 1 else x.mean(1)).astype(np.float64)


class Bank:
    def __init__(self, waves: dict, meta: list):
        self.meta = {m["id"]: m for m in meta if m["id"] in waves}
        self.ids = sorted(self.meta)
        self.wave = {i: np.asarray(waves[i], dtype=np.float64) for i in self.ids}
        self.norm = {i: float(np.linalg.norm(self.wave[i])) for i in self.ids}

    @classmethod
    def load(cls, folder=BANK_DIR):
        with np.load(folder / "snippets.npz") as z:
            waves = {k: z[k] for k in z.files}
        return cls(waves, json.loads((folder / "snippets.json").read_text()))

    def without_fold(self, fold: int, folds: dict) -> "Bank":
        """Bank as if fold `fold` had never been seen: keep snippets used by some clip outside the fold."""
        keep = [m for m in self.meta.values() if any(folds.get(c) != fold for c in m["source_clips"])]
        return Bank({m["id"]: self.wave[m["id"]] for m in keep}, keep)


def _energy(x2c, L):
    return np.sqrt(np.maximum(x2c[L:] - x2c[:-L], (RMS_FLOOR ** 2) * L))


def ncc_curve(x, x2c, s, s_norm):
    """NCC[t] = <x[t:t+L], s> / (||x[t:t+L]|| ||s||) for every full-overlap position t."""
    L = len(s)
    if L > len(x):
        return None
    return fftconvolve(x, s[::-1], mode="valid") / (_energy(x2c, L) * s_norm)


def chunk_consistency(x, s, t):
    """Energy-weighted fraction of 50 ms chunks whose best local lag equals the snippet lag (+-1 sample)."""
    ok = tot = 0.0
    for j in range(0, len(s) - CHUNK + 1, CHUNK):
        sc = s[j:j + CHUNK]
        en = float(sc @ sc)
        lo, hi = t + j - LAG, t + j + LAG + CHUNK
        if en <= 0 or lo < 0 or hi > len(x):
            continue
        win = np.lib.stride_tricks.sliding_window_view(x[lo:hi], CHUNK)
        nc = (win @ sc) / (np.sqrt(np.maximum((win * win).sum(1), (RMS_FLOOR ** 2) * CHUNK)) * np.sqrt(en))
        k = int(np.argmax(nc))
        tot += en
        if abs(k - LAG) <= 1 and nc[k] > 0.1:
            ok += en
    return ok / tot if tot else 0.0


def _features(x, clip_power, seg_s, t, ncc, edge):
    L = len(seg_s)
    gain = float(x[t:t + L] @ seg_s) / max(float(seg_s @ seg_s), 1e-12)
    noise_power = gain * gain * float(seg_s @ seg_s) / L
    snr = 10 * np.log10(max(clip_power, 1e-12) / max(noise_power, 1e-12))
    cons = chunk_consistency(x, seg_s, t)
    return dict(ncc=float(ncc), cons=float(cons), ncc_x_cons=float(ncc * cons), snr=float(np.clip(snr, -30, 30)),
                log_dur=float(np.log(L / SR)), edge=float(edge), gain=gain)


def candidates(x: np.ndarray, bank: Bank) -> list[dict]:
    """All candidate matches (before scoring/NMS): dict(snippet, onset, offset, kind, features...)."""
    n = len(x)
    x2c = np.concatenate([[0.0], np.cumsum(x * x)])
    clip_power = float(np.mean(x * x)) if n else 0.0
    out = []

    def add(sid, kind, t, seg_s, ncc, onset_sample, offset_sample):
        f = _features(x, clip_power, seg_s, t, ncc, kind != "full")
        out.append(dict(snippet=sid, kind=kind, onset=onset_sample / SR, offset=offset_sample / SR, **f))

    for sid in bank.ids:
        s, L = bank.wave[sid], len(bank.wave[sid])
        r = ncc_curve(x, x2c, s, bank.norm[sid])
        if r is not None and len(r):
            peaks, prop = find_peaks(r, height=MIN_NCC, distance=max(1, L // 2))
            for t in peaks[np.argsort(-prop["peak_heights"])[:TOP_K]]:
                add(sid, "full", int(t), s, r[t], int(t), int(t) + L)
        if L <= EDGE + CHUNK or n < EDGE:
            continue
        # head sub-template: snippet starts at t but would run past the clip end (t + L > n)
        head = s[:EDGE]
        rh = ncc_curve(x, x2c, head, float(np.linalg.norm(head)))
        lo = max(0, n - L + 1)
        if rh is not None and lo < len(rh):
            seg = rh[lo:]
            peaks, prop = find_peaks(seg, height=MIN_NCC)
            for p in peaks[np.argsort(-prop["peak_heights"])[:TOP_K]]:
                t = lo + int(p)
                inside = s[:n - t]
                ncc = float(x[t:n] @ inside) / (max(np.linalg.norm(x[t:n]), RMS_FLOOR * np.sqrt(n - t)) * np.linalg.norm(inside))
                add(sid, "head", t, inside, ncc, t, n)
        # tail sub-template: snippet ends at u + EDGE but started before the clip (u < L - EDGE)
        tail = s[L - EDGE:]
        rt = ncc_curve(x, x2c, tail, float(np.linalg.norm(tail)))
        hi = min(len(rt) if rt is not None else 0, L - EDGE)
        if rt is not None and hi > 0:
            peaks, prop = find_peaks(rt[:hi], height=MIN_NCC)
            for u in peaks[np.argsort(-prop["peak_heights"])[:TOP_K]]:
                u = int(u)
                end = u + EDGE
                inside = s[L - end:]
                ncc = float(x[:end] @ inside) / (max(np.linalg.norm(x[:end]), RMS_FLOOR * np.sqrt(end)) * np.linalg.norm(inside))
                add(sid, "tail", 0, inside, ncc, 0, end)
    return out


def feature_matrix(cands) -> np.ndarray:
    return np.array([[c[k] for k in FEATURES] for c in cands], dtype=np.float64).reshape(-1, len(FEATURES))


def score(cands, model: dict) -> None:
    """Add calibrated probability p to every candidate in place. model: {mean, std, w} from calibrate.py."""
    if not cands:
        return
    X = (feature_matrix(cands) - np.asarray(model["mean"])) / np.asarray(model["std"])
    z = np.asarray(model["w"][0]) + X @ np.asarray(model["w"][1:])
    for c, p in zip(cands, 1 / (1 + np.exp(-np.clip(z, -40, 40)))):
        c["p"] = float(p)


def nms(cands, min_p=0.05, min_gap=0.0) -> list[dict]:
    """Greedy by p: synthetic events never overlap, so any overlap (or gap < min_gap) with a kept event is dropped.
    Thresholding the result at any p >= min_p equals running NMS after that threshold."""
    kept = []
    for c in sorted((c for c in cands if c["p"] >= min_p), key=lambda c: -c["p"]):
        if all(c["offset"] + min_gap <= k["onset"] or c["onset"] >= k["offset"] + min_gap for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c["onset"])


def clip_record(cid, kept, bank: Bank, duration=None) -> dict:
    events = []
    for c in kept:
        m = bank.meta[c["snippet"]]
        off = c["offset"] if duration is None else min(c["offset"], duration)
        events.append(dict(onset=round(c["onset"], 4), offset=round(off, 4), tag=m["tag"], category=m["category"],
                           snippet=c["snippet"], kind=c["kind"], ncc=round(c["ncc"], 4), cons=round(c["cons"], 4),
                           p=round(c["p"], 5)))
    return dict(clip_id=cid, synthetic_score=max((e["p"] for e in events), default=0.0), events=events)
