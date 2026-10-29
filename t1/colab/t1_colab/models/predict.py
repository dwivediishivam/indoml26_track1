"""Sliding-window batched prediction with optional time-shift TTA (shared by infer.py and the monitor).

Windows: `window_s` long, `hop_s` apart, frame-aligned; overlapping outputs are blended with a
triangular weight floored at `floor`. A TTA shift of k frames (k * 20 ms) shifts the audio, predicts,
and shifts the outputs back; frames lost at the edge are edge-replicated.
"""
from __future__ import annotations

import numpy as np
import torch

from common.contract import N_CH

HOP = 320
SR = 16000


def frames_for(samples: int) -> int:
    return max(1, -(-int(samples) // HOP))


def window_starts(n: int, win: int, hop: int):
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    last = -(-(n - win) // HOP) * HOP           # frame-aligned final window reaching the end
    if starts[-1] < last:
        starts.append(last)
    return starts


def tri_weights(frames: int, floor: float) -> np.ndarray:
    t = (np.arange(frames, dtype=np.float32) + 0.5) / frames
    return np.maximum(floor, 1 - np.abs(2 * t - 1)).astype(np.float32)


def _shifted(wav: np.ndarray, k: int):
    if k > 0:
        return np.concatenate([np.zeros(k * HOP, np.float32), wav])
    if k < 0:
        return wav[-k * HOP:]
    return wav


@torch.inference_mode()
def predict_clips(model, wavs: dict, device, window_s=10.0, hop_s=5.0, floor=0.3, shifts=(0,), batch_size=16,
                  autocast=True) -> dict:
    """wavs: {clip_id: float32 [N]} -> {clip_id: float32 sigmoid probabilities [24, ceil(N/320)]}."""
    win, hop = int(round(window_s * SR)), int(round(hop_s * SR))
    if win % HOP or hop % HOP:
        raise ValueError("window and hop must be multiples of 20 ms")
    was_training = model.training
    model.eval()
    try:
        acc, wsum, jobs = {}, {}, []
        for cid, wav in wavs.items():
            for k in shifts:
                if k < 0 and len(wav) <= -k * HOP + HOP:
                    continue                                   # clip too short for this shift
                w = _shifted(wav, k)
                tw = frames_for(len(w))
                acc[(cid, k)] = np.zeros((N_CH, tw), np.float32)
                wsum[(cid, k)] = np.zeros(tw, np.float32)
                for s in window_starts(len(w), win, hop):
                    jobs.append((cid, k, s, w[s:s + win]))
        jobs.sort(key=lambda j: len(j[3]))
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            n = frames_for(max(len(j[3]) for j in chunk)) * HOP
            x = np.zeros((len(chunk), n), np.float32)
            for r, j in enumerate(chunk):
                x[r, :len(j[3])] = j[3]
            lengths = torch.tensor([len(j[3]) for j in chunk], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
                out = model(torch.from_numpy(x).to(device), lengths)
            prob = torch.sigmoid(out["logits"].float()).cpu().numpy()
            if not np.isfinite(prob).all():
                raise FloatingPointError("non-finite model output during prediction")
            for r, (cid, k, s, piece) in enumerate(chunk):
                f = frames_for(len(piece))
                left = s // HOP
                f = min(f, acc[(cid, k)].shape[1] - left)
                w = tri_weights(f, floor)
                acc[(cid, k)][:, left:left + f] += prob[r, :, :f] * w
                wsum[(cid, k)][left:left + f] += w
        result = {}
        for cid, wav in wavs.items():
            t = frames_for(len(wav))
            total, count = np.zeros((N_CH, t), np.float32), 0
            for k in shifts:
                if (cid, k) not in acc:
                    continue
                ws = wsum[(cid, k)]
                if np.any(ws <= 0):
                    raise RuntimeError(f"{cid}: sliding windows left frames uncovered")
                p = acc[(cid, k)] / ws[None]
                if k > 0:
                    p = p[:, k:k + t]
                elif k < 0:
                    p = np.concatenate([np.repeat(p[:, :1], -k, axis=1), p], axis=1)[:, :t]
                if p.shape[1] != t:
                    raise AssertionError(f"{cid}: TTA alignment produced {p.shape[1]} frames, expected {t}")
                total += p
                count += 1
            if count == 0:
                raise RuntimeError(f"{cid}: no prediction produced")
            result[cid] = total / count
        return result
    finally:
        model.train(was_training)
