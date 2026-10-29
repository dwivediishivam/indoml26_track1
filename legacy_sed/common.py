"""Audio, labels, exact leaderboard metrics, and interval postprocessing."""
from __future__ import annotations

import io
import json
import math
import os
import re
import wave
import zipfile
from pathlib import Path

import numpy as np

SR = 16000
FRAME_SEC = 0.02
CATEGORIES = (
    "animal", "vehicle_traffic", "baby_child", "singing_music",
    "phone_signal_alarm", "appliance_machine", "human_non_speech",
)
CATEGORY_INDEX = {name: i + 1 for i, name in enumerate(CATEGORIES)}


def decode_audio(value):
    """Accept a HF audio cell, WAV bytes, or a path; return 16 kHz mono float32."""
    if isinstance(value, dict):
        if value.get("array") is not None:
            audio, rate = np.asarray(value["array"], np.float32), value["sampling_rate"]
        elif value.get("bytes") is not None:
            import soundfile as sf
            audio, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
        elif value.get("path"):
            import soundfile as sf
            audio, rate = sf.read(value["path"], dtype="float32")
        else:
            raise ValueError("audio cell has neither array, bytes, nor path")
    elif isinstance(value, bytes):
        import soundfile as sf
        audio, rate = sf.read(io.BytesIO(value), dtype="float32")
    else:
        import soundfile as sf
        audio, rate = sf.read(str(value), dtype="float32")
    audio = np.asarray(audio, np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[0] > audio.shape[1] else audio.mean(axis=0)
    if audio.ndim != 1:
        raise ValueError(f"expected mono/stereo audio, found shape {audio.shape}")
    if int(rate) != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=int(rate), target_sr=SR)
    return np.nan_to_num(np.ascontiguousarray(audio, dtype=np.float32))


def spans_from_record(record):
    spans = []
    for event in record.get("NoiseSubCategoryTimeStamp") or []:
        try:
            start, end = float(event["start"]), float(event["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            spans.append((max(0.0, start), end, event.get("category", "")))
    return spans


def targets(spans, duration, n_frames=None):
    """20 ms, eight channels: any noise followed by the seven source categories."""
    count = n_frames or max(1, math.ceil(duration / FRAME_SEC))
    out = np.zeros((8, count), np.float32)
    for start, end, category in spans:
        lo = max(0, int(math.floor(start / FRAME_SEC)))
        hi = min(count, int(math.ceil(end / FRAME_SEC)))
        if hi <= lo:
            continue
        out[0, lo:hi] = 1
        idx = CATEGORY_INDEX.get(category)
        if idx is not None:
            out[idx, lo:hi] = 1
    return out


def intervals(prob, threshold=0.5, median=3, min_duration=0.08,
              merge_gap=0.06, pad=0.0, duration=None):
    """Convert frame probabilities to class-agnostic event spans."""
    from scipy.ndimage import median_filter

    bits = np.asarray(prob, dtype=np.float32) >= threshold
    if median > 1:
        bits = median_filter(bits.astype(np.uint8), size=median, mode="nearest").astype(bool)
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    raw = [[float(a * FRAME_SEC), float(b * FRAME_SEC)] for a, b in zip(starts, ends)]
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    result = []
    for a, b in merged:
        if b - a < min_duration:
            continue
        a, b = max(0.0, a - pad), b + pad
        if duration is not None:
            b = min(float(duration), b)
        if b > a:
            result.append((round(a, 3), round(b, 3)))
    return result


def event_counts(reference, predicted):
    candidates = []
    for ri, (ra, rb) in enumerate(reference):
        tolerance = max(0.2 * (rb - ra), 0.05)
        for pi, (pa, pb) in enumerate(predicted):
            if abs(pa - ra) <= tolerance and abs(pb - rb) <= tolerance:
                candidates.append((abs(pa - ra) + abs(pb - rb), ri, pi))
    used_ref, used_pred = set(), set()
    for _, ri, pi in sorted(candidates):
        if ri not in used_ref and pi not in used_pred:
            used_ref.add(ri)
            used_pred.add(pi)
    tp = len(used_ref)
    return tp, len(predicted) - tp, len(reference) - tp


def _event_mask(events, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in events:
        lo = max(0, int(start / 0.01))
        hi = min(n_frames, int(end / 0.01) + 1)
        mask[lo:hi] = True
    return mask


def dice(reference, predicted):
    both = reference + predicted
    if not both:
        return 1.0
    n_frames = int((max(end for _, end in both) + 0.5) / 0.01) + 1
    ref = _event_mask(reference, n_frames)
    pred = _event_mask(predicted, n_frames)
    denom = int(ref.sum()) + int(pred.sum())
    return 1.0 if denom == 0 else 2.0 * np.logical_and(ref, pred).sum() / denom


def score(references, predictions):
    tp = fp = fn = 0
    dice_scores = []
    for clip_id, ref in references.items():
        pred = predictions.get(clip_id, [])
        a, b, c = event_counts(ref, pred)
        tp += a; fp += b; fn += c
        dice_scores.append(dice(ref, pred))
    for clip_id, pred in predictions.items():
        if clip_id not in references:
            fp += len(pred)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    segment_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    return {"combined": f1 + segment_dice, "f1": f1, "dice": segment_dice,
            "tp": tp, "fp": fp, "fn": fn}


class ValidationArchive:
    """Reads the supplied validation ZIP without extracting its 3,196 audio files."""
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        self.records = json.loads(self.zip.read("validation/validationMetadata.json"))
        self.by_id = {Path(r["segmentFileName"]).stem: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("duplicate validation clip IDs")
        members = set(self.zip.namelist())
        for r in self.records:
            sub = "syntheticNoiseAudio" if r["syntheticData"] else "naturalNoisyAudio"
            member = f"validation/{sub}/{r['segmentFileName']}"
            if member not in members:
                raise FileNotFoundError(member)

    def read(self, record):
        sub = "syntheticNoiseAudio" if record["syntheticData"] else "naturalNoisyAudio"
        return decode_audio(self.zip.read(f"validation/{sub}/{record['segmentFileName']}"))

    def references(self, records=None):
        return {Path(r["segmentFileName"]).stem:
                [(a, b) for a, b, _ in spans_from_record(r)]
                for r in (records or self.records)}

    def close(self):
        self.zip.close()


def atomic_torch_save(payload, destination):
    import torch
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
