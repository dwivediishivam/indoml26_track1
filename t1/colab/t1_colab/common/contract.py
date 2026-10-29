"""Shared contract for every t1_final workstream. Do not change without updating all modules.

Paths
-----
ROOT/data/val/validation/{validationMetadata.json,naturalNoisyAudio,syntheticNoiseAudio,syntheticCleanRefAudio}
ROOT/data/blind/audio/*.wav            5,517 blind clips (16 kHz mono)
ROOT/work/folds.json                    clip_id -> fold 0..4 (grouped by district, stratified by synthetic)
ROOT/work/caches/<model>/{val,blind}.npz  one key per clip_id (no .wav), float16 [24, T]
ROOT/work/bank/{val_events.jsonl,blind_events.jsonl}  template-matched synthetic events (see bank/)

Cache layout (20 ms frames, frame i covers [0.02*i, 0.02*(i+1)) s, T = ceil(duration / 0.02)):
  0        union presence
  1..7     category presence, order = CATEGORIES
  8, 9     union onset, union offset (peak-shaped boundary probabilities)
  10..16   category onset (CATEGORIES order)
  17..23   category offset (CATEGORIES order)
Missing channels are NaN (e.g. legacy caches have no per-category boundaries).
All values are probabilities in [0, 1].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
FRAME = 0.02
CATEGORIES = ("animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech")
CH_UNION, CH_CAT, CH_ON, CH_OFF, CH_CAT_ON, CH_CAT_OFF = 0, slice(1, 8), 8, 9, slice(10, 17), slice(17, 24)
N_CH = 24
VAL_DIR = ROOT / "data/val/validation"
BLIND_DIR = ROOT / "data/blind/audio"
WORK = ROOT / "work"


def clip_id(name: str) -> str:
    return name[:-4] if name.endswith(".wav") else name


def load_validation():
    """Return {clip_id: dict(duration, synthetic, district, language, events=[(on, off, category, tag)])}."""
    meta = json.loads((VAL_DIR / "validationMetadata.json").read_text())
    out = {}
    for x in meta:
        ev = [(float(e["start"]), float(e["end"]), e["category"], e["tag"]) for e in x["NoiseSubCategoryTimeStamp"]]
        out[clip_id(x["segmentFileName"])] = dict(duration=float(x["duration"]), synthetic=bool(x["syntheticData"]),
                                                  district=x["district"], language=x["language"], events=ev)
    return out


def val_audio_path(cid: str, synthetic: bool, clean: bool = False) -> Path:
    sub = "syntheticCleanRefAudio" if clean else ("syntheticNoiseAudio" if synthetic else "naturalNoisyAudio")
    return VAL_DIR / sub / f"{cid}.wav"


def blind_ids():
    return sorted(clip_id(p.name) for p in BLIND_DIR.glob("*.wav"))


def references(val=None, subset=None):
    """Class-agnostic reference spans for metric.score; subset in {None, 'natural', 'synthetic'}."""
    val = val or load_validation()
    keep = lambda v: subset is None or (subset == "synthetic") == v["synthetic"]
    return {k: [(a, b) for a, b, *_ in v["events"]] for k, v in val.items() if keep(v)}


def n_frames(duration: float) -> int:
    return max(1, math.ceil(duration / FRAME))


def save_cache(path, items: dict):
    for k, v in items.items():
        if v.ndim != 2 or v.shape[0] != N_CH:
            raise ValueError(f"{k}: expected [{N_CH}, T], got {v.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial.npz")
    np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in items.items()})
    tmp.replace(path)


def load_cache(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def from_legacy(p: np.ndarray) -> np.ndarray:
    """ai2 legacy 74-channel cache [activity, 7 categories, 64 tags, onset, offset] -> contract layout."""
    p = np.asarray(p, dtype=np.float32)
    out = np.full((N_CH, p.shape[1]), np.nan, dtype=np.float32)
    out[CH_UNION] = p[0]; out[CH_CAT] = p[1:8]; out[CH_ON] = p[-2]; out[CH_OFF] = p[-1]
    return out


def folds():
    return json.loads((WORK / "folds.json").read_text())
