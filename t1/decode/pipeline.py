"""Shared clip-level prediction used identically by decode/tune.py and assemble/make_submission.py.

Config JSON (work/tuned/<name>.json):
{
  "name": str,
  "caches": [{"name", "format": "legacy"|"contract"|"auto", "val": path, "blind": path}, ...],
  "fusion": {"weights": {cache_name: w >= 0}, "mode": "prob"|"logit"},
  "natural": {"family": str, "params": {...}, "post": {...}},      # decoders.run config
  "synthetic": null | {"family", "params", "post"},                 # null -> same as natural
  "merge": {"bank_on", "synth_thr", "bank_p_thr", "ncc_thr", "margin", "neural_conf",
            "prior_thr", "prior_conf", "gap", "max_events", "prior_mode"},
  "bank": {"val": path, "blind": path},
  "metrics": {...}, "cv": {...}, "search": {...}                    # written by tune.py
}
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from common.contract import BLIND_DIR, FRAME, WORK, n_frames, val_audio_path
from decode import decoders, fusion, merge

BANK_DEFAULT = {"val": str(WORK / "bank" / "val_events.jsonl"), "blind": str(WORK / "bank" / "blind_events.jsonl")}
ROUND = 3


# ----------------------------------------------------------------------------- io helpers
def wav_duration(path):
    """Duration in seconds from the WAV header (soundfile if installed, else RIFF parsing)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except ImportError:
        pass
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            raise ValueError(f"not a WAV file: {path}")
        rate = align = None
        while True:
            ch = f.read(8)
            if len(ch) < 8:
                break
            cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
            if cid == b"fmt ":
                fmt = f.read(size)
                _, nch, rate, _, align = struct.unpack("<HHIIH", fmt[:14])
                f.seek(size % 2, 1)
            elif cid == b"data":
                if rate is None or not align:
                    raise ValueError(f"data before fmt chunk: {path}")
                if size == 0xFFFFFFFF:  # streaming header; use the file size
                    size = os.path.getsize(path) - f.tell()
                return (size // align) / rate
            else:
                f.seek(size + size % 2, 1)
    raise ValueError(f"no data chunk: {path}")


def val_durations(val, use_headers=True):
    out = {}
    for cid, v in val.items():
        d = v["duration"]
        if use_headers:
            p = val_audio_path(cid, v["synthetic"])
            if p.exists():
                try:
                    d = wav_duration(p)
                except Exception:
                    pass
        out[cid] = float(d)
    return out


def blind_durations(ids):
    return {cid: float(wav_duration(BLIND_DIR / f"{cid}.wav")) for cid in ids}


def load_bank(path, log=print):
    path = Path(path)
    if not path.exists():
        log(f"[bank] WARNING {path} not found: every clip is decoded as natural")
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["clip_id"]] = r
    log(f"[bank] {len(out)} records from {path}")
    return out


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def load_config(path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- prediction
def _syn_cfg(cfg):
    syn = cfg.get("synthetic") or cfg["natural"]
    return {**syn, "post": {**syn.get("post", {}), "prior": False}}


def predict(probs, duration, bank_record, cfg):
    """-> (spans [(onset, offset)], route). Never empty, 0 <= onset < offset <= duration."""
    duration = float(duration)
    if probs is None:
        return [(0.0, round(duration, ROUND))], "missing"
    natural = lambda: decoders.run(probs, duration, cfg["natural"])
    synthetic = lambda: decoders.run(probs, duration, _syn_cfg(cfg), fallback=False)
    events, r = merge.combine({"natural": natural, "synthetic": synthetic}, bank_record, cfg.get("merge", {}))
    spans = []
    for on, off, *_ in events:
        a = round(min(max(0.0, float(on)), duration), ROUND)
        b = round(min(max(0.0, float(off)), duration), ROUND)
        if b > a:
            spans.append((a, b))
    if not spans:
        on, off, *_ = decoders.fallback_event(probs, duration, 0.0)
        spans = [(round(on, ROUND), round(max(off, on + FRAME), ROUND))]
    spans.sort()
    return spans, r


def predict_clip(caches, cid, duration, bank_record, cfg):
    probs = fusion.fused_clip(caches, cid, cfg["fusion"], n_frames(duration))
    return predict(probs, duration, bank_record, cfg)
