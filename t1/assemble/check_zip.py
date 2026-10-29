#!/usr/bin/env python3
"""Independent validator for a Track 1 submission ZIP (stdlib only; soundfile optional).

  python assemble/check_zip.py work/submissions/<name>.zip [--audio data/blind/audio] [--expect 5517]

Checks: the archive holds exactly one root-level member named predictions.jsonl; every line is a
JSON object {"clip_id": str, "events": [{"onset": float, "offset": float}, ...]}; the clip ids are
exactly the wav stems of the blind audio dir, each once; no empty event lists; all values are
finite numbers with 0 <= onset < offset <= duration + 0.05 (duration from the WAV header).
Exit status 0 = valid, 1 = invalid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zipfile
from pathlib import Path

DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "data" / "blind" / "audio"


def wav_seconds(path):
    try:
        import soundfile as sf
        i = sf.info(str(path))
        return i.frames / i.samplerate
    except ImportError:
        pass
    with open(path, "rb") as f:
        h = f.read(12)
        if h[:4] not in (b"RIFF", b"RF64") or h[8:12] != b"WAVE":
            raise ValueError("not RIFF/WAVE")
        rate = block = None
        while True:
            c = f.read(8)
            if len(c) < 8:
                raise ValueError("no data chunk")
            name, size = c[:4], struct.unpack("<I", c[4:])[0]
            if name == b"fmt ":
                body = f.read(size)
                rate, block = struct.unpack("<I", body[4:8])[0], struct.unpack("<H", body[12:14])[0]
                f.seek(size & 1, 1)
            elif name == b"data":
                if size == 0xFFFFFFFF:
                    size = os.path.getsize(path) - f.tell()
                return (size // block) / rate
            else:
                f.seek(size + (size & 1), 1)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check(zip_path, audio_dir, expect, tol=0.05):
    errs, warn = [], []
    try:
        z = zipfile.ZipFile(zip_path)
    except Exception as e:
        return [f"cannot open zip: {e}"], warn, {}
    with z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if names != ["predictions.jsonl"]:
            errs.append(f"zip must contain only a root-level predictions.jsonl, found {names}")
        if "predictions.jsonl" not in z.namelist():
            return errs, warn, {}
        raw = z.read("predictions.jsonl").decode("utf-8")
    wavs = {p.stem: p for p in Path(audio_dir).glob("*.wav")}
    if expect and len(wavs) != expect:
        errs.append(f"audio dir has {len(wavs)} wavs, expected {expect}")
    seen, n_ev, counts = set(), 0, {}
    for ln, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            errs.append(f"line {ln}: blank"); continue
        try:
            o = json.loads(line)
        except Exception as e:
            errs.append(f"line {ln}: invalid json: {e}"); continue
        if not isinstance(o, dict) or not isinstance(o.get("clip_id"), str):
            errs.append(f"line {ln}: missing clip_id"); continue
        cid = o["clip_id"]
        if cid in seen:
            errs.append(f"line {ln}: duplicate {cid}"); continue
        seen.add(cid)
        if cid not in wavs:
            errs.append(f"line {ln}: clip_id not in blind audio: {cid}"); continue
        ev = o.get("events")
        if not isinstance(ev, list) or not ev:
            errs.append(f"{cid}: empty/missing events"); continue
        try:
            d = wav_seconds(wavs[cid])
        except Exception as e:
            errs.append(f"{cid}: cannot read wav header: {e}"); continue
        counts[len(ev)] = counts.get(len(ev), 0) + 1
        for e in ev:
            if not isinstance(e, dict) or not is_num(e.get("onset")) or not is_num(e.get("offset")):
                errs.append(f"{cid}: bad event {e}"); continue
            if not (0 <= e["onset"] < e["offset"] <= d + tol):
                errs.append(f"{cid}: out of range {e} (duration {d:.3f})")
            n_ev += 1
            extra = set(e) - {"onset", "offset"}
            if extra:
                warn.append(f"{cid}: extra event keys {sorted(extra)}")
    missing = set(wavs) - seen
    if missing:
        errs.append(f"{len(missing)} blind clips missing, e.g. {sorted(missing)[:3]}")
    return errs, warn, {"lines": len(seen), "events": n_ev, "count_hist": dict(sorted(counts.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--audio", default=str(DEFAULT_AUDIO))
    ap.add_argument("--expect", type=int, default=5517)
    a = ap.parse_args(argv)
    errs, warn, info = check(a.zip, a.audio, a.expect)
    for w in warn[:10]:
        print("WARN", w)
    for e in errs[:50]:
        print("ERROR", e)
    print(f"{a.zip}: {info}")
    print("VALID" if not errs else f"INVALID ({len(errs)} errors)")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
