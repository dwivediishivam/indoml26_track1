"""Build the organizers' synthetic noise-snippet bank from validation synthetic clips.

For every synthetic validation clip, noisy - clean is exactly zero outside the labelled spans
(measured), so each labelled span isolates one noise snippet, scaled by its mixing gain.
Snippets reused across clips are clustered: the bank keeps one unit-RMS representative per
cluster, plus usage counts and source clips (needed for leave-fold-out evaluation).

Usage: PYTHONPATH=. python -m bank.build_bank
Writes work/bank/snippets.npz, snippets.json, val_event_map.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from common.contract import SR, WORK, load_validation, val_audio_path

OUT = WORK / "bank"
FP_CORR, FP_DUR = 0.999, 0.002
CONTAIN_NCC = 0.98


def read_int(path):
    x, sr = sf.read(path, dtype="int16", always_2d=False)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr}")
    return x.astype(np.int32) if x.ndim == 1 else x.mean(1).astype(np.int32)


def fingerprint(s):
    win = np.hanning(512)
    frames = [np.abs(np.fft.rfft(s[i:i + 512] * win)) for i in range(0, max(1, len(s) - 512), 256)]
    v = np.log(np.mean(frames, 0) + 1e-8)
    return (v - v.mean()) / (v.std() + 1e-8)


def extract():
    """Return list of dict(clip, idx, tag, category, start, wave) for every synthetic validation event."""
    val, items = load_validation(), []
    for cid, v in sorted(val.items()):
        if not v["synthetic"]:
            continue
        noisy, clean = read_int(val_audio_path(cid, True)), read_int(val_audio_path(cid, True, clean=True))
        n = min(len(noisy), len(clean))
        r = (noisy[:n] - clean[:n]).astype(np.float64) / 32768.0
        for i, (on, off, cat, tag) in enumerate(v["events"]):
            # The template is the labelled span itself: detections then reproduce label conventions exactly.
            a, b = max(0, int(round(on * SR))), min(n, int(round(off * SR)))
            if b - a < 160 or not np.any(r[a:b]):
                continue
            items.append(dict(clip=cid, idx=i, tag=tag, category=cat, start=a, wave=r[a:b].copy()))
    return items


def cluster(items):
    F = np.stack([fingerprint(it["wave"]) for it in items])
    S = F @ F.T / F.shape[1]
    dur = np.array([len(it["wave"]) / SR for it in items])
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(S > FP_CORR, 1))):
        if abs(dur[i] - dur[j]) <= FP_DUR:
            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    return list(groups.values())


def contained(short, long):
    """Max NCC of `short` slid along `long` (detects crop variants of the same source snippet)."""
    c = fftconvolve(long, short[::-1], mode="valid")
    l2 = np.concatenate([[0.0], np.cumsum(long ** 2)])
    e = np.sqrt(np.maximum(l2[len(short):] - l2[:-len(short)], 1e-12)) * np.linalg.norm(short)
    return float(np.max(c / e))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = extract()
    groups = cluster(items)
    reps = []
    for g in groups:
        best = max(g, key=lambda i: float(np.sqrt(np.mean(items[i]["wave"] ** 2))))   # least quantisation error
        reps.append(dict(members=list(g), rep=best))
    # Crop variants (a shorter snippet contained in a longer one) stay separate templates: a full-length
    # template cannot match a clip that only contains its crop. They are only annotated.
    reps.sort(key=lambda r: -len(items[r["rep"]]["wave"]))
    for k, r in enumerate(reps):
        w, tag = items[r["rep"]]["wave"], items[r["rep"]]["tag"]
        r["crop_of"] = next((j for j in range(k) if items[reps[j]["rep"]]["tag"] == tag
                             and len(items[reps[j]["rep"]]["wave"]) > len(w)
                             and contained(w, items[reps[j]["rep"]]["wave"]) >= CONTAIN_NCC), None)
    waves, meta, event_map = {}, [], defaultdict(dict)
    for k, m in enumerate(reps):
        sid = f"s{k:04d}"
        rep = items[m["rep"]]
        w = rep["wave"] / (np.sqrt(np.mean(rep["wave"] ** 2)) + 1e-12)
        waves[sid] = w.astype(np.float32)
        meta.append(dict(id=sid, tag=rep["tag"], category=rep["category"], duration=round(len(w) / SR, 4),
                         uses=len(m["members"]), source_clips=sorted({items[i]["clip"] for i in m["members"]}),
                         crop_of=None if m["crop_of"] is None else f"s{m['crop_of']:04d}"))
        for i in m["members"]:
            event_map[items[i]["clip"]][items[i]["idx"]] = sid
    np.savez_compressed(OUT / "snippets.npz", **waves)
    (OUT / "snippets.json").write_text(json.dumps(meta, indent=1))
    (OUT / "val_event_map.json").write_text(json.dumps(event_map))
    uses = np.array([m["uses"] for m in meta])
    print(f"{len(items)} events -> {len(meta)} snippets; uses median {np.median(uses):.0f}, max {uses.max()}, "
          f"singletons {(uses == 1).sum()}")


if __name__ == "__main__":
    main()
