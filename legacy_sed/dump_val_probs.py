"""Dump frame probabilities for every validation clip to an npz cache.

Runs on Apple MPS (or CUDA/CPU). Resumable: an existing cache is loaded and
only missing clip ids are computed. The cache is what every postprocessing
sweep reads, so sweeps never touch the encoder again.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import Detector

VAL = Path(__file__).resolve().parents[1] / "work" / "validation"


def device_of(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_records():
    return json.loads((VAL / "validationMetadata.json").read_text())


def clip_path(rec):
    sub = "syntheticNoiseAudio" if rec["syntheticData"] else "naturalNoisyAudio"
    return VAL / sub / rec["segmentFileName"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = device_of(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Detector(backbone=ck.get("backbone", "microsoft/wavlm-base-plus"),
                     hidden=ck.get("hidden", 160), pretrained=False)
    sd = ck.get("model") or ck.get("state_dict")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), "unexpected", len(unexpected), flush=True)
    model.eval().to(dev)

    out = Path(args.out)
    cache = {}
    if out.exists():
        with np.load(out) as z:
            cache = {k: z[k] for k in z.files}
        print(f"resuming with {len(cache)} cached clips", flush=True)

    records = load_records()
    if args.limit:
        records = records[: args.limit]
    todo = [r for r in records if Path(r["segmentFileName"]).stem not in cache]
    print(f"{len(todo)} clips to run on {dev}", flush=True)

    # group by similar length so padding waste stays small
    todo.sort(key=lambda r: r["duration"])
    t0 = time.time()
    done = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i: i + args.batch]
        waves, ids, frames = [], [], []
        for rec in chunk:
            wav, sr = sf.read(clip_path(rec), dtype="float32")
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            assert sr == 16000, sr
            waves.append(wav)
            ids.append(Path(rec["segmentFileName"]).stem)
            frames.append(max(1, math.ceil(len(wav) / 16000 / 0.02)))
        width = max(len(w) for w in waves)
        batch = np.zeros((len(waves), width), np.float32)
        for j, w in enumerate(waves):
            batch[j, : len(w)] = w
        x = torch.from_numpy(batch).to(dev)
        with torch.no_grad():
            logits = model(x, output_frames=max(frames))
            prob = torch.sigmoid(logits.float()).cpu().numpy()
        for j, cid in enumerate(ids):
            cache[cid] = prob[j, :, : frames[j]].astype(np.float16)
        done += len(chunk)
        if (i // args.batch) % 25 == 0:
            rate = done / max(1e-6, time.time() - t0)
            print(f"{done}/{len(todo)} {rate:.1f} clips/s "
                  f"eta {(len(todo)-done)/max(rate,1e-6)/60:.1f} min", flush=True)
            np.savez_compressed(out.with_suffix(".tmp.npz"), **cache)
            out.with_suffix(".tmp.npz").replace(out)
    np.savez_compressed(out.with_suffix(".tmp.npz"), **cache)
    out.with_suffix(".tmp.npz").replace(out)
    print(f"wrote {len(cache)} clips to {out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
