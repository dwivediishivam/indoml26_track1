"""Dump SED frame/boundary probabilities for validation or blind clips.

Writes an npz keyed by clip id, each value a float16 array of shape
(n_out + 2, frames): channel 0 is the binary noise track, 1..7 the categories,
8.. the per-tag tracks, and the last two are onset / offset heatmaps.
Resumable: an existing output file is reloaded and only missing ids are run.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sed_model import load

ROOT = Path(__file__).resolve().parents[1]


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def val_clips():
    vd = ROOT / "work" / "validation"
    recs = json.loads((vd / "validationMetadata.json").read_text())
    out = []
    for r in recs:
        sub = "syntheticNoiseAudio" if r["syntheticData"] else "naturalNoisyAudio"
        out.append((Path(r["segmentFileName"]).stem, vd / sub / r["segmentFileName"]))
    return out


def dir_clips(d):
    return sorted((p.stem, p) for p in Path(d).rglob("*.wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="validation", help="'validation' or a wav directory")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    model, ck = load(args.checkpoint, dev)
    if args.fp16 and dev.type == "cuda":
        model = model.half()
    print(f"checkpoint step {ck.get('step')} on {dev}", flush=True)

    clips = val_clips() if args.source == "validation" else dir_clips(args.source)
    out = Path(args.out)
    cache = {}
    if out.exists():
        with np.load(out) as z:
            cache = {k: z[k] for k in z.files}
        print(f"resuming with {len(cache)} cached", flush=True)
    todo = [(c, p) for c, p in clips if c not in cache]
    print(f"{len(todo)} / {len(clips)} clips to run", flush=True)

    waves = {}
    lens = []
    for cid, path in todo:
        info = sf.info(str(path))
        lens.append((info.frames, cid, path))
    lens.sort()

    t0 = time.time()
    done = 0
    for i in range(0, len(lens), args.batch):
        chunk = lens[i: i + args.batch]
        arrs, ids, frames = [], [], []
        for n, cid, path in chunk:
            w, sr = sf.read(str(path), dtype="float32")
            if w.ndim == 2:
                w = w.mean(axis=1)
            if sr != 16000:
                import librosa
                w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            arrs.append(w); ids.append(cid)
            frames.append(max(1, math.ceil(len(w) / 16000 / 0.02)))
        width = max(len(a) for a in arrs)
        batch = np.zeros((len(arrs), width), np.float32)
        for j, a in enumerate(arrs):
            batch[j, : len(a)] = a
        x = torch.from_numpy(batch).to(dev)
        if args.fp16 and dev.type == "cuda":
            x = x.half()
        with torch.no_grad():
            fr, bd = model(x)
            T = max(frames)
            fr = torch.nn.functional.interpolate(fr.float(), size=T, mode="linear", align_corners=False)
            bd = torch.nn.functional.interpolate(bd.float(), size=T, mode="linear", align_corners=False)
            p = torch.sigmoid(torch.cat([fr, bd], 1)).cpu().numpy()
        for j, cid in enumerate(ids):
            cache[cid] = p[j, :, : frames[j]].astype(np.float16)
        done += len(chunk)
        if (i // args.batch) % 30 == 0:
            rate = done / max(1e-6, time.time() - t0)
            print(f"{done}/{len(lens)} {rate:.1f} clips/s eta {(len(lens)-done)/max(rate,1e-6)/60:.1f} min", flush=True)
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, **cache); tmp.replace(out)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **cache); tmp.replace(out)
    print(f"wrote {len(cache)} clips -> {out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
