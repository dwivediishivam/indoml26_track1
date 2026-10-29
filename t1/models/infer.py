"""Write a contract cache (float16 [24, T] sigmoid probabilities per clip) for val or blind clips.

  python -m models.infer --ckpt RUN_DIR|FILE.pt --split {val,blind} --out work/caches/<name>/{split}.npz \
      [--weights ema|student] [--tta] [--val-dir D] [--blind-dir D] [--window 10] [--hop 5] [--floor 0.3]

--ckpt RUN_DIR picks the top entry of RUN_DIR/checkpoints/best.json (student or EMA, by monitor score).
--ckpt latest.pt needs --weights (default ema). A retained best file carries its own weights.
--tta averages time shifts of 0, +20 ms and -20 ms (override with --tta-shifts-ms).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import contract  # noqa: E402
from models.data import read_wav, set_data_roots  # noqa: E402
from models.model import build_model  # noqa: E402
from models.predict import predict_clips  # noqa: E402
from models.utils import atomic_json, iso  # noqa: E402

FORMAT = "t1f_sed_v1"


def resolve_checkpoint(ckpt: str, weights: str | None):
    p = Path(ckpt).expanduser()
    if p.is_dir():
        index = p / "checkpoints" / "best.json"
        if not index.exists():
            raise FileNotFoundError(f"{index} missing; pass a .pt file explicitly (e.g. {p / 'latest.pt'})")
        rows = json.loads(index.read_text())["checkpoints"]
        if not rows:
            raise RuntimeError(f"{index} lists no checkpoints")
        if weights:
            rows = [r for r in rows if r["weights"] == weights]
            if not rows:
                raise RuntimeError(f"no retained {weights} checkpoint in {index}")
        p = index.parent / rows[0]["file"]
        print(f"[infer] using {p.name} (monitor combined {rows[0]['combined']:.4f})", flush=True)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_model(path: Path, weights: str | None, device):
    s = torch.load(path, map_location="cpu", weights_only=False)
    if s.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    if "ema" in s:                       # latest.pt (resumable state)
        kind = weights or "ema"
        state = s["ema"] if kind == "ema" else s["model"]
    else:                                # retained best file
        kind = s["weights"]
        if weights and weights != kind:
            raise ValueError(f"{path} holds {kind} weights, but --weights {weights} was requested")
        state = s["model"]
    model = build_model(s["config"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), s["config"], kind, int(s["step"])


def clip_list(split: str):
    if split == "val":
        val = contract.load_validation()
        return [(k, contract.val_audio_path(k, v["synthetic"])) for k, v in sorted(val.items())]
    ids = contract.blind_ids()
    if not ids:
        raise RuntimeError(f"no blind clips in {contract.BLIND_DIR}")
    return [(k, contract.BLIND_DIR / f"{k}.wav") for k in ids]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=("val", "blind"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", choices=("ema", "student"), default=None)
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--blind-dir", default=None)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--floor", type=float, default=0.3)
    ap.add_argument("--tta", action="store_true", help="average time shifts 0, +20 ms, -20 ms")
    ap.add_argument("--tta-shifts-ms", default=None, help="comma list of shifts in ms (multiples of 20)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256, help="clips per prediction chunk")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    if args.tta_shifts_ms:
        ms = [float(x) for x in args.tta_shifts_ms.split(",")]
    else:
        ms = [0.0, 20.0, -20.0] if args.tta else [0.0]
    if any(abs(m / 20 - round(m / 20)) > 1e-9 for m in ms):
        raise ValueError(f"TTA shifts must be multiples of 20 ms, got {ms}")
    shifts = tuple(int(round(m / 20)) for m in ms)

    set_data_roots(val_dir=args.val_dir, blind_dir=args.blind_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = resolve_checkpoint(args.ckpt, args.weights)
    model, cfg, kind, step = load_model(path, args.weights, device)
    clips = clip_list(args.split)
    if args.limit:
        clips = clips[:args.limit]
    print(f"[infer] {args.split}: {len(clips)} clips, weights={kind} step={step} shifts={shifts}", flush=True)

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(clips), args.chunk):
            chunk = clips[i:i + args.chunk]
            wavs = dict(zip([c for c, _ in chunk], pool.map(read_wav, [p for _, p in chunk])))
            probs = predict_clips(model, wavs, device, args.window, args.hop, args.floor, shifts, args.batch_size)
            for cid, p in probs.items():
                if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                    raise FloatingPointError(f"{cid}: invalid probabilities")
                results[cid] = p
            done = min(len(clips), i + args.chunk)
            print(f"[infer] {done}/{len(clips)} clips, {time.time() - t0:.0f}s", flush=True)
    missing = [c for c, _ in clips if c not in results]
    if missing:
        raise RuntimeError(f"{len(missing)} clips missing from predictions")
    contract.save_cache(out, results)
    atomic_json(out.with_suffix(".json"), {"checkpoint": str(path), "weights": kind, "step": step,
                                          "config_name": cfg.get("name"), "split": args.split, "clips": len(results),
                                          "window": args.window, "hop": args.hop, "floor": args.floor,
                                          "tta_shifts_frames": list(shifts), "written": iso(time.time()),
                                          "seconds": time.time() - t0})
    print(f"[infer] wrote {out} ({len(results)} clips) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
