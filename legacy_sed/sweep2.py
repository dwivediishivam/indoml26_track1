"""Full-validation coordinate-descent search over the multi-track decoder."""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import postproc2
from postproc import score
from sweep import load_val

_S = {}


def _init(files, weights):
    stack = None
    for path, w in zip(files, weights):
        with np.load(path) as z:
            cur = {k: z[k].astype(np.float32) * w for k in z.files}
        if stack is None:
            stack = cur
        else:
            for k in list(stack):
                if k in cur:
                    n = min(stack[k].shape[-1], cur[k].shape[-1])
                    stack[k] = stack[k][..., :n] + cur[k][..., :n]
    _S["P"] = stack
    _S["refs"], _S["durs"], _S["kind"] = load_val()


def _chunk(job):
    cfg, cids = job
    P, durs = _S["P"], _S["durs"]
    return {c: postproc2.decode(P[c], durs[c], **cfg) for c in cids}


def evaluate(cfg, pool, chunks):
    preds = {}
    for part in pool.map(_chunk, [(cfg, c) for c in chunks]):
        preds.update(part)
    refs, kind = _S["refs"], _S["kind"]
    out = {"all": score(refs, preds)}
    for k in ("natural", "synthetic"):
        out[k] = score({c: v for c, v in refs.items() if kind[c] == k}, preds)
    return out


GRID = {
    "hi": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "lo": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
    "smooth": [1, 2, 3, 5],
    "median": [1, 3, 5],
    "min_dur": [0.0, 0.04, 0.08, 0.10, 0.14, 0.18, 0.24, 0.32],
    "merge_gap": [0.0, 0.02, 0.04, 0.08, 0.14],
    "pad_on": [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06],
    "pad_off": [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06],
    "norm": [0.0, 0.25, 0.5, 0.75, 1.0],
    "split_drop": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    "split_min": [0.10, 0.16, 0.24, 0.32, 0.40],
    "track_thr": [1.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70],
    "track_min": [0.04, 0.08, 0.12, 0.18, 0.24],
    "track_cover": [0.3, 0.45, 0.6, 0.75, 0.9],
    "track_iou": [0.3, 0.45, 0.6, 0.75, 0.9],
    "use_cat": [0.0, 1.0],
    "use_tag": [0.0, 1.0],
    "bref": [0, 1, 2, 3, 5],
    "bref_thr": [0.15, 0.25, 0.35, 0.5, 0.7],
}
ORDER = ["hi", "lo", "min_dur", "pad_on", "pad_off", "smooth", "median", "merge_gap",
         "track_thr", "track_min", "track_cover", "track_iou", "use_cat", "use_tag",
         "split_drop", "split_min", "bref", "bref_thr", "norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="all", choices=["all", "natural", "synthetic"])
    ap.add_argument("--init", default=None)
    args = ap.parse_args()

    weights = args.weights or [1.0 / len(args.probs)] * len(args.probs)
    _init(args.probs, weights)
    cids = sorted(_S["refs"])
    missing = [c for c in cids if c not in _S["P"]]
    if missing:
        raise SystemExit(f"cache missing {len(missing)} clips e.g. {missing[:3]}")
    nchunk = args.workers * 4
    chunks = [cids[i::nchunk] for i in range(nchunk)]

    cfg = dict(postproc2.DEFAULT)
    if args.init:
        cfg.update(json.loads(Path(args.init).read_text()).get("params", {}))

    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.probs, weights)) as pool:
        best = evaluate(cfg, pool, chunks)
        print(f"start all={best['all']['combined']:.5f} f1={best['all']['f1']:.5f} "
              f"dice={best['all']['dice']:.5f}", flush=True)
        for rnd in range(args.rounds):
            improved = False
            for key in ORDER:
                base = cfg[key]
                cb, cv = best, base
                for v in GRID[key]:
                    if v == base:
                        continue
                    trial = dict(cfg); trial[key] = v
                    if trial["lo"] > trial["hi"]:
                        continue
                    res = evaluate(trial, pool, chunks)
                    if res[args.target]["combined"] > cb[args.target]["combined"] + 1e-6:
                        cb, cv = res, v
                if cv != base:
                    cfg[key], best, improved = cv, cb, True
                    print(f"  r{rnd} {key}={cv} -> all={best['all']['combined']:.5f} "
                          f"f1={best['all']['f1']:.5f} dice={best['all']['dice']:.5f} "
                          f"(nat {best['natural']['combined']:.4f} syn {best['synthetic']['combined']:.4f})",
                          flush=True)
            if not improved:
                print(f"round {rnd}: converged", flush=True)
                break

    payload = {"decoder": "postproc2", "params": cfg, "probs": args.probs,
               "weights": weights, "target": args.target, "validation": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(best["all"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
