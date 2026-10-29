"""Full-validation postprocessing search over a cached probability npz.

Every reported number is computed on all supplied validation clips with the
official metric in postproc.py. Coordinate descent over the decoder grid, with
natural / synthetic breakdowns.
"""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from postproc import DEFAULT, decode, score

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "work" / "validation"

_STATE = {}


def load_val():
    recs = json.loads((VAL / "validationMetadata.json").read_text())
    refs, durs, kind = {}, {}, {}
    for r in recs:
        cid = Path(r["segmentFileName"]).stem
        spans = []
        for ev in (r.get("NoiseSubCategoryTimeStamp") or []):
            try:
                a, b = float(ev["start"]), float(ev["end"])
            except Exception:
                continue
            if b > a:
                spans.append((max(0.0, a), b))
        refs[cid] = spans
        durs[cid] = float(r["duration"])
        kind[cid] = "synthetic" if r["syntheticData"] else "natural"
    return refs, durs, kind


def _init(probs_files, weights):
    probs = None
    for path, w in zip(probs_files, weights):
        with np.load(path) as z:
            cur = {k: z[k][0].astype(np.float32) * w for k in z.files}
        if probs is None:
            probs = cur
        else:
            for k in probs:
                if k in cur:
                    probs[k] = probs[k] + cur[k]
    _STATE["probs"] = probs
    _STATE["refs"], _STATE["durs"], _STATE["kind"] = load_val()


def _eval_chunk(job):
    cfg, cids = job
    probs, durs = _STATE["probs"], _STATE["durs"]
    return {c: decode(probs[c], durs[c], **cfg) for c in cids}


def evaluate(cfg, pool, chunks):
    preds = {}
    for part in pool.map(_eval_chunk, [(cfg, c) for c in chunks]):
        preds.update(part)
    refs, kind = _STATE["refs"], _STATE["kind"]
    out = {"all": score(refs, preds)}
    for k in ("natural", "synthetic"):
        sub = {c: v for c, v in refs.items() if kind[c] == k}
        out[k] = score(sub, preds)
    return out


GRID = {
    "hi":        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "lo":        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70],
    "smooth":    [1, 2, 3, 4, 5, 7],
    "median":    [1, 3, 5, 7, 9],
    "min_dur":   [0.0, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.30, 0.40],
    "merge_gap": [0.0, 0.02, 0.04, 0.06, 0.10, 0.16, 0.24],
    "pad_on":    [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.10],
    "pad_off":   [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.10],
    "split_drop": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    "split_min": [0.10, 0.16, 0.24, 0.30, 0.40],
    "norm":      [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "max_dur":   [0.0, 1.5, 2.5, 4.0],
}
ORDER = ["hi", "lo", "min_dur", "smooth", "median", "merge_gap", "split_drop",
         "split_min", "pad_on", "pad_off", "norm", "max_dur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="all", choices=["all", "natural", "synthetic"])
    ap.add_argument("--init", default=None, help="json file with a starting config")
    args = ap.parse_args()

    weights = args.weights or [1.0 / len(args.probs)] * len(args.probs)
    assert len(weights) == len(args.probs)

    _init(args.probs, weights)
    cids = sorted(_STATE["refs"])
    missing = [c for c in cids if c not in _STATE["probs"]]
    if missing:
        raise SystemExit(f"probability cache missing {len(missing)} clips, e.g. {missing[:3]}")
    nchunk = args.workers * 4
    chunks = [cids[i::nchunk] for i in range(nchunk)]

    cfg = dict(DEFAULT)
    if args.init:
        cfg.update(json.loads(Path(args.init).read_text()).get("params", {}))

    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.probs, weights)) as pool:
        best = evaluate(cfg, pool, chunks)
        print(f"start {args.target} combined={best[args.target]['combined']:.5f} "
              f"(all={best['all']['combined']:.5f})", flush=True)
        for rnd in range(args.rounds):
            improved = False
            for key in ORDER:
                base = cfg[key]
                cand_best, cand_val = best, base
                for v in GRID[key]:
                    if v == base:
                        continue
                    trial = dict(cfg); trial[key] = v
                    if trial["lo"] > trial["hi"]:
                        continue
                    res = evaluate(trial, pool, chunks)
                    if res[args.target]["combined"] > cand_best[args.target]["combined"] + 1e-6:
                        cand_best, cand_val = res, v
                if cand_val != base:
                    cfg[key] = cand_val
                    best = cand_best
                    improved = True
                    print(f"  r{rnd} {key}={cand_val} -> all={best['all']['combined']:.5f} "
                          f"f1={best['all']['f1']:.5f} dice={best['all']['dice']:.5f}", flush=True)
            if not improved:
                print(f"round {rnd}: converged", flush=True)
                break

    payload = {"params": cfg, "probs": args.probs, "weights": weights,
               "target": args.target, "validation": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(best, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
