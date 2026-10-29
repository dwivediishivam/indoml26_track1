"""Learned boundary refiner for natural-route predictions.

Natural predictions are found but run too long (pred/ref duration ~1.5x; most misses are near-misses).
For every predicted span on a natural-route clip, features = fused probability tracks (union, 7 categories,
union onset/offset) in +-0.4 s windows around the predicted onset and offset, plus span statistics. Two
gradient-boosted regressors predict (ref_onset - onset) and (ref_offset - offset), trained on spans that
overlap a reference (IoU >= 0.2).

  eval:  PYTHONPATH=. python tools/refiner.py eval  --config work/tuned/X.json   (5-fold CV, official scorer)
  fit:   PYTHONPATH=. python tools/refiner.py fit   --config work/tuned/X.json   -> work/refiner/<name>.joblib
  apply: PYTHONPATH=. python tools/refiner.py apply --config work/tuned/X.json --out work/submissions/X_ref.zip
Only use caches that never trained on validation audio (otherwise features are optimistic).
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from common.contract import FRAME, WORK, blind_ids, folds as load_folds, load_validation, n_frames
from common.metric import score as metric
from decode import fusion, pipeline

W = 20                     # +-0.4 s window (frames)
CH = list(range(10))       # union, 7 categories, union onset, union offset
MAX_SHIFT = 1.0


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(a[1] - a[0] + b[1] - b[0] - inter, 1e-9)


def window(p, t):
    i = int(round(t / FRAME))
    idx = np.clip(np.arange(i - W, i + W + 1), 0, p.shape[1] - 1)
    return p[CH][:, idx].ravel()


def features(p, span, duration):
    a, b = span
    p = np.nan_to_num(p, nan=0.0)
    lo, hi = int(a / FRAME), max(int(a / FRAME) + 1, int(np.ceil(b / FRAME)))
    inside = p[0, lo:hi]
    stats = [b - a, duration, a, duration - b, float(inside.mean()), float(inside.max()), float(inside.min()),
             float(a <= 0.0), float(b >= duration - 1e-3)]
    return np.concatenate([window(p, a), window(p, b), stats]).astype(np.float32)


def predictions(cfg, split, ids, val=None):
    """-> {cid: (probs, spans, route, duration)} for all ids."""
    caches = fusion.load_caches(cfg["caches"], split, ids=ids, log=lambda *a: None)
    bank_path = (cfg.get("bank") or {}).get(split)
    bank = pipeline.load_bank(bank_path, log=lambda *a: None) if bank_path else {}
    durs = pipeline.val_durations({k: val[k] for k in ids}) if split == "val" else pipeline.blind_durations(ids)
    out = {}
    for cid in ids:
        p = fusion.fused_clip(caches, cid, cfg["fusion"], n_frames(durs[cid]))
        spans, route = pipeline.predict(p, durs[cid], bank.get(cid), cfg)
        out[cid] = (p, spans, route, durs[cid])
    return out


def training_rows(pred, val, cids):
    X, y_on, y_off = [], [], []
    for cid in cids:
        p, spans, route, dur = pred[cid]
        if route != "natural" or p is None:
            continue
        refs = [(a, b) for a, b, *_ in val[cid]["events"]]
        for s in spans:
            best = max(refs, key=lambda r: iou(s, r), default=None)
            if best is None or iou(s, best) < 0.2:
                continue
            X.append(features(p, s, dur))
            y_on.append(np.clip(best[0] - s[0], -MAX_SHIFT, MAX_SHIFT))
            y_off.append(np.clip(best[1] - s[1], -MAX_SHIFT, MAX_SHIFT))
    return np.array(X), np.array(y_on), np.array(y_off)


def fit(X, y_on, y_off, seed=0):
    kw = dict(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, random_state=seed)
    return HistGradientBoostingRegressor(**kw).fit(X, y_on), HistGradientBoostingRegressor(**kw).fit(X, y_off)


def refine(models, p, spans, dur, shrink=1.0):
    if p is None or not spans:
        return spans
    m_on, m_off = models
    X = np.stack([features(p, s, dur) for s in spans])
    d_on, d_off = m_on.predict(X) * shrink, m_off.predict(X) * shrink
    out = []
    for (a, b), da, db in zip(spans, d_on, d_off):
        na = a if a <= 0.0 else min(max(0.0, a + da), dur)
        nb = b if b >= dur - 1e-3 else min(max(0.0, b + db), dur)
        out.append((round(na, 3), round(nb, 3)) if nb - na >= 0.04 else (a, b))
    return sorted(out)


def evaluate(cfg):
    val, fmap = load_validation(), load_folds()
    ids = sorted(val)
    pred = predictions(cfg, "val", ids, val)
    refs = {k: [(a, b) for a, b, *_ in v["events"]] for k, v in val.items()}
    base = {k: pred[k][1] for k in ids}
    s0 = metric(refs, base)
    results = {}
    for shrink in (0.5, 0.75, 1.0):
        new = dict(base)
        for f in range(5):
            train = [k for k in ids if fmap[k] != f]
            test = [k for k in ids if fmap[k] == f]
            models = fit(*training_rows(pred, val, train))
            for k in test:
                p, spans, route, dur = pred[k]
                if route == "natural":
                    new[k] = refine(models, p, spans, dur, shrink)
        results[shrink] = metric(refs, new)
        nat = [k for k in ids if not val[k]["synthetic"]]
        print(f"shrink {shrink}: combined {results[shrink]['combined']:.5f} (F1 {results[shrink]['event_f1']:.4f} "
              f"Dice {results[shrink]['segment_dice']:.4f}) | natural {metric({k: refs[k] for k in nat}, {k: new[k] for k in nat})['combined']:.5f}",
              flush=True)
    nat = [k for k in ids if not val[k]["synthetic"]]
    print(f"baseline: combined {s0['combined']:.5f} (F1 {s0['event_f1']:.4f} Dice {s0['segment_dice']:.4f}) | natural "
          f"{metric({k: refs[k] for k in nat}, {k: base[k] for k in nat})['combined']:.5f}")
    best = max(results, key=lambda s: results[s]["combined"])
    print(json.dumps({"best_shrink": best, "cv_combined": results[best]["combined"], "baseline": s0["combined"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "fit", "apply"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--shrink", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = pipeline.load_config(args.config)
    name = Path(args.config).stem
    model_path = WORK / "refiner" / f"{name}.joblib"
    if args.mode == "eval":
        evaluate(cfg)
    elif args.mode == "fit":
        val = load_validation()
        ids = sorted(val)
        models = fit(*training_rows(predictions(cfg, "val", ids, val), val, ids))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, model_path)
        print(f"saved {model_path}")
    else:
        models = joblib.load(model_path)
        ids = blind_ids()
        pred = predictions(cfg, "blind", ids)
        lines, changed = [], 0
        for cid in ids:
            p, spans, route, dur = pred[cid]
            new = refine(models, p, spans, dur, args.shrink) if route == "natural" else spans
            changed += new != spans
            lines.append(json.dumps({"clip_id": cid, "events": [{"onset": float(a), "offset": float(b)} for a, b in new]}))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.jsonl", "\n".join(lines) + "\n")
        print(f"wrote {out}: {len(lines)} clips, {changed} natural clips refined")


if __name__ == "__main__":
    main()
