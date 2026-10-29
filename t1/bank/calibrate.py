"""Calibrate the bank matcher on validation with leave-fold-out banks, then write honest val events.

For a clip in fold f, only snippets seen in other folds are available, so unseen-snippet misses are
measured realistically. The logistic match model is also fitted leave-fold-out for the written
val_events.jsonl; the final model (calibration.json, used on blind) is fitted on all folds.

Usage: PYTHONPATH=. python -m bank.calibrate --workers 16
Writes work/bank/{calibration.json, val_events.jsonl, val_candidates.npz}
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from bank.matcher import BANK_DIR, FEATURES, Bank, candidates, clip_record, feature_matrix, nms, read_wav, score
from common.contract import folds as load_folds, load_validation, references, val_audio_path
from common.metric import score as metric

_STATE = {}


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    bank, folds = Bank.load(), load_folds()
    _STATE.update(folds=folds, banks={f: bank.without_fold(f, folds) for f in range(5)}, val=load_validation())


def _clip(cid):
    v, f = _STATE["val"][cid], _STATE["folds"][cid]
    x = read_wav(val_audio_path(cid, v["synthetic"]))
    cands = candidates(x, _STATE["banks"][f])
    for c in cands:
        c["label"] = int(v["synthetic"] and any(
            abs(c["onset"] - on) <= max(0.02, 0.05 * (off - on)) and abs(c["offset"] - off) <= max(0.02, 0.05 * (off - on))
            for on, off, *_ in v["events"]))
    return cid, f, len(x), cands


def fit_logistic(X, y, l2=1e-2, iters=100):
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mean) / std]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -40, 40)))
        H = Z.T @ (Z * (p * (1 - p) + 1e-9)[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, Z.T @ (p - y) + l2 * w)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return dict(features=list(FEATURES), mean=mean.tolist(), std=std.tolist(), w=w.tolist())


def bank_only_scores(records, val, thr):
    """Score bank events alone: synthetic references vs bank predictions on every clip (natural FPs count)."""
    preds = {cid: [(e["onset"], e["offset"]) for e in r["events"] if e["p"] >= thr] for cid, r in records.items()}
    syn = references(val, "synthetic")
    s = metric(syn, {k: preds[k] for k in syn})
    natural_fp = sum(len(p) for k, p in preds.items() if not val[k]["synthetic"])
    tp, fp, fn = s["tp"], s["fp"] + natural_fp, s["fn"]
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    flagged_nat = sum(1 for k, r in records.items() if not val[k]["synthetic"] and r["synthetic_score"] >= thr)
    flagged_syn = sum(1 for k, r in records.items() if val[k]["synthetic"] and r["synthetic_score"] >= thr)
    return dict(thr=thr, synthetic_f1_with_natural_fp=f1, synthetic_dice=s["segment_dice"], tp=tp, fp=fp, fn=fn,
                natural_fp_events=natural_fp, natural_clips_flagged=flagged_nat, synthetic_clips_flagged=flagged_syn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    val = load_validation()
    with Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_clip, sorted(val), chunksize=8)
    by_fold = {f: [] for f in range(5)}
    for cid, f, n, cands in results:
        by_fold[f].append((cid, n, cands))
    allc = [c for _, _, _, cands in results for c in cands]
    X, y = feature_matrix(allc), np.array([c["label"] for c in allc], dtype=np.float64)
    print(f"{len(allc)} candidates, {int(y.sum())} positives over {sum(len(v['events']) for v in val.values() if v['synthetic'])} synthetic events")
    np.savez_compressed(BANK_DIR / "val_candidates.npz", X=X, y=y, features=np.array(FEATURES))

    bank = Bank.load()
    records = {}
    for f in range(5):
        train = [c for g in range(5) if g != f for _, _, cands in by_fold[g] for c in cands]
        model = fit_logistic(feature_matrix(train), np.array([c["label"] for c in train], dtype=np.float64))
        for cid, n, cands in by_fold[f]:
            score(cands, model)
            records[cid] = clip_record(cid, nms(cands), bank, duration=n / 16000)
    with open(BANK_DIR / "val_events.jsonl.partial", "w") as fh:
        for cid in sorted(records):
            fh.write(json.dumps(records[cid]) + "\n")
    os.replace(BANK_DIR / "val_events.jsonl.partial", BANK_DIR / "val_events.jsonl")

    sweep = [bank_only_scores(records, val, t) for t in np.round(np.arange(0.05, 1.0, 0.05), 2)]
    best = max(sweep, key=lambda s: s["synthetic_f1_with_natural_fp"])
    final = fit_logistic(X, y)
    final.update(thr=best["thr"], leave_fold_out_sweep=sweep)
    (BANK_DIR / "calibration.json").write_text(json.dumps(final, indent=1))
    for s in sweep:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
    print("best", json.dumps(best))


if __name__ == "__main__":
    main()
