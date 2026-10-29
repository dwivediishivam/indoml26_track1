"""Score tuned configs on validation with test-mix weighting (synthetic clips x W), full set and fold 4.
Usage: PYTHONPATH=. python tools/compare_configs.py interim_clean final_f4 clean_full clean_full_tm"""
import sys
import numpy as np
from common.contract import load_validation, folds as load_folds
from decode import fusion, pipeline, fastmetric  # noqa: F401

W = 1.57
val, fm = load_validation(), load_folds()
ids = sorted(val)
durs = pipeline.val_durations(val)

def clip_stats(refs, preds):
    cands = []
    for i, (ra, rb) in enumerate(refs):
        tol = max(0.2 * (rb - ra), 0.05)
        for j, (pa, pb) in enumerate(preds):
            if abs(pa - ra) <= tol and abs(pb - rb) <= tol:
                cands.append((abs(pa - ra) + abs(pb - rb), i, j))
    ur, up = set(), set()
    for _, i, j in sorted(cands):
        if i not in ur and j not in up:
            ur.add(i); up.add(j)
    ev = refs + preds
    if not ev:
        d = 1.0
    else:
        n = int((max(e for _, e in ev) + .5) / .01) + 1
        r, p = np.zeros(n, bool), np.zeros(n, bool)
        for s, e in refs: r[max(0, int(s / .01)):min(n, int(e / .01) + 1)] = True
        for s, e in preds: p[max(0, int(s / .01)):min(n, int(e / .01) + 1)] = True
        t = r.sum() + p.sum(); d = 1.0 if not t else 2 * (r & p).sum() / t
    return len(ur), len(preds), len(refs), d

def agg(rows, keep):
    tp = fp = fn = ds = n = 0.0
    for cid, (a, npred, nref, d) in rows.items():
        if not keep(cid): continue
        w = W if val[cid]["synthetic"] else 1.0
        tp += w * a; fp += w * (npred - a); fn += w * (nref - a); ds += w * d; n += w
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1 + ds / n, f1, ds / n

for name in sys.argv[1:]:
    cfg = pipeline.load_config(f"work/tuned/{name}.json")
    caches = fusion.load_caches(cfg["caches"], "val", ids=ids, log=lambda *a: None)
    bank = pipeline.load_bank(cfg["bank"]["val"], log=lambda *a: None) if cfg.get("bank") else {}
    rows = {}
    for cid in ids:
        preds, _ = pipeline.predict_clip(caches, cid, durs[cid], bank.get(cid), cfg)
        rows[cid] = clip_stats([(a, b) for a, b, *_ in val[cid]["events"]], preds)
    full = agg(rows, lambda c: True); f4 = agg(rows, lambda c: fm[c] == 4)
    nat4 = agg(rows, lambda c: fm[c] == 4 and not val[c]["synthetic"])
    print(f"{name:16s} weighted full {full[0]:.4f} (F1 {full[1]:.4f} D {full[2]:.4f}) | weighted fold4 {f4[0]:.4f} | fold4 natural {nat4[0]:.4f}", flush=True)
