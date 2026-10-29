"""Diagnose natural-clip errors of a tuned config on validation.
Usage: PYTHONPATH=. python tools/natural_errors.py work/tuned/interim_clean.json
"""
import collections
import sys

import numpy as np

from common.contract import load_validation
from decode import fusion, pipeline

cfg = pipeline.load_config(sys.argv[1])
val = load_validation()
ids = sorted(k for k, v in val.items() if not v["synthetic"])
caches = fusion.load_caches(cfg["caches"], "val", ids=ids, log=lambda *a: None)
bank = pipeline.load_bank(cfg["bank"]["val"], log=lambda *a: None) if cfg.get("bank") else {}
durs = pipeline.val_durations({k: val[k] for k in ids})

def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(a[1] - a[0] + b[1] - b[0] - inter, 1e-9)

fn_kind, fp_kind = collections.Counter(), collections.Counter()
fn_by = collections.defaultdict(lambda: [0, 0])     # tag -> [missed, total]
dur_by = collections.defaultdict(lambda: [0, 0])
on_err, off_err, n_pred, n_ref = [], [], 0, 0
for cid in ids:
    refs = [(a, b, c, t) for a, b, c, t in val[cid]["events"]]
    preds, _ = pipeline.predict_clip(caches, cid, durs[cid], bank.get(cid), cfg)
    n_pred += len(preds); n_ref += len(refs)
    cands = []
    for i, (ra, rb, *_ ) in enumerate(refs):
        tol = max(0.2 * (rb - ra), 0.05)
        for j, (pa, pb) in enumerate(preds):
            if abs(pa - ra) <= tol and abs(pb - rb) <= tol:
                cands.append((abs(pa - ra) + abs(pb - rb), i, j))
    ur, up = set(), set()
    for _, i, j in sorted(cands):
        if i not in ur and j not in up:
            ur.add(i); up.add(j)
    for i, (ra, rb, c, t) in enumerate(refs):
        d = rb - ra
        bucket = "<0.25" if d < 0.25 else "0.25-0.5" if d < 0.5 else "0.5-1" if d < 1 else "1-3" if d < 3 else ">=3"
        dur_by[bucket][1] += 1; fn_by[t][1] += 1
        if i in ur:
            continue
        dur_by[bucket][0] += 1; fn_by[t][0] += 1
        ov = [(iou((ra, rb), p), p) for p in preds]
        best = max(ov, default=(0, None))
        if best[0] >= 0.3:
            fn_kind["near-miss (IoU>=0.3, endpoint outside tol)"] += 1
            on_err.append(best[1][0] - ra); off_err.append(best[1][1] - rb)
        elif best[0] > 0:
            fn_kind["weak overlap (IoU<0.3: merged/fragmented)"] += 1
        else:
            fn_kind["no prediction at all"] += 1
    for j, p in enumerate(preds):
        if j in up:
            continue
        best = max((iou(p, (a, b)) for a, b, *_ in refs), default=0)
        fp_kind["overlaps a ref (boundary/merge)" if best > 0 else "spurious (no ref overlap)"] += 1

print(f"natural clips {len(ids)}: refs {n_ref}, preds {n_pred}")
print("FN breakdown:", dict(fn_kind))
print("FP breakdown:", dict(fp_kind))
if on_err:
    print("near-miss signed error (pred - ref) onset q10/50/90:", np.percentile(on_err, [10, 50, 90]).round(3),
          " offset:", np.percentile(off_err, [10, 50, 90]).round(3))
print("recall by duration:", {k: f"{1 - v[0] / v[1]:.2f} of {v[1]}" for k, v in sorted(dur_by.items())})
print("worst tags by misses:", sorted(((v[0], t, f"{1 - v[0] / v[1]:.2f}") for t, v in fn_by.items()), reverse=True)[:10])

# merge vs fragment per tag: a pred overlapping >=2 refs = merge; a ref overlapped by >=2 preds = fragmentation
mf = collections.defaultdict(lambda: collections.Counter())
for cid in ids:
    refs = val[cid]["events"]
    preds, _ = pipeline.predict_clip(caches, cid, durs[cid], bank.get(cid), cfg)
    for a, b, c, t in refs:
        k = sum(1 for p in preds if iou((a, b), p) > 0.05)
        mf[t]["refs"] += 1
        if k >= 2: mf[t]["fragmented"] += 1
        for p in preds:
            if iou((a, b), p) > 0.05 and sum(1 for a2, b2, *_ in refs if iou((a2, b2), p) > 0.05) >= 2:
                mf[t]["in_merged_pred"] += 1; break
        best = max(preds, key=lambda p: iou((a, b), p))
        if iou((a, b), best) > 0.05: mf[t]["dur_ratio_sum"] += (best[1] - best[0]) / (b - a)
for t in ["<horn>", "[breathing]", "<bird sound>", "<bird squawking>", "[inhaling]", "<child talking>"]:
    m = mf[t]; print(f"{t:18s} refs {m['refs']:4d} fragmented {m['fragmented']:4d} in-merged-pred {m['in_merged_pred']:4d} mean pred/ref dur {m['dur_ratio_sum'] / max(1, m['refs']):.2f}")
