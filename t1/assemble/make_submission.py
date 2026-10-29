#!/usr/bin/env python3
"""Build the blind submission ZIP from a tuned config.

  python -m assemble.make_submission --config work/tuned/<name>.json --out work/submissions/<name>.zip

Loads the blind caches named in the config (legacy caches are converted with from_legacy), the
blind bank events, applies fusion -> decoding -> merge exactly as decode/tune.py did on validation
(decode/pipeline.py), writes predictions.jsonl, validates it (5,517 unique blind ids, no empty
lists, finite floats, 0 <= onset < offset <= duration + 0.05 with durations from WAV headers),
zips it with the file at the root, and prints summary statistics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import collections
import json
import math
import time
import zipfile

import numpy as np

from common.contract import blind_ids, load_validation
from decode import fusion, merge, pipeline

EXPECTED_CLIPS = 5517
EXPECTED_SYNTHETIC = 2080
TOL = 0.05


def validate(text, ids, durs):
    """Re-parse the exact JSONL payload; returns a list of error strings (empty = valid)."""
    errors, seen = [], set()
    want = set(ids)
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            errors.append(f"line {ln}: blank line"); continue
        try:
            obj = json.loads(line)
        except Exception as e:
            errors.append(f"line {ln}: bad json ({e})"); continue
        cid = obj.get("clip_id")
        if cid not in want:
            errors.append(f"line {ln}: unknown clip_id {cid!r}"); continue
        if cid in seen:
            errors.append(f"line {ln}: duplicate clip_id {cid}"); continue
        seen.add(cid)
        ev = obj.get("events")
        if not isinstance(ev, list) or not ev:
            errors.append(f"{cid}: empty or missing events"); continue
        d = durs[cid]
        for e in ev:
            a, b = e.get("onset"), e.get("offset")
            if not all(isinstance(x, float) or (isinstance(x, int) and not isinstance(x, bool)) for x in (a, b)):
                errors.append(f"{cid}: non-numeric event {e}"); continue
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= d + TOL):
                errors.append(f"{cid}: invalid event {e} (duration {d:.3f})")
    missing = want - seen
    if missing:
        errors.append(f"{len(missing)} blind clips missing, e.g. {sorted(missing)[:3]}")
    return errors


def hist(counts, top=8):
    c = collections.Counter(min(int(n), top) for n in counts)
    return {(str(k) if k < top else f"{top}+"): c.get(k, 0) for k in range(1, top + 1)}


def fmt_hist(h, total=None):
    total = total or max(1, sum(h.values()))
    return " ".join(f"{k}:{v}({100 * v / total:.0f}%)" for k, v in h.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="output .zip path")
    ap.add_argument("--blind", action="append", default=[], help="override blind cache path NAME=PATH")
    ap.add_argument("--bank", default=None, help="override blind bank events jsonl")
    ap.add_argument("--allow-missing-bank", action="store_true", help="decode everything as natural if no bank file")
    ap.add_argument("--allow-count", action="store_true", help=f"do not require exactly {EXPECTED_CLIPS} blind clips")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = pipeline.load_config(args.config)
    ids = blind_ids()
    if len(ids) != EXPECTED_CLIPS and not args.allow_count:
        raise SystemExit(f"found {len(ids)} blind wavs, expected {EXPECTED_CLIPS} (use --allow-count to override)")
    print(f"[submit] {len(ids)} blind clips; reading WAV headers", flush=True)
    durs = pipeline.blind_durations(ids)

    overrides = {}
    for kv in args.blind:
        k, _, v = kv.partition("=")
        overrides[k] = v
    weights = cfg["fusion"]["weights"]
    used = [s for s in cfg["caches"] if weights.get(s["name"], 0) > 0]
    for s in used:
        p = overrides.get(s["name"], s.get("blind"))
        if not p or not Path(p).exists():
            raise SystemExit(f"blind cache for {s['name']} not found: {p!r} (pass --blind {s['name']}=PATH)")
    caches = fusion.load_caches(used, "blind", ids=ids, overrides=overrides)
    no_cache = [c for c in ids if all(c not in caches[s["name"]] for s in used)]
    if no_cache:
        print(f"[submit] WARNING {len(no_cache)} clips have no cache entry -> whole-clip fallback", flush=True)

    mcfg = {**merge.DEFAULT, **cfg.get("merge", {})}
    bank = {}
    if mcfg.get("bank_on"):
        bpath = args.bank or (cfg.get("bank") or {}).get("blind") or pipeline.BANK_DEFAULT["blind"]
        if not Path(bpath).exists():
            if not args.allow_missing_bank:
                raise SystemExit(f"config uses bank events but {bpath} is missing (or pass --allow-missing-bank)")
            print(f"[submit] WARNING bank file {bpath} missing: all clips decoded as natural", flush=True)
        else:
            bank = pipeline.load_bank(bpath)
        miss = len(set(ids) - set(bank))
        if bank and miss:
            print(f"[submit] WARNING {miss} blind clips have no bank record (decoded as natural)", flush=True)

    preds, routes = {}, collections.Counter()
    for i, cid in enumerate(ids):
        spans, r = pipeline.predict_clip(caches, cid, durs[cid], bank.get(cid), cfg)
        preds[cid] = spans
        routes[r] += 1
        if (i + 1) % 1000 == 0:
            print(f"[submit] decoded {i + 1}/{len(ids)}", flush=True)

    lines = [json.dumps({"clip_id": cid, "events": [{"onset": float(a), "offset": float(b)} for a, b in preds[cid]]})
             for cid in ids]
    text = "\n".join(lines) + "\n"
    errors = validate(text, ids, durs)
    if errors:
        print("\n".join(errors[:30]))
        raise SystemExit(f"[submit] VALIDATION FAILED ({len(errors)} problems); nothing written")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".predictions.jsonl")
    jsonl.write_text(text)
    tmp = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", text)
    os.replace(tmp, out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert names == ["predictions.jsonl"], names
        assert z.read("predictions.jsonl").decode() == text

    # ---------------------------------------------------------------- statistics
    n_ev = np.array([len(preds[c]) for c in ids])
    flag = np.array([float((bank.get(c) or {}).get("synthetic_score", 0.0) or 0.0) >= mcfg["synth_thr"]
                     if mcfg.get("bank_on") else False for c in ids])
    lens = np.array([b - a for c in ids for a, b in preds[c]])
    starts0 = np.mean([a <= 1e-6 for c in ids for a, _ in preds[c]])
    ends = np.mean([b >= durs[c] - 1e-3 for c in ids for _, b in preds[c]])
    try:
        val = load_validation()
    except FileNotFoundError:  # validation set not installed: report blind statistics only
        val = {}
    vn = [len(v["events"]) for v in val.values() if not v["synthetic"]] or [0]
    vs = [len(v["events"]) for v in val.values() if v["synthetic"]] or [0]
    stats = {
        "clips": len(ids), "events": int(n_ev.sum()), "events_per_clip": float(n_ev.mean()),
        "median_events": float(np.median(n_ev)), "max_events": int(n_ev.max()),
        "mean_event_s": float(lens.mean()), "frac_onset_at_0": float(starts0), "frac_offset_at_end": float(ends),
        "routes": dict(routes),
        "bank_flag": {"synthetic_flagged_clips": int(flag.sum()), "natural_clips": int((~flag).sum()),
                      "expected_synthetic": EXPECTED_SYNTHETIC,
                      "events_flagged": int(n_ev[flag].sum()), "events_unflagged": int(n_ev[~flag].sum()),
                      "per_clip_flagged": float(n_ev[flag].mean()) if flag.any() else None,
                      "per_clip_unflagged": float(n_ev[~flag].mean()) if (~flag).any() else None},
        "hist_all": hist(n_ev), "hist_flagged": hist(n_ev[flag]), "hist_unflagged": hist(n_ev[~flag]),
        "val_ref": {"natural_per_clip": float(np.mean(vn)), "synthetic_per_clip": float(np.mean(vs)),
                    "hist_natural": hist(vn), "hist_synthetic": hist(vs)},
        "val_pred": cfg.get("val_counts"), "val_metrics": (cfg.get("metrics") or {}).get("full"),
        "cv": (cfg.get("cv") or {}).get("mean"), "config": str(args.config), "zip": str(out),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=1))
    print("\n================ SUBMISSION ================")
    print(f"zip: {out}  (predictions.jsonl at root; copy at {jsonl})")
    print(f"clips={stats['clips']} events={stats['events']} per-clip mean={stats['events_per_clip']:.2f} "
          f"median={stats['median_events']:.0f} max={stats['max_events']} mean-len={stats['mean_event_s']:.3f}s")
    print(f"onset at 0: {100 * starts0:.1f}% (val natural ~14%)  offset at end: {100 * ends:.1f}% (val natural ~22%)")
    print(f"routes: {dict(routes)}")
    bf = stats["bank_flag"]
    print(f"bank flag: synthetic={bf['synthetic_flagged_clips']} (expected ~{EXPECTED_SYNTHETIC}) "
          f"natural={bf['natural_clips']} (expected ~{EXPECTED_CLIPS - EXPECTED_SYNTHETIC}); "
          f"events flagged={bf['events_flagged']} ({bf['per_clip_flagged'] or 0:.2f}/clip) "
          f"unflagged={bf['events_unflagged']} ({bf['per_clip_unflagged'] or 0:.2f}/clip)")
    print(f"blind hist all       : {fmt_hist(stats['hist_all'])}")
    print(f"blind hist flagged   : {fmt_hist(stats['hist_flagged'])}")
    print(f"blind hist unflagged : {fmt_hist(stats['hist_unflagged'])}")
    print(f"val ref natural      : {fmt_hist(stats['val_ref']['hist_natural'])} "
          f"(mean {stats['val_ref']['natural_per_clip']:.2f})")
    print(f"val ref synthetic    : {fmt_hist(stats['val_ref']['hist_synthetic'])} "
          f"(mean {stats['val_ref']['synthetic_per_clip']:.2f})")
    vc = cfg.get("val_counts") or {}
    if vc:
        print(f"val pred natural     : {vc.get('pred_hist_natural')}")
        print(f"val pred synthetic   : {vc.get('pred_hist_synthetic')}")
        print(f"val pred/ref events  : {vc.get('pred_events')}/{vc.get('ref_events')} over {vc.get('clips')} clips")
    m = (cfg.get("metrics") or {}).get("full")
    if m:
        print(f"val combined={m['combined']:.5f} (F1 {m['event_f1']:.4f} Dice {m['segment_dice']:.4f}); "
              f"CV mean={stats['cv']}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
