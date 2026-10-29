"""Run the full validation bank + final calibration on the blind clips.

Usage: PYTHONPATH=. python -m bank.detect_blind --workers 16
Writes work/bank/blind_events.jsonl (same schema as val_events.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from bank.matcher import BANK_DIR, Bank, candidates, clip_record, nms, read_wav, score
from common.contract import BLIND_DIR, SR, blind_ids

_STATE = {}


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _STATE.update(bank=Bank.load(), model=json.loads((BANK_DIR / "calibration.json").read_text()))


def _clip(cid):
    x = read_wav(BLIND_DIR / f"{cid}.wav")
    cands = candidates(x, _STATE["bank"])
    score(cands, _STATE["model"])
    return clip_record(cid, nms(cands), _STATE["bank"], duration=len(x) / SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    ids = blind_ids()
    if len(ids) != 5517:
        raise SystemExit(f"expected 5517 blind clips, found {len(ids)} in {BLIND_DIR}")
    thr = json.loads((BANK_DIR / "calibration.json").read_text())["thr"]
    out = BANK_DIR / "blind_events.jsonl"
    with Pool(args.workers, initializer=_init) as pool, open(str(out) + ".partial", "w") as fh:
        records = []
        for i, rec in enumerate(pool.imap(_clip, ids, chunksize=8)):
            fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            if i % 500 == 0:
                print(f"{i}/{len(ids)}", flush=True)
    os.replace(str(out) + ".partial", out)
    flagged = [r for r in records if r["synthetic_score"] >= thr]
    n_ev = [sum(e["p"] >= thr for e in r["events"]) for r in flagged]
    print(f"clips flagged synthetic at p>={thr}: {len(flagged)}/{len(records)} "
          f"({len(flagged) / len(records):.3f}; organizers' synthetic share is 2080/5517 = 0.377)")
    if n_ev:
        print(f"bank events per flagged clip: mean {np.mean(n_ev):.2f} (validation synthetic mean 3.18)")


if __name__ == "__main__":
    main()
