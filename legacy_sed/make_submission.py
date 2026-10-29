"""Build the Track 1 submission ZIP from a cached blind probability stack.

Applies the decoder parameters chosen by a full-validation sweep, then checks
the official structural requirements: exactly one root-level predictions.jsonl,
one line per evaluation clip, no duplicates, monotonic onset < offset.
"""
from __future__ import annotations

import argparse, json, sys, zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import postproc2
import postproc

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--params", required=True, help="sweep json with a params block")
    ap.add_argument("--audio-dir", default=str(ROOT / "work" / "blind"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.params).read_text())
    params = cfg["params"]
    decoder = postproc2 if cfg.get("decoder") == "postproc2" else postproc

    weights = args.weights or [1.0 / len(args.probs)] * len(args.probs)
    stack = None
    for path, w in zip(args.probs, weights):
        with np.load(path) as z:
            cur = {k: z[k].astype(np.float32) * w for k in z.files}
        if stack is None:
            stack = cur
        else:
            for k in list(stack):
                if k in cur:
                    n = min(stack[k].shape[-1], cur[k].shape[-1])
                    stack[k] = stack[k][..., :n] + cur[k][..., :n]

    wavs = sorted(Path(args.audio_dir).rglob("*.wav"))
    print(f"{len(wavs)} blind clips, {len(stack)} cached probability rows")
    ids = [p.stem for p in wavs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate clip ids in the blind audio directory")
    missing = [c for c in ids if c not in stack]
    if missing:
        raise SystemExit(f"probability cache missing {len(missing)} clips e.g. {missing[:3]}")

    lines = []
    n_events = 0
    for path, cid in zip(wavs, ids):
        dur = sf.info(str(path)).duration
        events = decoder.decode(stack[cid], dur, **params)
        clean = []
        for a, b in events:
            a, b = max(0.0, float(a)), min(float(dur), float(b))
            if b > a:
                clean.append({"onset": round(a, 3), "offset": round(b, 3)})
        n_events += len(clean)
        lines.append(json.dumps({"clip_id": cid, "events": clean}))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.jsonl", "\n".join(lines) + "\n")

    # structural validation of what was actually written
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert names == ["predictions.jsonl"], names
        rows = [json.loads(x) for x in zf.read("predictions.jsonl").decode().splitlines()]
    seen = set()
    for r in rows:
        assert r["clip_id"] not in seen, f"duplicate {r['clip_id']}"
        seen.add(r["clip_id"])
        for e in r["events"]:
            assert e["offset"] > e["onset"]
    assert seen == set(ids), f"coverage mismatch: {len(seen)} vs {len(ids)}"
    print(f"OK {out}  clips={len(rows)}  events={n_events} "
          f"({n_events/max(1,len(rows)):.2f}/clip)  root-level predictions.jsonl only")


if __name__ == "__main__":
    main()
