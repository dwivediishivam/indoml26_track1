"""Pre-render K synthetic mixtures to npz shards (for clusters whose data-loader CPUs are too slow).

    python -m synthgen.write_offline_set --out work/synth/offline_v1 --num 20000 \
        [--config my.yaml] [--set widen_prob=0.3 --set sources.noise.gold.weight=0.5] \
        [--exclude-folds 0] [--shard-size 500] [--workers 8] [--save-clean]

Shard i holds clips rendered with rng = default_rng([seed, i]), so the set is reproducible and the run is
resumable (finished shards are skipped). Each shard_XXXXX.npz is uncompressed (memory-mappable):
    wav      float16 [total]      concatenated mixtures
    offsets  int64 [K+1]          clip j = wav[offsets[j]:offsets[j+1]]
    events   JSON str             per clip [[on, off, category, tag], ...] (seconds)
    snr      JSON str             per clip [snr_db, ...] aligned with events
    clean    float16 [total]      only with --save-clean
Read them back with synthgen.dataset.OfflineSynthDataset. config.json records the resolved config.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from common.contract import SR
from synthgen.mixer import build_mixer, deep_merge, load_config, parse_overrides

_MIXER = None


def _init(cfg: dict):
    global _MIXER
    _MIXER = build_mixer(cfg)


def render_shard(job) -> str:
    out, seed, index, count, save_clean = job
    rng = np.random.default_rng([seed, index])
    wavs, cleans, events, snrs = [], [], [], []
    for _ in range(count):
        r = _MIXER.sample(rng)
        wavs.append(r["wav"].astype(np.float16))
        if save_clean:
            cleans.append(r["clean"].astype(np.float16))
        events.append([list(e) for e in r["events"]])
        snrs.append([x["snr_db"] for x in r["info"]])
    arrays = dict(wav=np.concatenate(wavs), offsets=np.cumsum([0] + [len(w) for w in wavs]).astype(np.int64),
                  events=np.array(json.dumps(events)), snr=np.array(json.dumps(snrs)))
    if save_clean:
        arrays["clean"] = np.concatenate(cleans)
    path = Path(out) / f"shard_{index:05d}.npz"
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.partial.npz")  # hidden: never matches shard_*.npz
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return f"{path.name}: {count} clips, {len(arrays['wav']) / SR / 3600:.2f} h"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, required=True, help="total mixtures K")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", help="YAML/JSON merged over synthgen/config_default.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, JSON value")
    ap.add_argument("--exclude-folds", type=int, nargs="*", default=[], help="never use clean refs of these folds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--save-clean", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow a config that differs from out/config.json")
    a = ap.parse_args()

    cfg = load_config(a.config, parse_overrides(a.set))
    if a.exclude_folds:
        folds = sorted(set(cfg["sources"]["speech"]["val_clean"].get("exclude_folds") or []) | set(a.exclude_folds))
        cfg = deep_merge(cfg, {"sources": {"speech": {"val_clean": {"exclude_folds": folds}}}})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(cfg=cfg, seed=a.seed, num=a.num, shard_size=a.shard_size, save_clean=a.save_clean)
    meta_path = out / "config.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != json.loads(json.dumps(meta)) and not a.force:
        raise SystemExit(f"{meta_path} differs from this run's config; use a new --out or --force")
    meta_path.write_text(json.dumps(meta, indent=1))

    n_shards = math.ceil(a.num / a.shard_size)
    jobs = [(str(out), a.seed, i, min(a.shard_size, a.num - i * a.shard_size), a.save_clean) for i in range(n_shards)
            if not (out / f"shard_{i:05d}.npz").exists()]
    print(f"{len(jobs)}/{n_shards} shards to render", flush=True)
    if a.workers > 1:
        with mp.Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
            for msg in pool.imap_unordered(render_shard, jobs):
                print(msg, flush=True)
    else:
        _init(cfg)
        for job in jobs:
            print(render_shard(job), flush=True)


if __name__ == "__main__":
    main()
