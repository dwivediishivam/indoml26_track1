"""Resumable Vaani staging to a memory-mapped waveform blob.

This preserves the successful ai2 data format while adding an explicit manifest
version and source-shard checks. It is safe to restart after a pre-emption.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

SR = 16000
TIERS = {"verified_timestamps": "gold", "unverified_timestamps": "silver",
         "no_timestamps": "bronze"}


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix+".partial")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--repo", default="ARTPARK-IISc/Vaani-Noise-Event-Dataset")
    parser.add_argument("--snapshot-dir", help="Downloaded dataset directory; avoids all network access")
    args = parser.parse_args()
    if not args.snapshot_dir and not args.hf_token:
        raise SystemExit("Provide --hf-token or HF_TOKEN after accepting dataset access")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "READY").exists():
        print("Training data already staged", flush=True); return
    if args.snapshot_dir:
        snapshot = Path(args.snapshot_dir)
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(args.repo, repo_type="dataset", token=args.hf_token,
                                          allow_patterns=["data/*.parquet"], max_workers=24))
    shards = sorted(snapshot.glob("data/*.parquet"))
    if not shards:
        raise RuntimeError("No parquet shards downloaded")
    import pyarrow.parquet as pq
    import soundfile as sf
    import librosa
    state_file = out / "stage_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"done": [], "audio_bytes": 0, "manifest_bytes": 0}
    done = set(state["done"])
    blob_path, manifest_path = out / "audio.f16", out / "manifest.jsonl"
    if "audio_bytes" not in state:
        raise RuntimeError("Legacy staging state lacks transaction offsets; use a new output directory")
    # Roll back an interrupted shard before decoding it again.
    for path, committed in ((blob_path, state["audio_bytes"]), (manifest_path, state["manifest_bytes"])):
        if path.exists():
            if path.stat().st_size < committed: raise RuntimeError(f"Truncated staging file: {path}")
            with path.open("r+b") as handle: handle.truncate(committed)
        elif committed: raise RuntimeError(f"Missing staging file: {path}")
    offset = state["audio_bytes"] // 2
    counts = state.get("counts", {"gold": 0, "silver": 0, "bronze": 0})
    print(f"Staging {len(shards)} local shards; {len(done)} already committed", flush=True)
    with blob_path.open("ab") as blob, manifest_path.open("a", encoding="utf8") as manifest:
        for shard in shards:
            if shard.name in done:
                continue
            for row in pq.read_table(shard).to_pylist():
                tier = TIERS.get(row.get("annotationQuality"))
                audio = row.get("audio") or {}
                if tier is None or not audio.get("bytes"):
                    continue
                try:
                    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                except RuntimeError:
                    continue
                if wav.ndim == 2: wav = wav.mean(axis=1)
                if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                spans = []
                for event in row.get("NoiseSubCategoryTimeStamp") or []:
                    try:
                        start, end = float(event["start"]), float(event["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if end > start >= 0:
                        spans.append([start, end, event.get("category", ""), event.get("tag", "")])
                if tier != "bronze" and not spans: continue
                wav = np.nan_to_num(wav).astype(np.float16)
                blob.write(wav.tobytes())
                manifest.write(json.dumps({"offset": offset, "samples": len(wav), "tier": tier,
                                           "spans": spans, "categories": row.get("NoiseCategory") or []}) + "\n")
                offset += len(wav); counts[tier] += 1
            blob.flush(); manifest.flush()
            os.fsync(blob.fileno()); os.fsync(manifest.fileno())
            done.add(shard.name)
            atomic_json(state_file, {"version": 3, "done": sorted(done), "counts": counts,
                                    "audio_bytes": blob_path.stat().st_size, "manifest_bytes": manifest_path.stat().st_size})
            print(f"Staged {len(done)}/{len(shards)} shards; {offset/SR/3600:.2f} hours; {counts}", flush=True)
    atomic_json(out / "READY", {"version": 3, "hours": offset / SR / 3600, "counts": counts})


if __name__ == "__main__":
    main()
