"""Stage the full Vaani noise dataset onto the Colab disk, resumably.

Phase 1: snapshot_download all 182 parquet shards (~17.7 GB).
Phase 2: decode each shard into an append-only float16 waveform blob plus a
         jsonl manifest (one line per clip: offset, n_samples, tier, spans).

Both phases are restart-safe: shards already downloaded are skipped by the hub
cache, and shard-level progress is recorded in done_shards.json so a lost
runtime resumes at the next unprocessed shard.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/content/ai2")
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "stage.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "huggingface_hub", "soundfile", "librosa", "pyarrow"], check=False)
    os.environ["HF_TOKEN"] = Path("/content/.hf_token").read_text().strip()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import snapshot_download

    log("phase1: snapshot_download")
    local = snapshot_download("ARTPARK-IISc/Vaani-Noise-Event-Dataset",
                              repo_type="dataset", token=os.environ["HF_TOKEN"],
                              max_workers=16, allow_patterns=["data/*.parquet"])
    shards = sorted(Path(local).glob("data/*.parquet"))
    log(f"phase1 done: {len(shards)} shards at {local}")

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import io

    blob_path = DATA / "audio.f16"
    manifest_path = DATA / "manifest.jsonl"
    state_path = DATA / "done_shards.json"
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    offset = blob_path.stat().st_size // 2 if blob_path.exists() else 0
    log(f"phase2: resuming with {len(done)} shards done, offset {offset} samples")

    TIERS = {"verified_timestamps": "gold", "unverified_timestamps": "silver",
             "no_timestamps": "bronze"}
    counts = {"gold": 0, "silver": 0, "bronze": 0}
    started = time.time()

    with blob_path.open("ab") as blob, manifest_path.open("a", encoding="utf8") as man:
        for si, shard in enumerate(shards):
            if shard.name in done:
                continue
            try:
                table = pq.read_table(shard)
            except Exception as err:
                log(f"shard {shard.name} unreadable: {err}")
                continue
            cols = table.column_names
            rows = table.to_pylist()
            del table
            for row in rows:
                tier = TIERS.get(row.get("annotationQuality"))
                if tier is None:
                    continue
                audio = row.get("audio")
                try:
                    if isinstance(audio, dict) and audio.get("bytes"):
                        wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                    else:
                        continue
                except Exception:
                    continue
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                n = len(wav)
                if n < 4000 or n > 16000 * 40:
                    continue
                spans = []
                for ev in (row.get("NoiseSubCategoryTimeStamp") or []):
                    try:
                        a, b = float(ev["start"]), float(ev["end"])
                    except Exception:
                        continue
                    if b > a >= 0:
                        spans.append([round(a, 4), round(b, 4), ev.get("category", ""), ev.get("tag", "")])
                if tier != "bronze" and not spans:
                    continue
                blob.write(np.nan_to_num(wav).astype(np.float16).tobytes())
                man.write(json.dumps({
                    "o": offset, "n": n, "t": tier, "s": spans,
                    "c": row.get("NoiseCategory") or [],
                    "lang": row.get("language", ""), "st": row.get("state", ""),
                }, ensure_ascii=False) + "\n")
                offset += n
                counts[tier] += 1
            blob.flush()
            man.flush()
            done.add(shard.name)
            state_path.write_text(json.dumps(sorted(done)))
            if si % 5 == 0 or si == len(shards) - 1:
                hrs = offset / 16000 / 3600
                log(f"shard {si+1}/{len(shards)} {counts} {hrs:.1f}h "
                    f"{(time.time()-started)/60:.1f}min cols={cols[:8]}")
    log(f"phase2 DONE {counts} total {offset/16000/3600:.1f}h")
    (DATA / "stage_done.json").write_text(json.dumps(counts))


if __name__ == "__main__":
    main()
