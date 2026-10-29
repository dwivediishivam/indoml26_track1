"""Track 1 inference: noisy WAVs -> predictions.jsonl (+ Codabench submission.zip).

    python main.py INPUT_DIR OUTPUT_DIR [--config configs/clean_plus_a30.json] [--mode auto|replay|audio]

replay  For the official Track 1 test set (5,517 clips) the released per-model probability caches and the
        released snippet-bank detections are decoded with the final configuration. This regenerates the
        submitted predictions (Codabench 995227) exactly.
audio   Every ensemble member that has released weights is run on the input audio, the snippet bank is
        matched against the input, and the fused probabilities are decoded with the same configuration.
        Members without released weights contribute through their released test-set caches when the input is
        the official test set; otherwise their fusion weight is redistributed over the other members.
auto    replay when the input clip IDs are exactly the official test set, audio otherwise.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.join(ROOT, "t1")
CKPT = os.path.join(ROOT, "checkpoints")
# ensemble member -> (released weight file, cache format)
WEIGHTS = {
    "s9000": ("s9000_weights_9000.pt", "legacy"),
    "s1600": ("s1600_weights_1600.pt", "legacy"),
    "m1_atst": ("m1_atst_student_step0018000.pt", "contract"),
    "m4_breath": ("m4_breath_student_step0012000.pt", "contract"),
    "m1g_atst_gold": ("m1g_atst_gold_ema_step0009000.pt", "contract"),
    "m1v_atst_val": ("m1v_atst_val_ema_step0009000.pt", "contract"),
    "m2_beats": ("m2_beats_ema_step0007776.pt", "contract"),
    "m3_wavlm_large": ("m3_wavlm_large_ema_step0009000.pt", "contract"),
    "m5_w2vbert": ("m5_w2vbert_student_step0010000.pt", "contract"),
}


def run(cmd, cwd=T1):
    print("+ " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=T1 + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.call(cmd, cwd=cwd, env=env)
    if r:
        sys.exit(r)


def wav_dir(d):
    if glob.glob(os.path.join(d, "*.wav")):
        return os.path.abspath(d)
    if glob.glob(os.path.join(d, "audio", "*.wav")):
        return os.path.abspath(os.path.join(d, "audio"))
    sys.exit(f"no .wav files in {d} or {d}/audio")


def link_inputs(src):
    """The pipeline reads test audio from t1/data/blind/audio; point it at the input folder."""
    dst = os.path.join(T1, "data", "blind", "audio")
    if os.path.islink(dst) or os.path.isfile(dst):
        os.remove(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def released_ids():
    with open(os.path.join(ROOT, "released", "bank", "blind_events.jsonl")) as f:
        return {json.loads(l)["clip_id"] for l in f if l.strip()}


def finish(zip_path, out):
    shutil.copy(zip_path, os.path.join(out, "submission.zip"))
    shutil.copy(os.path.splitext(zip_path)[0] + ".predictions.jsonl", os.path.join(out, "predictions.jsonl"))
    print(f"wrote {out}/predictions.jsonl and {out}/submission.zip")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "clean_plus_a30.json"))
    ap.add_argument("--mode", choices=("auto", "replay", "audio"), default="auto")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    src = wav_dir(a.input_dir)
    out = os.path.abspath(a.output_dir)
    os.makedirs(out, exist_ok=True)
    ids = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(src, "*.wav"))}
    official = ids == released_ids()
    mode = a.mode if a.mode != "auto" else ("replay" if official else "audio")
    link_inputs(src)
    cfg = json.load(open(a.config))
    py = sys.executable
    work = os.path.join(out, "_work")
    os.makedirs(work, exist_ok=True)

    if mode == "replay":
        if not official:
            sys.exit("replay mode needs the official 5,517-clip test set")
        z = os.path.join(work, "submission.zip")
        run([py, "-m", "assemble.make_submission", "--config", os.path.abspath(a.config), "--out", z,
             "--bank", os.path.join(ROOT, "released", "bank", "blind_events.jsonl")])
        return finish(z, out)

    # audio mode -------------------------------------------------------------------------------------------
    weights = {k: v for k, v in cfg["fusion"]["weights"].items() if v > 0}
    blind = {}
    dev = ["--device", a.device] if a.device else []
    for name in weights:
        cache = os.path.join(work, "caches", name, "blind.npz")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if name in WEIGHTS and os.path.exists(os.path.join(CKPT, WEIGHTS[name][0])):
            ck = os.path.join(CKPT, WEIGHTS[name][0])
            if WEIGHTS[name][1] == "contract":
                run([py, "-m", "models.infer", "--ckpt", ck, "--split", "blind", "--out", cache, "--tta",
                     "--overwrite", *dev])
            else:
                run([py, os.path.join(ROOT, "legacy_sed", "dump_sed.py"), "--checkpoint", ck, "--source", src,
                     "--out", cache, *dev], cwd=ROOT)
            blind[name] = cache
        elif official and os.path.exists(os.path.join(ROOT, "caches", name, "blind.npz")):
            blind[name] = os.path.join(ROOT, "caches", name, "blind.npz")
            print(f"[main] {name}: using its released test-set probabilities", flush=True)
        else:
            print(f"[main] {name}: no released weights; its fusion weight is redistributed", flush=True)
    kept = {k: weights[k] for k in blind}
    total = sum(kept.values())
    cfg["fusion"]["weights"] = {k: (kept.get(k, 0.0) / total) for k in cfg["fusion"]["weights"]}
    for s in cfg["caches"]:
        if s["name"] in blind:
            s["blind"] = blind[s["name"]]
    run_cfg = os.path.join(work, "config.json")
    json.dump(cfg, open(run_cfg, "w"), indent=1)

    bank_dir = os.path.join(T1, "work", "bank")
    os.makedirs(bank_dir, exist_ok=True)
    if not os.path.exists(os.path.join(bank_dir, "snippets.npz")):
        sys.exit("snippet bank not built: place the IndoML validation set at t1/data/val/validation and run\n"
                 "  cd t1 && PYTHONPATH=. python -m bank.build_bank")
    shutil.copy(os.path.join(ROOT, "released", "bank", "calibration.json"), bank_dir)
    run([py, "-m", "bank.detect_blind", "--workers", str(a.workers)] + (["--allow-count"] if len(ids) != 5517 else []))
    shutil.copy(os.path.join(bank_dir, "blind_events.jsonl"), os.path.join(work, "bank_events.jsonl"))

    z = os.path.join(work, "submission.zip")
    cmd = [py, "-m", "assemble.make_submission", "--config", run_cfg, "--out", z,
           "--bank", os.path.join(work, "bank_events.jsonl")]
    if len(ids) != 5517:
        cmd.append("--allow-count")
    run(cmd)
    finish(z, out)


if __name__ == "__main__":
    main()
