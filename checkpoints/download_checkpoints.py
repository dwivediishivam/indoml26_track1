"""Download the Track 1 checkpoints and released probability caches from Hugging Face (SHA-256 verified).

    python checkpoints/download_checkpoints.py --caches        # caches only: enough to regenerate the submission
    python checkpoints/download_checkpoints.py --weights       # model weights for inference on new audio
    python checkpoints/download_checkpoints.py --all
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "dwivediishivam/indoml26_track1"
WEIGHTS = {
    "s9000_weights_9000.pt": "a3acdf3711e314eeb203c346c184255b6ec462ef91b5b3e8115714b8531300d4",
    "s1600_weights_1600.pt": "4ad8a99a114034e420fb5667b00c7a9f005607cc5865a2ddc6fcba669aeb3ff8",
    "m1_atst_student_step0018000.pt": "22593e67928cd982e39e7449c9902253af850132800e4daa1bcedc5fe8e3b0fe",
    "m4_breath_student_step0012000.pt": "87a98690a0afc47e6314ca6990ef76e7399fc1e782e17f67c954f003ca443028",
    "m1g_atst_gold_ema_step0009000.pt": "20b800abd61ba3af6c7cd08e8c0f47017b68b9aa9be04224908673f4b9f886cc",
    "m1v_atst_val_ema_step0009000.pt": "dcc28aa5d5b18f26a3864a84975d791ad72a5ec11d58fe1b0bd117c14ec14f82",
    "m2_beats_ema_step0007776.pt": "bd8b426f68a468186966c916f7f3fd01742305d6ce5051d6262f8fdea25d23de",
    "m3_wavlm_large_ema_step0009000.pt": "01ea580186e65c705f5c0140482305b33fb745ece0bf657662d572abaab676bd",
    "m5_w2vbert_student_step0010000.pt": "75e770db43df5ca1c6535190bcfbd1665c6a3c0f8307a9f792b6671d601f7733",
}
MEMBERS = ["s9000", "s1600", "m1_atst", "m4_breath", "m7_atst_s77", "m8_atst_layermix", "m5_w2vbert",
           "m6_atst_colab", "m1v_atst_val", "m2_beats", "m3_wavlm_large", "m1g_atst_gold"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--caches", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.weights or a.caches or a.all):
        a.caches = True
    if a.weights or a.all:
        for n, h in WEIGHTS.items():
            dst = os.path.join(HERE, n)
            if not os.path.exists(dst):
                hf_hub_download(REPO, f"checkpoints/{n}", local_dir=ROOT)
            if sha256(dst) != h:
                sys.exit(f"SHA-256 mismatch for {n}")
            print("ok", n)
    if a.caches or a.all:
        for m in MEMBERS:
            for f in ("blind.npz", "val.npz"):
                hf_hub_download(REPO, f"caches/{m}/{f}", local_dir=ROOT)
        print("ok caches (12 members x {val, blind})")


if __name__ == "__main__":
    main()
