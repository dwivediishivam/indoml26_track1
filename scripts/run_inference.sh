#!/usr/bin/env bash
# Full inference on any folder of 16 kHz WAVs. Usage: bash scripts/run_inference.sh INPUT_DIR OUT_DIR
# Needs: pretrained backbones (bash t1/models/backbones/fetch_weights.sh) and the snippet bank
# (cd t1 && PYTHONPATH=. python -m bank.build_bank, with the validation set at t1/data/val/validation).
set -euo pipefail
cd "$(dirname "$0")/.."
python checkpoints/download_checkpoints.py --all
python main.py "$1" "$2" --mode auto
