#!/usr/bin/env bash
# Regenerate the final Track 1 submission (Codabench 995227) from the released caches and bank records.
# Usage: bash scripts/reproduce_submission.sh TEST_AUDIO_DIR OUT_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
python checkpoints/download_checkpoints.py --caches
python main.py "$1" "$2" --mode replay
