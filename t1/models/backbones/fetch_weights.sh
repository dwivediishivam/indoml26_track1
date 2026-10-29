#!/usr/bin/env bash
# Download every checkpoint the backbones need. Run on a machine WITH internet (not on compute nodes),
# then copy the directory to the cluster (e.g. rsync -a ~/indoml-offline/model/ cluster:~/indoml-offline/model/).
#
#   bash models/backbones/fetch_weights.sh            # -> ~/indoml-offline/model
#   INDOML_MODEL_DIR=/path bash models/backbones/fetch_weights.sh
#
# Result:
#   $DEST/ATST-F_strong_1.pt     PretrainedSED release v0.0.1
#   $DEST/BEATs_strong_1.pt      PretrainedSED release v0.0.1
#   $DEST/wavlm-base-plus/       HF snapshot microsoft/wavlm-base-plus
#   $DEST/wavlm-large/           HF snapshot microsoft/wavlm-large
#   $DEST/w2v-bert-2.0/          HF snapshot facebook/w2v-bert-2.0 (HF weights only, skips conformer_shaw.pt)
set -euo pipefail

DEST="${INDOML_MODEL_DIR:-$HOME/indoml-offline/model}"
REL="https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1"
mkdir -p "$DEST"

fetch() {  # url out
  if [ -s "$2" ]; then echo "[skip] $2 exists"; return; fi
  echo "[get ] $1"
  curl -fL --retry 5 --retry-delay 5 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

fetch "$REL/ATST-F_strong_1.pt" "$DEST/ATST-F_strong_1.pt"
fetch "$REL/BEATs_strong_1.pt"  "$DEST/BEATs_strong_1.pt"

# Known-good hash of ATST-F_strong_1.pt (344480550 bytes, verified in Phase 2).
want=fbf2577958e3648d55ee8cea7e0e5260c4505fb0c13964e1ec81bc367cee5eda
got=$(sha256 "$DEST/ATST-F_strong_1.pt")
[ "$got" = "$want" ] && echo "[ok  ] ATST-F_strong_1.pt sha256" || { echo "[FAIL] ATST-F_strong_1.pt sha256 $got"; exit 1; }
echo "[info] BEATs_strong_1.pt sha256 $(sha256 "$DEST/BEATs_strong_1.pt")"

hf_get() {  # repo dir
  local repo=$1 dir=$2
  local pats=("*.json" "*.safetensors" "*.bin" "*.txt" "*.md")
  echo "[hf  ] $repo -> $dir"
  if command -v hf >/dev/null; then
    hf download "$repo" --local-dir "$dir" --include "${pats[@]}"
  elif command -v huggingface-cli >/dev/null; then
    huggingface-cli download "$repo" --local-dir "$dir" --include "${pats[@]}"
  else
    python3 - "$repo" "$dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.md"])
PY
  fi
  [ -f "$dir/config.json" ] || { echo "[FAIL] $dir/config.json missing"; exit 1; }
}

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf_get microsoft/wavlm-base-plus "$DEST/wavlm-base-plus"
hf_get microsoft/wavlm-large     "$DEST/wavlm-large"
hf_get facebook/w2v-bert-2.0     "$DEST/w2v-bert-2.0"

echo "done -> $DEST"
du -sh "$DEST"/* 2>/dev/null || true
