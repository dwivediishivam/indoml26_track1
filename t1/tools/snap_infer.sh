#!/bin/bash
# Inference with each model's SECOND retained checkpoint (snapshot ensemble members): caches -> work/caches/<m>_snap2
source /etc/profile.d/modules.sh; module add cuda12.8/toolkit/12.8.1
cd $HOME/t1_final
export PYTHONPATH=$HOME/t1_final:$HOME/t1_final/deps HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 INDOML_MODEL_DIR=$HOME/indoml-offline/model
for m in "$@"; do
  run=$HOME/indoml-work/runs_t1f/$m
  f=$(python3 -c "import json;r=json.load(open('$run/checkpoints/best.json'))['checkpoints'];print(r[1]['file'] if len(r)>1 else '')")
  [ -z "$f" ] && { echo "SNAP SKIP $m"; continue; }
  for s in val blind; do
    ~/.venvs/indoml/bin/python -u -m models.infer --ckpt "$run/checkpoints/$f" --split $s --out "work/caches/${m}_snap2/$s.npz" --tta --overwrite || echo "SNAP FAIL $m $s"
  done
  echo "SNAP OK $m $f"
done
echo SNAP DONE
