#!/bin/bash
# Inference with each run's CURRENT best retained checkpoint -> work/caches_early/<m>/{val,blind}.npz
source /etc/profile.d/modules.sh; module add cuda12.8/toolkit/12.8.1
cd $HOME/t1_final
export PYTHONPATH=$HOME/t1_final:$HOME/t1_final/deps HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 INDOML_MODEL_DIR=$HOME/indoml-offline/model
for m in "$@"; do
  run=$HOME/indoml-work/runs_t1f/$m
  f=$(python3 -c "import json;print(json.load(open('$run/checkpoints/best.json'))['checkpoints'][0]['file'])")
  cp "$run/checkpoints/$f" /tmp/early_$m.pt   # freeze it: the trainer may delete it while we read
  for s in val blind; do
    ~/.venvs/indoml/bin/python -u -m models.infer --ckpt /tmp/early_$m.pt --split $s --out "work/caches_early/$m/$s.npz" --tta --overwrite || echo "EARLY FAIL $m $s"
  done
  echo "EARLY OK $m $f"
done
echo EARLY DONE
