#!/usr/bin/env bash
#SBATCH --job-name=t1f-smoke
#SBATCH --account=sanand
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Pre-flight: every candidate trains 30 steps (10 frozen + 20 unfrozen) at batch 4, then writes a
# validation cache. Catches config, weight-path, data, shape and loss bugs before the long runs.
#   cd <t1_final root>; sbatch models/launch/smoke.sbatch            # all candidates
#   CANDS="m1_atst m2_beats" sbatch --export=ALL models/launch/smoke.sbatch
set -uo pipefail
T1_ROOT="${T1_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$T1_ROOT" || exit 2
if ! type module >/dev/null 2>&1; then set +u; source /etc/profile.d/modules.sh; set -u; fi
module add cuda12.8/toolkit/12.8.1
PY="${INDOML_PYTHON:-$HOME/.venvs/indoml/bin/python}"
export PYTHONPATH="$T1_ROOT:$T1_ROOT/deps${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export INDOML_MODEL_DIR="${INDOML_MODEL_DIR:-$HOME/indoml-offline/model}"
SCRATCH="${SCRATCH_DIR:-$HOME/indoml-work/smoke_t1f/$SLURM_JOB_ID}"
read -r -a LIST <<< "${CANDS:-m1_atst m1v_atst_val m2_beats m3_wavlm_large m4_breath m5_w2vbert}"
declare -A RESULT
for c in "${LIST[@]}"; do
  echo "===== $c"
  DEADLINE="$(date -d '+15 minutes' --iso-8601=seconds)"
  # Full-validation inference takes minutes, so it is exercised once (first candidate) unless INFER_ALL=1.
  infer_ok=true
  if "$PY" -u -m models.train --config "models/configs/$c.yaml" --out "$SCRATCH/$c" --deadline "$DEADLINE" \
       --set optim.max_steps=30 --set optim.freeze_steps=10 --set optim.batch_size=4; then
    if [[ "$c" == "${LIST[0]}" || "${INFER_ALL:-0}" == 1 ]]; then
      "$PY" -u -m models.infer --ckpt "$SCRATCH/$c" --split val --out "$SCRATCH/$c/cache_val.npz" || infer_ok=false
    fi
  else
    infer_ok=false
  fi
  if $infer_ok; then
    RESULT[$c]=PASS
  else
    RESULT[$c]=FAIL
  fi
done
echo "===== summary"
for c in "${LIST[@]}"; do echo "$c ${RESULT[$c]}"; done
