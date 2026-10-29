#!/usr/bin/env bash
#SBATCH --job-name=t1f-multi
#SBATCH --account=sanand
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=480G
#SBATCH --time=04:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Several candidates share ONE GPU: the account allows few concurrent jobs (medium 2, long 1, short 3),
# but a B200 (192 GB) has memory and compute for 2-3 trainers. Each candidate runs omni_train.sbatch's
# train -> infer(val, blind) flow as a background process; logs go to RUN_ROOT/<cand>.job.log.
#   CANDS="m1v_atst_val m2_beats m4_breath" WORKERS=5 DEADLINE=2026-10-17T10:15:00+05:30 \
#     sbatch --export=ALL models/launch/omni_multi.sbatch
set -uo pipefail
T1_ROOT="${T1_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$T1_ROOT" || exit 2
read -r -a LIST <<< "${CANDS:?set CANDS=\"cand1 cand2 ...\"}"
RUN_ROOT="${RUN_ROOT:-$HOME/indoml-work/runs_t1f}"
mkdir -p "$RUN_ROOT"
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} / ${#LIST[@]} - 1 ))}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
declare -A PIDS
for c in "${LIST[@]}"; do
  CONFIG="models/configs/$c.yaml" T1_ROOT="$T1_ROOT" RUN_ROOT="$RUN_ROOT" \
    TRAIN_EXTRA="--set train.workers=$WORKERS ${TRAIN_EXTRA:-}" \
    bash models/launch/omni_train.sbatch > "$RUN_ROOT/$c.job.log" 2>&1 &
  PIDS[$c]=$!
  echo "[multi] started $c pid=${PIDS[$c]} workers=$WORKERS log=$RUN_ROOT/$c.job.log"
  sleep 20   # stagger model loading / data index warm-up
done
rc=0
for c in "${LIST[@]}"; do
  wait "${PIDS[$c]}"; r=$?
  echo "[multi] $c finished rc=$r"
  (( r != 0 )) && rc=$r
done
exit $rc
