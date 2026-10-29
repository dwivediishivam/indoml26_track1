#!/usr/bin/env bash
#SBATCH --job-name=t1f-train
#SBATCH --account=sanand
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# One candidate per job: train until the deadline, then write val + blind caches.
#   cd <t1_final root>
#   sbatch --job-name=t1f-m1_atst --export=ALL,CONFIG=models/configs/m1_atst.yaml models/launch/omni_train.sbatch
#
# Environment (all optional except CONFIG):
#   CONFIG             models/configs/<candidate>.yaml (relative to T1_ROOT)
#   T1_ROOT            t1_final root (default: the submit directory)
#   DEADLINE           ISO-8601 with offset by which the caches must exist (e.g. 2026-10-17T13:00:00+05:30);
#                      training stops INFER_RESERVE_MIN before min(DEADLINE, Slurm job end)
#   INFER_RESERVE_MIN  minutes reserved for inference (default 40)
#   STAGED_DIR         staged Gold/Silver/Bronze blob (default ~/indoml-work/data)
#   VAL_DIR / BLIND_DIR / WORK_DIR   data roots (default T1_ROOT/data/val/validation, T1_ROOT/data/blind/audio, T1_ROOT/work)
#   RUN_ROOT           run outputs (default ~/indoml-work/runs_t1f); a run dir with latest.pt is resumed
#   CACHE_DIR          cache output dir (default WORK_DIR/caches/<candidate>)
#   TTA                1 = average 0/+20/-20 ms shifts at inference (default 1)
#   TRAIN_EXTRA        extra args for models.train, e.g. "--set optim.batch_size=12"
#   INDOML_PYTHON      interpreter (default ~/.venvs/indoml/bin/python)
set -Eeuo pipefail

: "${CONFIG:?set CONFIG=models/configs/<candidate>.yaml}"
T1_ROOT="${T1_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$T1_ROOT"
if [[ ! -f models/train.py || ! -f common/contract.py ]]; then
  echo "T1_ROOT=$T1_ROOT is not the t1_final root (models/train.py, common/contract.py missing)" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then echo "CONFIG not found: $T1_ROOT/$CONFIG" >&2; exit 2; fi

# CUDA toolkit module, as in Phase 2 scripts/omni_environment.sh (indoml_load_cuda).
if ! type module >/dev/null 2>&1; then
  if [[ ! -r /etc/profile.d/modules.sh ]]; then
    echo 'Cluster module initializer /etc/profile.d/modules.sh is unavailable.' >&2
    exit 2
  fi
  set +u; source /etc/profile.d/modules.sh; set -u
fi
module add cuda12.8/toolkit/12.8.1

PY="${INDOML_PYTHON:-$HOME/.venvs/indoml/bin/python}"
[[ -x "$PY" ]] || { echo "Python not found: $PY" >&2; exit 2; }
export PYTHONPATH="$T1_ROOT:$T1_ROOT/deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export INDOML_MODEL_DIR="${INDOML_MODEL_DIR:-$HOME/indoml-offline/model}"

NAME="$(basename "$CONFIG" .yaml)"
RUN_ROOT="${RUN_ROOT:-$HOME/indoml-work/runs_t1f}"
RUN_DIR="$RUN_ROOT/$NAME"
STAGED_DIR="${STAGED_DIR:-$HOME/indoml-work/data}"
VAL_DIR="${VAL_DIR:-$T1_ROOT/data/val/validation}"
BLIND_DIR="${BLIND_DIR:-$T1_ROOT/data/blind/audio}"
WORK_DIR="${WORK_DIR:-$T1_ROOT/work}"
CACHE_DIR="${CACHE_DIR:-$WORK_DIR/caches/$NAME}"
export JOB_MINUTES="${JOB_MINUTES:-240}" INFER_RESERVE_MIN="${INFER_RESERVE_MIN:-40}" DEADLINE="${DEADLINE:-}"
TTA="${TTA:-1}"

TRAIN_DEADLINE="$("$PY" - <<'PY'
import datetime as d, os, subprocess
now = d.datetime.now().astimezone()
end, source = None, "JOB_MINUTES"
jid = os.environ.get("SLURM_JOB_ID")
if jid:
    out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%e"], capture_output=True, text=True).stdout.strip()
    try:
        end, source = d.datetime.fromisoformat(out).astimezone(), "squeue end time"
    except ValueError:
        end = None
if end is None:
    end = now + d.timedelta(minutes=int(os.environ["JOB_MINUTES"]))
user = os.environ.get("DEADLINE")
if user:
    u = d.datetime.fromisoformat(user)
    if u.tzinfo is None:
        raise SystemExit("DEADLINE must include a UTC offset, e.g. +05:30")
    if u < end:
        end, source = u, "DEADLINE"
deadline = end - d.timedelta(minutes=int(os.environ["INFER_RESERVE_MIN"]))
import sys
print(f"[deadline] end from {source}: {end.isoformat(timespec='seconds')}", file=sys.stderr)
print(deadline.isoformat(timespec="seconds"))
PY
)"
echo "[job] candidate=$NAME run_dir=$RUN_DIR train_deadline=$TRAIN_DEADLINE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RESUME=()
if [[ -f "$RUN_DIR/latest.pt" ]]; then RESUME=(--resume); echo "[job] resuming from $RUN_DIR/latest.pt"; fi
set +e
# shellcheck disable=SC2086
"$PY" -u -m models.train --config "$CONFIG" --out "$RUN_DIR" --deadline "$TRAIN_DEADLINE" \
  --staged-dir "$STAGED_DIR" --val-dir "$VAL_DIR" --work-dir "$WORK_DIR" ${RESUME[@]+"${RESUME[@]}"} ${TRAIN_EXTRA:-}
rc=$?
set -e
echo "[job] training exit code $rc"
if [[ ! -s "$RUN_DIR/checkpoints/best.json" ]]; then
  echo "[job] no retained checkpoint in $RUN_DIR/checkpoints; skipping inference" >&2
  exit $(( rc == 0 ? 3 : rc ))
fi

TTA_FLAG=()
[[ "$TTA" == 1 ]] && TTA_FLAG=(--tta)
for split in val blind; do
  "$PY" -u -m models.infer --ckpt "$RUN_DIR" --split "$split" --out "$CACHE_DIR/$split.npz" \
    --val-dir "$VAL_DIR" --blind-dir "$BLIND_DIR" --overwrite ${TTA_FLAG[@]+"${TTA_FLAG[@]}"}
done
echo "[job] caches written to $CACHE_DIR"
exit $rc
