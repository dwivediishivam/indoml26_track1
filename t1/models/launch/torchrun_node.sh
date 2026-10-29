#!/usr/bin/env bash
# 8-GPU box launcher (no Slurm). DEADLINE is required: ISO-8601 with offset by which caches must exist.
#
#   per-GPU: one candidate per GPU (GPU i <- i-th candidate), each trains then writes val/blind caches
#     DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/torchrun_node.sh per-gpu m1_atst m2_beats m3_wavlm_large
#   DDP: one candidate on NPROC GPUs (optim.batch_size is per GPU; LR is not rescaled)
#     DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/torchrun_node.sh ddp m3_wavlm_large 8
#
# Optional: PY (default python), RUN_ROOT (default ~/indoml-work/runs_t1f), STAGED_DIR, VAL_DIR, BLIND_DIR,
# WORK_DIR, CACHE_ROOT (default WORK_DIR/caches), INFER_RESERVE_MIN (default 40), TTA (default 1),
# GPU_OFFSET (first GPU index for per-gpu mode, default 0), TRAIN_EXTRA (e.g. "--set train.workers=6").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T1_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$T1_ROOT"
MODE="${1:-}"; shift || true
: "${DEADLINE:?set DEADLINE (ISO-8601 with offset)}"
PY="${PY:-python}"
export PYTHONPATH="$T1_ROOT:$T1_ROOT/deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
RUN_ROOT="${RUN_ROOT:-$HOME/indoml-work/runs_t1f}"
STAGED_DIR="${STAGED_DIR:-$HOME/indoml-work/data}"
VAL_DIR="${VAL_DIR:-$T1_ROOT/data/val/validation}"
BLIND_DIR="${BLIND_DIR:-$T1_ROOT/data/blind/audio}"
WORK_DIR="${WORK_DIR:-$T1_ROOT/work}"
CACHE_ROOT="${CACHE_ROOT:-$WORK_DIR/caches}"
INFER_RESERVE_MIN="${INFER_RESERVE_MIN:-40}"
TTA="${TTA:-1}"
mkdir -p "$RUN_ROOT/logs"

TRAIN_DEADLINE="$(DEADLINE="$DEADLINE" R="$INFER_RESERVE_MIN" "$PY" -c '
import datetime as d, os
u = d.datetime.fromisoformat(os.environ["DEADLINE"])
assert u.tzinfo is not None, "DEADLINE needs a UTC offset"
print((u - d.timedelta(minutes=int(os.environ["R"]))).isoformat(timespec="seconds"))')"
echo "[node] train deadline $TRAIN_DEADLINE (caches by $DEADLINE)"

infer_all() {  # name
  local name=$1 tta=()
  [[ "$TTA" == 1 ]] && tta=(--tta)
  if [[ ! -s "$RUN_ROOT/$name/checkpoints/best.json" ]]; then
    echo "[node] $name: no retained checkpoint; skipping inference" >&2; return 3
  fi
  for split in val blind; do
    "$PY" -u -m models.infer --ckpt "$RUN_ROOT/$name" --split "$split" --out "$CACHE_ROOT/$name/$split.npz" \
      --val-dir "$VAL_DIR" --blind-dir "$BLIND_DIR" --overwrite ${tta[@]+"${tta[@]}"}
  done
}

train_args() {  # name -> fills the TARGS array with args for models.train
  local name=$1
  TARGS=(--config "models/configs/$name.yaml" --out "$RUN_ROOT/$name" --deadline "$TRAIN_DEADLINE"
         --staged-dir "$STAGED_DIR" --val-dir "$VAL_DIR" --work-dir "$WORK_DIR")
  [[ -f "$RUN_ROOT/$name/latest.pt" ]] && TARGS+=(--resume)
  # shellcheck disable=SC2206
  [[ -n "${TRAIN_EXTRA:-}" ]] && TARGS+=($TRAIN_EXTRA)
  return 0
}

case "$MODE" in
  per-gpu)
    [[ $# -ge 1 ]] || { echo "usage: $0 per-gpu CAND [CAND ...]" >&2; exit 2; }
    i="${GPU_OFFSET:-0}"; pids=()
    for name in "$@"; do
      [[ -f "models/configs/$name.yaml" ]] || { echo "missing models/configs/$name.yaml" >&2; exit 2; }
    done
    for name in "$@"; do
      log="$RUN_ROOT/logs/$name.log"
      echo "[node] GPU $i <- $name (log $log)"
      train_args "$name"
      ( set +e
        export CUDA_VISIBLE_DEVICES=$i
        "$PY" -u -m models.train "${TARGS[@]}"; rc=$?
        echo "[node] $name training exit code $rc"
        infer_all "$name"; irc=$?
        [[ $rc -ne 0 ]] && exit $rc
        exit $irc ) >"$log" 2>&1 &
      pids+=($!); i=$((i + 1))
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    exit $fail
    ;;
  ddp)
    name="${1:?usage: $0 ddp CAND [NPROC]}"; nproc="${2:-8}"
    [[ -f "models/configs/$name.yaml" ]] || { echo "missing models/configs/$name.yaml" >&2; exit 2; }
    log="$RUN_ROOT/logs/$name.ddp.log"
    echo "[node] DDP $name on $nproc GPUs (log $log)"
    train_args "$name"
    set +e
    torchrun --standalone --nproc_per_node="$nproc" -m models.train "${TARGS[@]}" >"$log" 2>&1
    rc=$?
    set -e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" infer_all "$name" >>"$log" 2>&1
    exit $rc
    ;;
  *)
    sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
