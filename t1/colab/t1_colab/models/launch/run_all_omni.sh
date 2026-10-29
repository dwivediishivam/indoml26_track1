#!/usr/bin/env bash
# Print the sbatch command for every candidate (default), or submit them with --submit.
#   bash models/launch/run_all_omni.sh                      # dry run: print commands only
#   bash models/launch/run_all_omni.sh m1_atst m3_wavlm_large
#   DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/run_all_omni.sh --submit
# Variables exported to the jobs: T1_ROOT, plus DEADLINE / RUN_ROOT / STAGED_DIR / VAL_DIR / BLIND_DIR /
# WORK_DIR / TTA / TRAIN_EXTRA / INFER_RESERVE_MIN when set in the calling shell.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T1_ROOT="$(cd "$HERE/../.." && pwd)"
SUBMIT=0
CANDS=()
for a in "$@"; do
  case "$a" in
    --submit) SUBMIT=1 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option $a" >&2; exit 2 ;;
    *) CANDS+=("$a") ;;
  esac
done
if [[ ${#CANDS[@]} -eq 0 ]]; then
  CANDS=(m1_atst m1v_atst_val m2_beats m3_wavlm_large m4_breath m5_w2vbert)
fi
EXPORTS="ALL,T1_ROOT=$T1_ROOT"
for v in DEADLINE RUN_ROOT STAGED_DIR VAL_DIR BLIND_DIR WORK_DIR TTA TRAIN_EXTRA INFER_RESERVE_MIN; do
  if [[ -n "${!v:-}" ]]; then
    if [[ "${!v}" == *,* ]]; then echo "$v contains a comma, which sbatch --export cannot carry" >&2; exit 2; fi
    EXPORTS+=",$v=${!v}"
  fi
done
for c in "${CANDS[@]}"; do
  cfg="models/configs/$c.yaml"
  [[ -f "$T1_ROOT/$cfg" ]] || { echo "missing $T1_ROOT/$cfg" >&2; exit 2; }
  cmd=(sbatch --job-name="t1f-$c" --chdir="$T1_ROOT" --export="$EXPORTS,CONFIG=$cfg" "$T1_ROOT/models/launch/omni_train.sbatch")
  if [[ $SUBMIT == 1 ]]; then
    "${cmd[@]}"
  else
    printf '%q ' "${cmd[@]}"; echo
  fi
done
if [[ $SUBMIT != 1 ]]; then echo "# dry run: nothing submitted (add --submit)"; fi
