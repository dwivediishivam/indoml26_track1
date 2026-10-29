#!/bin/bash
# Final stage (Mac): pull model caches, run three tunes in parallel, build candidate ZIPs.
#   bash tools/final_stage.sh [MINUTES]      (default 35)
# Tunes:
#   legacy_bank_f4  legacy caches only, fold 4 only        -> honest baseline on fold 4
#   final_f4        legacy + every new cache, fold 4 only  -> needed because m1v/m2/m3/m5/m6 trained on folds 0-3
#   clean_full      legacy + caches never trained on validation (m1_atst, m4_breath), full val + 5-fold CV
# Promote final_f4 or clean_full only if it beats legacy_bank_f4 on fold 4 by a clear margin (~0.02+).
set -uo pipefail
cd "$(dirname "$0")/.."
MIN="${1:-35}"
PY=/Users/dwivediishivam/code/indoml/ai3/.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1
mkdir -p work/caches work/logs
rsync -a -e "ssh -o ControlPath=$HOME/.ssh/cm-omni" user@cluster:~/t1_final/work/caches/ work/caches/ || echo "[final] cluster cache pull failed"
for z in work/caches/*.zip ../colab_out/*.zip; do [ -f "$z" ] && unzip -oq "$z" -d work/caches; done   # Colab zips
echo "[final] caches:"; for d in work/caches/*/; do n=$(basename "$d"); [ -f "$d/val.npz" ] && [ -f "$d/blind.npz" ] && echo "  $n OK" || echo "  $n INCOMPLETE"; done
MODELS=$(for d in work/caches/*/; do [ -f "$d/val.npz" ] && [ -f "$d/blind.npz" ] && basename "$d"; done | tr '\n' ' ')
CLEAN=$(for n in m1_atst m4_breath m7_atst_s77 m8_atst_layermix; do [[ " $MODELS " == *" $n "* ]] && echo -n "$n "; done)
echo "[final] models: $MODELS | clean: $CLEAN"
$PY -m decode.tune --name legacy_bank_f4 --legacy --no-models --minutes 8 --workers 2 --init work/tuned/legacy_bank.json \
    --only-folds 4 --cv none > work/logs/tune_legacy_bank_f4.log 2>&1 &
$PY -m decode.tune --name final_f4 --legacy --models $MODELS --minutes "$MIN" --workers 5 --init work/tuned/legacy_bank.json \
    --only-folds 4 --cv none > work/logs/tune_final_f4.log 2>&1 &
if [ -n "$CLEAN" ]; then
  $PY -m decode.tune --name clean_full --legacy --models $CLEAN --minutes "$MIN" --workers 3 --init work/tuned/legacy_bank.json \
      --weight-grid 0.2 > work/logs/tune_clean_full.log 2>&1 &
fi
wait
for n in legacy_bank_f4 final_f4 clean_full; do
  [ -f "work/tuned/$n.json" ] || continue
  $PY -m assemble.make_submission --config "work/tuned/$n.json" --out "work/submissions/$n.zip" > "work/logs/submit_$n.log" 2>&1 \
    && $PY assemble/check_zip.py "work/submissions/$n.zip" | tail -1
  echo "== $n"; grep -A3 "^  full" "work/logs/tune_$n.log" | head -3; grep "per-fold" "work/logs/tune_$n.log" | head -2
done
