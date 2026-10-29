#!/usr/bin/env bash
# Colab A100: stage Vaani -> train $CAND (default m6_atst_colab) until DEADLINE-40min -> val/blind caches -> zip.
#   HF_TOKEN=hf_... VAL_ZIP=/content/drive/MyDrive/indoml/validation.zip \
#   BLIND_ZIP=/content/drive/MyDrive/indoml/input_data.zip DEADLINE=2026-10-17T10:10:00+05:30 \
#   bash /content/t1_final/colab/run_colab.sh
# Resumable: re-running skips finished steps (staging is shard-resumable, training uses --resume).
set -euo pipefail
: "${HF_TOKEN:?}" "${VAL_ZIP:?}" "${BLIND_ZIP:?}" "${DEADLINE:?}"
CAND="${CAND:-m6_atst_colab}"; WORKERS="${WORKERS:-8}"
echo "[colab] candidate $CAND"
cd /content/t1_final
pip -q install einops pyarrow librosa soundfile pyyaml huggingface_hub
mkdir -p /content/eval/val /content/eval/blind /content/weights /content/runs data/blind
[ -d /content/eval/val/validation ] || unzip -q "$VAL_ZIP" -d /content/eval/val -x '__MACOSX/*'
[ -d /content/eval/blind/audio ] || unzip -q "$BLIND_ZIP" -d /content/eval/blind -x '__MACOSX/*'
ln -sfn /content/eval/val data/val
ln -sfn /content/eval/blind/audio data/blind/audio
ls data/val/validation/validationMetadata.json >/dev/null
echo "[colab] blind clips: $(ls data/blind/audio | wc -l) (expect 5517)"
[ -s /content/weights/ATST-F_strong_1.pt ] || curl -fL --retry 5 -o /content/weights/ATST-F_strong_1.pt \
  https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/ATST-F_strong_1.pt
echo "[colab] staging Vaani (17.7 GB parquet -> float16 blob); $(date)"
python colab/stage_data.py --out /content/vaani_staged --hf-token "$HF_TOKEN"
export PYTHONPATH=/content/t1_final
TRAIN_DEADLINE="$(python -c "import datetime as d; print((d.datetime.fromisoformat('$DEADLINE') - d.timedelta(minutes=40)).isoformat())")"
RUN="/content/runs/$CAND"
RESUME=(); [ -f "$RUN/latest.pt" ] && RESUME=(--resume)
echo "[colab] training until $TRAIN_DEADLINE; $(date)"
python -u -m models.train --config "models/configs/$CAND.yaml" --out "$RUN" --deadline "$TRAIN_DEADLINE" \
  --staged-dir /content/vaani_staged --set train.workers=$WORKERS ${RESUME[@]+"${RESUME[@]}"} || echo "[colab] train exit $?"
for split in val blind; do
  python -u -m models.infer --ckpt "$RUN" --split "$split" --out "work/caches/$CAND/$split.npz" --tta --overwrite
done
(cd work/caches && zip -qr "/content/${CAND}_caches.zip" "$CAND")
cp "/content/${CAND}_caches.zip" "$(dirname "$VAL_ZIP")/" 2>/dev/null || true
echo "[colab] DONE: /content/${CAND}_caches.zip (also copied next to VAL_ZIP on Drive); $(date)"
