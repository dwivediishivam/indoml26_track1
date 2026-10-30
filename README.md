# Multi-Backbone SED Ensemble with Snippet-Bank Matching for Indic Noise Event Detection

**Datathon@IndoML 2026, Track 1: Noise Event Detection** · Team **TDS** (IIIT Delhi):
Shivam Dwivedi, Animish Yadav, Shubhi Jain

| | |
|---|---|
| Final Codabench submission | **995227** (`clean_plus_a30.zip`), leaderboard rank **6** |
| Test score | Event F1 **0.7711** · Segment Dice **0.8232** · combined **1.5943** |
| Model weights and probability caches | https://huggingface.co/dwivediishivam/indoml26_track1 |
| Technical report | [`docs/technical_report.pdf`](docs/technical_report.pdf) (LaTeX source alongside) |

## 1. Task

Locate every noise event (onset and offset) in 16 kHz Indic speech clips from the Vaani corpus. The metric is
**Event F1**, where an event matches when both its onset and its offset are within max(0.2·duration, 50 ms) of the
reference, plus **Segment Dice** at 10 ms resolution. Combined = F1 + Dice, with a maximum of 2. The test set has
5,517 clips: 3,437 natural and 2,080 synthetic.

## 2. Approach in brief

1. **Neural SED ensemble.** Twelve frame-level detectors are fused in the logit domain. Each detector is a
   pretrained audio or speech encoder fused with a mel-CNN, followed by a 2-layer BiGRU and 24 output tracks per
   20 ms frame: union presence, 7 category presence tracks, union onset/offset, and 7 category onset/offset pairs.
   The encoders are ATST-F, BEATs, WavLM-base+, WavLM-large and w2v-BERT 2.0.
2. **Snippet bank for synthetic mixtures.** The organizers' synthetic clips are clean speech plus sample-exact
   copies of noise snippets.
   - We isolate the snippets from the validation pairs (`noisy − clean` at the labelled spans): 206 distinct
     snippets out of 2,213 events.
   - We find them in test audio with FFT normalised cross-correlation and a chunk-lag consistency test.
   - A logistic model, calibrated leave-fold-out on validation, scores each candidate.
3. **Routing decoder.** For each clip, a calibrated synthetic score decides how it is decoded:
   - **bank events**, merged with confident neural events;
   - a **category-aware hysteresis decoder** on the fused probabilities, for natural clips;
   - a **prior** fallback.

   All thresholds are tuned on validation with test-mixture weighting. Every clip is guaranteed at least one event.

## 3. Training data and preprocessing

| Data | Use |
|---|---|
| **Vaani Noise Event Dataset** (`ARTPARK-IISc/Vaani-Noise-Event-Dataset`, 90,637 segments, 154.6 h, CC BY 4.0) | **Gold** (`verified_timestamps`, 11,111 segments, 21.8 h) and **Silver** (`unverified_timestamps`, 61,642 segments, 100.3 h): frame targets from the timestamps. **Bronze** (`no_timestamps`, 17,884 segments, 32.4 h): clip-level tags through a MIL head. Staged once into a float16 blob with a JSONL manifest |
| **IndoML validation set** (2,501 clips: 1,806 natural, 695 synthetic with clean references) | 5 folds (`t1/work/folds.json`); **fold 4 (479 clips: 344 natural + 135 synthetic) is the monitoring and selection hold-out**. Natural clips from folds 0–3 are a training source for the `m1v`, `m2`, `m3`, `m5`, `m6` and `m1g` members. Synthetic pairs provide the snippet bank and the clean speech for synthetic mixing |

**Synthetic-mixture generator** (`t1/synthgen/`). It rebuilds the organizers' construction
`noisy = clean + Σ gᵢ·snippetᵢ` with sample-exact labels. Event counts, gaps (≥200 ms), gains and durations are
measured on the 695 validation synthetic clips (`synthgen/config_default.yaml`).
- **Noise sources:** the validation snippet bank (sampled in proportion to its use) and, for the "widened"
  members, isolated Gold event spans.
- **Speech sources:** validation clean references (held-out folds excluded) and Gold/Silver speech regions ≥50 ms
  from any event.

**Batch composition** (`t1/models/data.py`): every batch draws fixed fractions from each source, for example
Gold 0.40 / Silver 0.27 / Bronze 0.05 / synthetic 0.28 for `m1_atst`.
- Crops are event-biased: 80% contain a whole event and 10% are hard negatives. Default crops are 10 s; the breath
  specialist uses 4 s.

**Augmentations:**
- random gain;
- whole-frame time shifts that move the labels with the audio;
- label-union mixup within a tier;
- FilterAugment-style band gains on the waveform;
- optional Freq-MixStyle on the STFT.

## 4. Models, losses and training

| Member | Encoder (pretrained source) | Training sources | Released weights |
|---|---|---|---|
| `s9000`, `s1600` | WavLM-base+ with learned layer weights, 2-layer BiGRU, 1+7+64 frame tracks, onset/offset heatmaps, attention-MIL (`legacy_sed/`) | Gold 1.0 / Silver 0.55 / Bronze 0.25 tier weights | steps 9,000 and 1,600 |
| `m1_atst` | ATST-F strong (PretrainedSED v0.0.1) + mel-CNN | Gold/Silver/Bronze/synthetic | student, step 18,000 |
| `m4_breath` | WavLM-base+ + mel-CNN; `human_non_speech` ×3; 4 s crops | Gold/Silver/synthetic | student, step 12,000 |
| `m1v_atst_val` | ATST-F strong + mel-CNN | + validation natural, folds 0–3 | EMA, step 9,000 |
| `m1g_atst_gold` | ATST-F strong + mel-CNN, Gold continuation | + validation natural, folds 0–3 | EMA, step 9,000 |
| `m2_beats` | BEATs strong (PretrainedSED v0.0.1) + mel-CNN | widened synthetic + validation natural | EMA, step 7,776 |
| `m3_wavlm_large` | WavLM-large + mel-CNN | widened synthetic + validation natural | EMA, step 9,000 |
| `m5_w2vbert` | w2v-BERT 2.0 + mel-CNN | widened synthetic + validation natural | student, step 10,000 |
| `m6_atst_colab` | ATST-F strong + mel-CNN | + validation natural | test-set probabilities (`caches/`) |
| `m7_atst_s77` | ATST-F strong + mel-CNN, seed 77 | Gold/Silver/Bronze/synthetic | test-set probabilities (`caches/`) |
| `m8_atst_layermix` | ATST-F strong with all-layer mixing + mel-CNN, seed 55 | Gold/Silver/Bronze/synthetic | test-set probabilities (`caches/`) |

Every member's YAML is in `t1/models/configs/`.

**Losses** (`t1/models/losses.py`):
- presence: BCE/focal per track;
- union: soft-Dice;
- boundaries: RED-style focal loss on Gaussian onset/offset targets;
- Bronze clips: linear-softmax MIL;
- tier weights (presence / boundary): Gold 1.0 / 1.0, Silver 0.5 / 0.1, synthetic 1.0 / 1.0, validation natural
  1.5 / 1.5. Silver background frames are weighted ×0.5.

**Training** (`t1/models/train.py`):
1. **Stage 1** trains the new parameters (heads, BiGRU, fusion, mel-CNN) with the encoder frozen, for 1,000–1,500
   steps at learning rate 1e-3.
2. **Stage 2** unfreezes the encoder at 1e-5 to 2e-5 with layer-wise learning-rate decay of 0.85–0.9.

AdamW, bf16, gradient clipping and EMA weights throughout. The cosine schedule is fitted to a wall-clock deadline
from measured step times. A monitor scores fold 4 every few hundred steps and keeps the two best checkpoints
(student or EMA).

**Inference** (`t1/models/infer.py`): 10 s windows with a 5 s hop, triangular overlap weights (floor 0.3), and
time-shift test-time augmentation over {0, +20, −20} ms. The result is a float16 [24 × T] probability cache per clip.

**Decoding** (`t1/decode/`, `t1/assemble/`). The final configuration is `configs/clean_plus_a30.json`:
- **Fusion:** logit averaging of the 12 caches. 70% of the mass goes to the six members never trained on
  validation natural clips (`clean`), and 5% each to the other six (`plus_a30`).
- **Natural decoder:** hysteresis 0.85 / 0.61, median filter of 5 frames, minimum duration 40 ms, merge gap 160 ms,
  padding −20 / −30 ms, and category-track splitting of fused blobs (threshold 0.30, cover 0.90, IoU 0.30).
- **Bank merge:** synthetic threshold 0.32, bank confidence 0.56, margin 0.18, neural confidence 0.64, gap 200 ms,
  at most 5 events.
- **Tuning:** `t1/decode/tune.py` searches these settings with Optuna TPE on validation, weighting synthetic clips
  ×1.57 to match the test mixture (`_tm`), and reports a 5-fold CV estimate. `_f4` configs are tuned on fold 4 only.

## 5. Results

| Submission | Configuration | Validation (2,501) | Fold-4 hold-out (479) | **Test** |
|---|---|---|---|---|
| 938975 (Phase 1) | `s9000` + `s1600`, 0.8 / 0.2 | 1.134 | — | 1.2129 |
| 995013 | `interim_clean` | 1.4682 | 1.4662 | 1.5803 |
| 995109 | `final_f4_tm` | 1.5407 | 1.5120 | 1.5854 |
| 995114 | `final_f4` | 1.5406 | 1.5117 | 1.5857 |
| 995138 | `clean_full_tm` | 1.4824 | 1.4650 | 1.5875 |
| **995227** | **`clean_plus_a30`** | 1.5328 | 1.4795 | **1.5943** (F1 0.7711, Dice 0.8232) |

The validation columns for ensembles with validation-trained members include training folds; fold 4 is held
out. The final decode routes 1,974 test clips through the bank, 3,482 through the natural decoder and 61 through
the prior, emitting 13,268 events in total.

Findings:
- **The snippet bank transfers to the test set.** It flags 1,974 test clips as synthetic (the test set has 2,080), and
  Event F1 rose from 0.47 for our Phase 1 entry to 0.77.
- **The neural ensemble carries the natural clips.** The category-aware decoder splits overlapping events into
  separate tracks, which matters because 30% of consecutive natural event pairs overlap.
- **Members never trained on validation natural clips gave the most reliable test gains.** `clean_full_tm`, with six such
  members, scores lower on validation than `final_f4_tm` but higher on the test set. Adding the other six at 30% total
  weight (`clean_plus_a30`) gave the best test score.

## 6. Environment and hardware

| | Training | Inference |
|---|---|---|
| Python | 3.10 (cluster); Google Colab runtime (`s9000`, `s1600`, `m6`, `m7`, `m8`) | 3.10–3.11 tested |
| PyTorch / CUDA | 2.9.1+cu128 / CUDA 12.8 (cluster) | 2.9.1 (CPU or cu128) |
| GPU | 1× NVIDIA B200, 183,359 MiB: `m1v`, `m2`, `m4` (job `t1f-long`); `m3`, `m5` (job `t1f-med`). 1× NVIDIA H200 MIG 3g.71gb slice: `m1`, `m1g`. NVIDIA A100 (Google Colab): `s9000`, `s1600`. Google Colab GPU runtime: `m6`, `m7`, `m8` | any CUDA GPU with ≥16 GB, or CPU |
| Cluster | SLURM, 16 CPU cores and 128 GB RAM per GPU job; launchers in `t1/models/launch/` | — |

System dependencies: `libsndfile` (bundled in the `soundfile` wheels).

**Pretrained dependencies.** `bash t1/models/backbones/fetch_weights.sh` downloads them to `$INDOML_MODEL_DIR`:
- ATST-F strong and BEATs strong: `ATST-F_strong_1.pt` and `BEATs_strong_1.pt` from the PretrainedSED release v0.0.1
  (https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1)
- `microsoft/wavlm-base-plus`, `microsoft/wavlm-large`, `facebook/w2v-bert-2.0` (Hugging Face)

## 7. Setup

```bash
git clone https://github.com/dwivediishivam/indoml26_track1.git && cd indoml26_track1
python -m venv .venv && source .venv/bin/activate
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # or CPU wheels
pip install -r requirements_inference.txt              # add requirements_train.txt for training
```

## 8. Inference

**Regenerate the final submission** on the official test set (5,517 WAVs). The output is identical to the
uploaded `predictions.jsonl` of 995227:
```bash
python checkpoints/download_checkpoints.py --caches     # released probability caches (12 members)
python main.py TEST_AUDIO_DIR OUT_DIR --mode replay     # or: bash scripts/reproduce_submission.sh TEST_AUDIO_DIR OUT_DIR
```

**Run on any audio:**
```bash
python checkpoints/download_checkpoints.py --all
bash t1/models/backbones/fetch_weights.sh                # pretrained encoders -> $INDOML_MODEL_DIR
# snippet bank: place the IndoML validation set at t1/data/val/validation, then
(cd t1 && PYTHONPATH=. python -m bank.build_bank)        # 206 snippets from 2,213 validation events
python main.py INPUT_DIR OUT_DIR                          # --mode auto picks replay for the official test set
```

In `audio` mode every member with released weights runs on the input, the bank is matched against it, and the
same configuration decodes the fused probabilities. On the official test set `m6`, `m7` and `m8` contribute
through their released test-set probabilities. For other inputs their fusion weight is spread over the remaining
members.

- **Input:** a folder of 16 kHz mono WAVs. An `audio/` sub-folder, as in the Codabench `input_data` ZIP, is
  accepted.
- **Output:** `OUT_DIR/predictions.jsonl`, one line per clip, `{"clip_id": "...", "events": [{"onset": s, "offset": s}, ...]}`,
  and `OUT_DIR/submission.zip` with `predictions.jsonl` at the root. Event times are in seconds, rounded to
  milliseconds; every clip has at least one event.

## 9. Training

```bash
pip install -r requirements_train.txt
# Vaani parquet shards -> staged blob (manifest.jsonl + audio.f16); see t1/colab/stage_data.py
cd t1
PYTHONPATH=. python -m bank.build_bank                          # snippet bank from validation
PYTHONPATH=. python -m bank.calibrate --workers 16               # leave-fold-out logistic calibration
python -m models.train --config models/configs/m1_atst.yaml --out runs/m1_atst \
    --staged-dir <staged Vaani> --val-dir data/val/validation --work-dir work      # one member (any YAML)
python -m models.infer --ckpt runs/m1_atst --split blind --out work/caches/m1_atst/blind.npz --tta
python -m models.infer --ckpt runs/m1_atst --split val   --out work/caches/m1_atst/val.npz   --tta
PYTHONPATH=. python -m decode.tune --name clean_full_tm --legacy --models m1_atst m4_breath m7_atst_s77 m8_atst_layermix \
    --minutes 35 --weight-grid 0.2                               # decoder + fusion search (T1_SYN_WEIGHT=1.57)
PYTHONPATH=. python -m assemble.make_submission --config work/tuned/clean_full_tm.json --out work/submissions/x.zip
```

The SLURM launchers (`t1/models/launch/`) and the Colab runner (`t1/colab/`) are the ones used for the actual
runs. `legacy_sed/train_sed.py` trains the WavLM-base+ members `s9000` and `s1600`. `t1/RUNBOOK.md` records the
full lane schedule.

## 10. Repository layout

```
main.py                  inference entry point (replay / audio)
t1/common/               I/O contract (24 channels, 20 ms), exact metric, folds, frame targets
t1/models/               encoders, SED model, losses, trainer, inference, configs/, launch/
t1/synthgen/             synthetic-mixture generator matched to the organizers' construction
t1/bank/                 snippet bank: build, matched-filter detection, calibration
t1/decode/, t1/assemble/ fusion, decoders, bank merge, tuner, submission builder
legacy_sed/              WavLM-base+ SED (s9000, s1600): training, inference, decoders
configs/                 final and earlier tuned configurations (cache paths relative to t1/)
released/bank/           bank calibration and test-set bank detections used by the submission
checkpoints/             download script and checkpoint table (weights on Hugging Face)
scripts/                 reproduce_submission.sh, run_inference.sh
docs/                    technical report (PDF + LaTeX)
```

## 11. References

1. S. Pulikodan et al., "VAANI: Capturing the language landscape for an inclusive digital India," arXiv:2603.28714, 2026.
2. F. Schmid et al., "Effective Pre-Training of Audio Transformers for Sound Event Detection," ICASSP 2025 (PretrainedSED: ATST-F / BEATs strong).
3. X. Li, N. Shao, X. Li, "Self-Supervised Audio Teacher-Student Transformer for Both Clip-Level and Frame-Level Tasks" (ATST-Frame), IEEE/ACM TASLP, 2024.
4. S. Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers," ICML 2023.
5. S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE JSTSP, 2022.
6. Seamless Communication et al., "Seamless: Multilingual Expressive and Streaming Speech Translation" (w2v-BERT 2.0), 2023.
7. A. Mesaros, T. Heittola, T. Virtanen, "Metrics for Polyphonic Sound Event Detection," Applied Sciences, 2016.
8. H. Nam et al., "FilterAugment: An Acoustic Environmental Data Augmentation Method," ICASSP 2022.
9. T. Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework," KDD 2019.
10. A. Tarvainen, H. Valpola, "Mean teachers are better role models" (EMA weights), NeurIPS 2017.
