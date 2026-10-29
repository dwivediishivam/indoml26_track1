# models/: SED training and inference runbook

Pipeline: backbone (`models/backbones`) → masked 2-layer BiGRU → heads. The heads output 24 logits per 20 ms frame in the `common/contract.py` order:
- union presence
- 7 category presence tracks
- union onset and offset
- 7 category onsets
- 7 category offsets

A linear-softmax MIL head gives `clip_logits` [7]. An optional clip tag head can be enabled with `model.tag_head: true`.

Inference writes `work/caches/<candidate>/{val,blind}.npz`, which holds float16 sigmoid probabilities [24, T]. These caches are the input for `decode/` and `assemble/`.

## Files
| file | role |
|---|---|
| `config.py` | strict YAML config: `DEFAULTS`, plus `--set key=value` overrides; unknown keys raise an error |
| `data.py` | tier datasets, `MixtureBatchSampler`, collate function, augmentations |
| `model.py` | `SEDModel`, `build_model(cfg)` |
| `losses.py` | presence BCE/focal, union soft-Dice, RED-style focal boundary loss, MIL, tier weights |
| `predict.py` | sliding windows (10 s / 5 s hop, triangular weights with a 0.3 floor) and shift TTA |
| `monitor.py` | fixed subset, fixed simple per-category decoder, `common.metric.score` |
| `train.py` | two-stage trainer with deadline planning, EMA, top-2 retention, resume, DDP |
| `infer.py` | writes the cache from a run dir or a checkpoint |
| `configs/` | one YAML file per candidate (m1, m1v, m2, m3, m4, m5) |
| `launch/` | `omni_train.sbatch`, `run_all_omni.sh` (dry-run unless `--submit`), `torchrun_node.sh` |

### Data
`data.py` reads these sources:
- **Gold, Silver, Bronze:** the Phase 2 staged blob (`READY`, `manifest.jsonl`, `audio.f16`, default `~/indoml-work/data`).
- **val_natural:** natural validation clips from `data/val/validation` whose fold (`work/folds.json`) is not in `data.val_holdout_folds`.
- **synthetic:** `models.data:build_synthgen`, which wraps `synthgen.build_mixer` and `synthgen.dataset.SynthDataset`. Validation clean speech from `exclude_folds` is never used.

Each batch contains `round(fraction × batch_size)` items per source.

**Crops**
- Crops are event-biased: 80% contain a random whole event and 10% are hard negatives. The default length is 10 s; clips shorter than that are used whole.
- Targets come from `common.targets.frame_targets` on the cropped duration, so events cut by the crop get no boundary target at the cut.
- Bronze clips are used whole, up to `data.bronze_max_seconds`; longer Bronze clips are excluded and the count is logged. Bronze is trained with MIL only.

**Augmentations**
- random gain
- whole-frame time shift that moves the labels with the audio
- label-union mixup within a tier
- FilterAugment-style band gains on the waveform (GPU)
- optional Freq-MixStyle on the STFT
- There is no reverb.

### Loss tier weights
| tier | presence | boundary |
|---|---|---|
| gold | 1.0 | 1.0 |
| silver | 0.5 | 0.1 |
| synthetic | 1.0 | 1.0 |
| val_natural | 1.5 | 1.5 |
| bronze | MIL only | MIL only |

- Silver background frames get ×0.5 weight.
- `loss.channel_multipliers` (for example m4's `human_non_speech: 3.0`) scales that category's presence, onset and offset channels.

## Train
```bash
cd <t1_final root>          # must contain common/ and models/
python -m models.train --config models/configs/m1_atst.yaml --out ~/indoml-work/runs_t1f/m1_atst \
    --deadline 2026-10-17T09:30:00+05:30 \
    [--resume] [--staged-dir ~/indoml-work/data] [--val-dir data/val/validation] [--work-dir work] \
    [--set optim.batch_size=12 --set train.workers=8]
```

**Stages and schedule**
- Stage 1 trains only the new parameters (heads, BiGRU, fusion and mel_cnn) for `optim.freeze_steps` steps, at `head_lr` 1e-3. The encoder is frozen and kept in eval mode.
- Stage 2 unfreezes the encoder at `encoder_lr`, with layer-wise decay `layer_decay^(depth from top)`.
- The optimiser is AdamW with bf16 autocast, gradient clipping and EMA weights.
- The cosine LR schedule is fitted to the planned step budget:
  1. At the start, the budget is `optim.max_steps`.
  2. After a 100-step benchmark in stage 1, the budget is recomputed from the measured seconds per step, the deadline, and the monitor cost. Stage-2 speed is estimated with `train.stage2_slowdown_guess`.
  3. The budget is recomputed again after the stage-2 benchmark and after the first monitor.
- Training stops at the budget, at `deadline − reserve`, or on SIGTERM/SIGINT/SIGUSR1. It then runs a final monitor if there is time and saves `latest.pt`.
- Non-finite gradient steps are skipped and logged. After 20 in a row, training raises an error.

**Monitor**
- The monitor runs at step `train.first_monitor` and then every `train.monitor_every` steps. It uses every clip of fold 4 (natural and synthetic), a fixed decoder (threshold 0.5 per category track plus a union fallback), and `common.metric.score`.
- It evaluates both the student and the EMA weights. The best 2 by `combined` score are kept in `checkpoints/{student,ema}_stepNNNNNNN.pt`, indexed in `checkpoints/best.json`.
- Monitor scores are for ranking only. Full-validation decoding is done in `decode/`.

**Run directory**
- `latest.pt`: resumable state, written atomically every 10 minutes.
- `latest.json`
- `status.json`: the latest log row, plus deadline, rates and best checkpoints.
- `history.jsonl`: every log row.
- `config.resolved.json`
- `checkpoints/`

**Resume**
- `--resume` requires `latest.pt` and an identical resolved config. If anything differs, it prints the differences.
- A non-empty `--out` without `--resume` is refused.

**DDP:** run `torchrun --standalone --nproc_per_node=8 -m models.train ...`. `LOCAL_RANK` turns DDP on automatically. `optim.batch_size` is per GPU, and the LR is not rescaled.

Watch a run:
```bash
cat ~/indoml-work/runs_t1f/m1_atst/status.json
tail -f t1f-m1_atst-<jobid>.out
grep '"monitor"' ~/indoml-work/runs_t1f/m1_atst/history.jsonl
```

## Infer
```bash
python -m models.infer --ckpt ~/indoml-work/runs_t1f/m1_atst --split val   --out work/caches/m1_atst/val.npz   --tta
python -m models.infer --ckpt ~/indoml-work/runs_t1f/m1_atst --split blind --out work/caches/m1_atst/blind.npz --tta
```
**Choosing the checkpoint (`--ckpt`)**
- A run dir uses the top entry of `checkpoints/best.json`.
- Add `--weights ema|student` to restrict the choice to one kind.
- A `latest.pt` path uses `--weights` (default `ema`).

**Options**
- `--tta` averages shifts of 0, +20 and −20 ms. `--tta-shifts-ms` sets the shifts directly; they must be multiples of 20 ms.
- `--val-dir` and `--blind-dir` override the data roots.
- An existing `--out` is refused without `--overwrite`.
- A sidecar `<out>.json` records the checkpoint, weights, step and TTA settings.

## OMNI cluster (1 GPU per job)
```bash
cd ~/indoml/t1_final                                   # wherever t1_final lives on the cluster
bash models/launch/run_all_omni.sh                     # prints sbatch commands only
DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/run_all_omni.sh --submit           # all 6
DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/run_all_omni.sh --submit m3_wavlm_large
```
`omni_train.sbatch` requests account sanand, medium/medium, 1 GPU, 16 CPUs, 64 GB and 04:00:00. It runs offline with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, uses `~/.venvs/indoml/bin/python`, and loads the `cuda12.8/toolkit/12.8.1` module.

**Job flow**
1. The job takes the earlier of the Slurm job end (from `squeue`) and `DEADLINE`. Training stops `INFER_RESERVE_MIN` (default 40) minutes before that time.
2. If `latest.pt` exists, the job resumes from it.
3. After training, the job writes `val` and then `blind` caches with TTA into `$WORK_DIR/caches/<candidate>/`.

**Environment variables**
- `STAGED_DIR`, `VAL_DIR`, `BLIND_DIR`, `WORK_DIR`, `RUN_ROOT` (default `~/indoml-work/runs_t1f`) and `CACHE_DIR` change the paths.
- `TRAIN_EXTRA` passes extra arguments to `models.train`.
- `TTA=0` disables TTA.

## 8-GPU box
```bash
DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/torchrun_node.sh per-gpu m1_atst m1v_atst_val m2_beats m3_wavlm_large m4_breath m5_w2vbert
DEADLINE=2026-10-17T13:00:00+05:30 bash models/launch/torchrun_node.sh ddp m3_wavlm_large 8
```
In per-gpu mode, GPU *i* trains the *i*-th candidate and then infers; logs go to `$RUN_ROOT/logs/<cand>.log`. With 6 jobs on one box, consider `TRAIN_EXTRA="--set train.workers=8"` so the data workers don't overload the CPUs.

## Candidates
| config | backbone | sources (gold/silver/bronze/synth/val_nat) | notes |
|---|---|---|---|
| `m1_atst` | ATST-F strong + mel_cnn | .40/.27/.05/.28/0 | no validation audio in training |
| `m1v_atst_val` | ATST-F strong + mel_cnn | .35/.25/.05/.25/.10 | val_natural folds 0–3 |
| `m2_beats` | BEATs strong + mel_cnn | .35/.25/.05/.25/.10 | |
| `m3_wavlm_large` | WavLM-large (layer mix) + mel_cnn | .35/.25/.05/.25/.10 | encoder LR 2e-5, layer decay 0.9 |
| `m4_breath` | WavLM-base-plus + mel_cnn | .52/.18/0/.30/0 | 4 s crops, batch 32, human_non_speech ×3 |
| `m5_w2vbert` | w2v-BERT 2.0 + mel_cnn | .35/.25/.05/.25/.10 | grad checkpointing, batch 12 |

All candidates hold out fold 4 for monitoring. The synthetic mixer also excludes fold 4's clean speech; config validation enforces this.

**Weight paths** are placeholders under `~/indoml-offline/model`:
- `ATST-F_strong_1.pt`
- `BEATs_strong_1.pt`
- `wavlm-large/`
- `wavlm-base-plus/`
- `w2v-bert-2.0/`

`models/backbones/fetch_weights.sh` populates that directory. To use another location, set `INDOML_MODEL_DIR` or override a path, for example `--set backbone.parts.0.path=/x/wavlm-large`.

## Interface assumptions
- **Backbones:** `models.backbones.build_backbone(name, **kw)` returns a module with these members:
  - `.out_dim`
  - `forward(wav, lengths)`, returning `[B, ceil(N/320), out_dim]`. Any other shape raises an error.
  - `encoder_parameters()`, `new_parameters()`, `layer_groups()`, and optionally `freeze_encoder(bool)`

  The backbone YAML section is passed through as-is: `fusion` takes `parts=[...]`, `proj_dim` and `part_dropout`.
- **Synthetic data:** the synthgen items have the keys `wav` (zero-padded), `length`, `target [24, ceil(S/320)]`, `clip_cats`, `tier='synthetic'`, `weight` and `boundary_weight`. The collate function drops the padded target frames. `SynthDataset` items are deterministic per (seed, epoch, index), and the sampler draws each of the 1M indices at most once per pass.
