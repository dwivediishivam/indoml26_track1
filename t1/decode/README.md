# decode/ + assemble/ runbook

Run everything from `t1_final/`. Pure numpy. scipy, optuna and soundfile are optional and used when installed:
- without scipy, a numpy median filter is used;
- without optuna, the search is seeded random search plus coordinate refinement;
- without soundfile, durations come from a RIFF header parser.

Run `pip install optuna` for TPE.

| file | role |
|---|---|
| `decode/legacy.py` | proven ai2 decoder ported to the 24-ch contract (`BEST` = decoder_search.json `best.decoder`) |
| `decode/decoders.py` | `legacy`, `hysteresis_per_category`, `csebb`, `peaks` + post-steps (`run`, `postprocess`, `synthetic_prior`, `fallback_event`) |
| `decode/fusion.py` | cache loading (legacy 74-ch → `from_legacy`), per-clip weighted prob/logit fusion, NaN-aware, aligned to `n_frames(duration)` |
| `decode/merge.py` | `combine(neural_events, bank_record, cfg)`: bank / prior / natural routing |
| `decode/pipeline.py` | the one prediction path shared by tuning and submission |
| `decode/fastmetric.py` | exact per-clip re-implementation of `common/metric.py` (tuner cross-checks against it at the end) |
| `decode/tune.py` | search CLI (multiprocessing, Optuna TPE or random+coordinate, CV report, atomic best-config saves) |
| `assemble/make_submission.py` | blind decode → validated `predictions.jsonl` → ZIP + stats |
| `assemble/check_zip.py` | independent stdlib-only ZIP validator |

## Step 1: legacy caches + bank events (no new models)

Prerequisites:
- `/Users/dwivediishivam/code/indoml/ai2/work/probs_{s9000,s1600}.npz` (validation)
- `probs_blind_{s9000,s1600}.npz` (blind)
- `work/bank/{val,blind}_events.jsonl` from `bank/`. If missing, every clip is decoded as natural.

```bash
# 0. optional smoke test (200 clips, 2 minutes)
python -m decode.tune --name smoke --legacy --no-models --limit 200 --minutes 2 --workers 4

# 1. tune: the first warm-start trial is the proven decoder (0.8/0.2 prob blend, bank off) ~= 1.141
python -m decode.tune --name legacy_bank --legacy --no-models --minutes 30
#    narrower and faster: --families legacy,hysteresis_per_category ; per-category biases: --percat
#    CV by independent per-fold re-tuning instead of selection: --cv retune --cv-frac 0.4

# 2. build + check the submission
python -m assemble.make_submission --config work/tuned/legacy_bank.json --out work/submissions/legacy_bank.zip
python assemble/check_zip.py work/submissions/legacy_bank.zip
```

What to read in the tune log:
- `NEW BEST` lines: the full objective with natural/synthetic breakdowns and routes.
- The final block:
  - full, natural and synthetic metrics;
  - in-sample per-fold scores;
  - the 5-fold CV mean and per-fold scores (tune on 4 folds, score the 5th), plus the optimism gap;
  - `verify common.metric.score ... OK`.

If the CV mean sits well below the full score, prefer a simpler family or a config tuned with `--cv retune`.

## Step 2: add the new model caches

Model caches live at `work/caches/<model>/{val,blind}.npz` (24 channels, 20 ms).

```bash
python -m decode.tune --name step2 --legacy --models beats atst --minutes 45 \
    --init work/tuned/legacy_bank.json --weight-grid 0.1
python -m assemble.make_submission --config work/tuned/step2.json --out work/submissions/step2.zip
python assemble/check_zip.py work/submissions/step2.zip
```

Notes:
- `--models` with no names uses every `work/caches/*/val.npz`.
- `--cache NAME=VAL[,BLIND]` adds arbitrary contract caches; `--legacy-cache NAME=VAL[,BLIND]` adds arbitrary legacy caches.
- `--weights s9000=0.6,beats=0.4` sets the initial weights. `--freeze-weights` (with `--fusion-mode prob|logit`) skips the weight search.
- `--init` warm-starts from an earlier config's `flat_params`.
- `make_submission --blind NAME=PATH` overrides a blind cache path. `--bank PATH` overrides the blind bank file.

## Tuner CLI (all flags)

- `--name` (required)
- caches: `--legacy`, `--legacy-cache`, `--models [...]`, `--no-models`, `--cache`
- fusion weights: `--weights`, `--freeze-weights`, `--fusion-mode`, `--weight-grid STEP`
- decoders: `--families legacy,hysteresis_per_category,csebb,peaks`, `--percat`
- bank: `--bank`, `--bank-blind`, `--no-bank`
- budget and parallelism: `--minutes`, `--workers`, `--start-method fork|spawn`, `--seed`
- search: `--sampler auto|tpe|random`, `--refine-frac` (final share of the time spent on coordinate refinement), `--init`
- CV: `--cv select|retune|none`, `--cv-frac`
- debugging: `--limit N`, `--no-headers`, `--no-verify`

Outputs:
- `work/tuned/<name>.json`, rewritten atomically after every improvement;
- `work/tuned/<name>.trials.jsonl`, one line per trial with the per-(fold, subset) tp/npred/nref/dice sums, so CV can be recomputed offline.

## Config JSON (`work/tuned/<name>.json`)

```json
{
  "name": "legacy_bank",
  "caches": [{"name": "s9000", "format": "legacy", "val": "...probs_s9000.npz", "blind": "...probs_blind_s9000.npz"}],
  "fusion": {"weights": {"s9000": 0.8, "s1600": 0.2}, "mode": "prob"},
  "natural":   {"family": "legacy", "params": {"hi": 0.9, "lo": 0.9, "median": 3, "...": "..."},
                "post": {"edge_tol": 0.0, "fallback_len": 1.0}},
  "synthetic": null,
  "merge": {"bank_on": true, "synth_thr": 0.5, "bank_p_thr": 0.5, "margin": 0.2, "neural_conf": 0.7,
            "prior_mode": "drop", "prior_thr": 1.01, "prior_conf": 0.3, "gap": 0.2, "max_events": 5},
  "bank": {"val": "work/bank/val_events.jsonl", "blind": "work/bank/blind_events.jsonl"},
  "flat_params": {"...": "search-space values (used by --init)"},
  "metrics": {"full": {}, "natural": {}, "synthetic": {}, "per_fold": [], "routes": {}, "verified_exact": {}},
  "val_counts": {"pred_hist_natural": {}, "pred_hist_synthetic": {}, "ref_hist_natural": {}, "ref_hist_synthetic": {}},
  "cv": {"mode": "select", "mean": 0.0, "std": 0.0, "per_fold": [{"fold": 0, "trial": 0, "train": 0.0, "test": 0.0}]},
  "search": {"trial": 0, "trials": 0, "elapsed_s": 0, "objective": 0.0}
}
```

`synthetic: null` means the synthetic route reuses the natural decoder config. The synthetic prior is applied in `merge`, not in the decoder.

Decoder `post` keys:
- `edge_tol`: snap onsets below it to 0 and offsets above `duration - edge_tol` to `duration`.
- `min_score`, `min_len`.
- `fallback_len`: length of the max-mean union window used when nothing is detected; 0 means the whole clip.
- `prior`, `prior_gap`, `prior_max`, `prior_mode`: an optional in-decoder synthetic prior.

Merge routing:
- **bank**: `synthetic_score >= synth_thr`. Output is the bank events with `p >= bank_p_thr` (`ncc >= ncc_thr` only when `p` is absent), plus neural events with score ≥ `neural_conf`. The neural events must sit more than `margin` away from the bank events. The prior still applies: no overlap, gaps ≥ 0.2 s, at most `max_events` in total.
- **prior**: `prior_thr <= synthetic_score < synth_thr`, or a bank clip where no bank event survives the threshold. Output is the neural synthetic decode under the prior, with scores ≥ `prior_conf`.
- **natural**: everything else.

Outputs are never empty.

## Decoder families (params)

- `legacy`: `hi lo smooth median min_dur merge_gap pad_on pad_off split_drop split_min track_thr track_min track_cover track_iou use_cat bref bref_thr`. The union, category and union-boundary channels are read from the contract layout.
- `hysteresis_per_category`: works per track, over the union plus each present category.
  - Steps: median smoothing, hi/lo hysteresis, merging gaps shorter than `gap`, `min_dur`, then biases.
  - Thresholds: `hi lo_delta` (union) and `cat_hi cat_lo_delta`, with optional per-category `cat_hi_delta[7]`. `gate` zeroes the category tracks wherever the union track is below it.
  - Biases: `on_bias off_bias union_on_bias union_off_bias`, with optional `cat_on_bias[7] cat_off_bias[7]`.
  - Merging tracks: `union_cover` drops a union-track event when category events already cover at least that fraction of it. `dedup_iou` (0.8) removes cross-track near-duplicates, keeping the higher score.
- `csebb`: a clean-room change-detection SEBB decoder.
  - Step filter `delta = mean(next tau/2) - mean(prev tau/2)`, with `tau` in {0.06..0.32}. Change points are the local maxima and minima of delta with `|delta| >= min_step`, with parabolic refinement.
  - Segments below `seg_thr` are gaps. An interior dip separates two boxes only if it is ≥ `gamma_abs` below the neighbouring maxima and ≤ `(1 - gamma_rel)` of them.
  - Confidence is the mean posterior, compared with `thr` (union) or `cat_thr` (categories).
  - Shares the track-merging keys and biases of `hysteresis_per_category`.
- `peaks`: uses the onset/offset peaks of the boundary channels: per category if those channels are present, else the union.
  - Peaks pass `on_thr`/`off_thr`, with parabolic sub-frame times `(i + 0.5 + d) * 20 ms + shift`. Clip-edge boundaries are implicit when presence ≥ `edge_pres`.
  - Each onset is paired with one of its next `max_next` offsets. A pair is kept when the presence mean is ≥ `pres_thr`; it is scored `(1 - alpha) * presence + alpha * sqrt(on * off)`.
  - Pairs are taken greedily one-to-one, with `nms_iou`. `fill_*` adds union hysteresis blobs that no pair covers.
