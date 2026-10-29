# bank/ — detecting synthetic noise snippets with a matched filter

The organizers' synthetic clips are built as `clean speech + sample-exact copies of noise snippets`. There are about 206 distinct snippets in validation and an estimated ~290 overall. They are placed without overlap, with gaps of at least 200 ms. Validation provides both the noisy and the clean audio, so each labelled span isolates its snippet exactly. A matched-filter test showed the blind synthetic clips draw on the same bank.

| Step | Command | Output | CPU cost (16 workers) |
|---|---|---|---|
| 1 | `PYTHONPATH=. python -m bank.build_bank` | `work/bank/snippets.{npz,json}`, `val_event_map.json` | ~1 min |
| 2 | `PYTHONPATH=. python -m bank.calibrate --workers 16` | `calibration.json`, `val_events.jsonl` (leave-fold-out, honest), `val_candidates.npz` | ~10–20 min |
| 3 | `PYTHONPATH=. python -m bank.detect_blind --workers 16` | `blind_events.jsonl` | ~20–40 min |

Run all commands from `t1_final/`. Steps 2 and 3 are embarrassingly parallel across clips. Step 3 needs only step 1 and `calibration.json`.

## How it works

**Candidate generation.** For every snippet, compute the NCC of that snippet against every full-overlap position in the clip, using FFT correlation. Keep up to 3 peaks per snippet with NCC ≥ 0.35. Edge sub-templates (the first or last 250 ms of a snippet) catch events that the clip boundary cuts off.

**Features computed for each candidate:**
- `ncc`: the NCC value.
- `cons`: chunk-lag consistency, i.e. whether each 50 ms chunk peaks at exactly the same lag, to within ±1 sample. A true sample-exact copy does this; natural look-alikes don't.
- `snr`: the implied SNR.
- `log_dur`: log of the snippet duration.
- `edge`: whether the candidate came from an edge sub-template.

**Scoring.** A logistic model maps the features to a match probability `p`. It is fitted leave-fold-out for the validation output and on all folds for blind.

**Non-maximum suppression.** Candidates are processed greedily in order of `p`; any overlap with an already-kept event is dropped. Thresholding after NMS is equivalent to thresholding before it, so `decode/tune.py` can choose the final `p` threshold.

## Record schema

The schema is the same for `val_events.jsonl` and `blind_events.jsonl`. Each line:

```json
{"clip_id": "...", "synthetic_score": 0.98,
 "events": [{"onset": 0.436, "offset": 0.961, "tag": "<horn>", "category": "vehicle_traffic",
             "snippet": "s0012", "kind": "full|head|tail", "ncc": 0.81, "cons": 0.93, "p": 0.99}]}
```

## Sanity targets to check after running

- In the step 2 printout, `natural_clips_flagged` at the chosen threshold should be about 0. `synthetic_f1_with_natural_fp` should be high, around 0.95 or more if the bank generalises across folds.
- Step 3 should flag roughly 0.377 of the blind clips (2,080 of 5,517), with about 3.2 events per flagged clip. A large deviation means the bank or the calibration doesn't transfer, and `p` should not be trusted blindly.
