# t1_final — Track 1 final pipeline (17 Oct 2026, deadline 12:00 IST)

None of this code has been executed yet. The only checks so far are `py_compile` and `bash -n`. Run the smoke steps (P3, A0) before the long jobs.

Run everything from the `t1_final/` root with `PYTHONPATH=.` (the sbatch files set it themselves).

```
common/    contract (paths, 24-ch 20 ms cache layout, categories), exact scorer, folds, frame targets
bank/      synthetic noise-snippet bank: build, matched-filter detection, leave-fold-out calibration
synthgen/  generator replicating the organisers' synthetic mixtures (exact labels)
models/    backbones (ATST-F strong, BEATs strong, WavLM, w2v-BERT, mel-CNN, fusion), trainer, inference, launchers
decode/    decoders (legacy, per-category hysteresis, cSEBB, peak pairing), fusion, bank merge, CV tuner
assemble/  blind decode -> predictions.jsonl -> validated ZIP
data/      val/validation (2,501 clips with metadata + clean refs), blind/audio (5,517 clips)
work/      folds.json (done), bank/, caches/<model>/{val,blind}.npz, tuned/, submissions/
```

## Why the pipeline looks like this (measured on validation/blind)

- **Synthetic clips dominate the event score.** Synthetic is 2,080/5,517 test clips but about 54% of test events, and event F1 is pooled over events. In validation, every synthetic event comes from a reused bank of about 206 snippets, inserted sample-exactly, and the blind synthetic clips use the same bank. This is the job of `bank/` and `synthgen/`.
- **Boundary precision is not the bottleneck; detecting each event as its own unit is.** Boundary jitter of σ=20 ms still gives F1 0.995, and a 40 ms frame grid costs no F1.
- **Natural clips need per-category tracks.** Events overlap in 30% of consecutive natural pairs; a single binary track caps natural F1 at 0.90.
- **Breaths matter most on natural clips.** Breathing/inhaling are 40% of natural events, which is why the m4 breath specialist exists.
- **Never emit an empty clip.** Every clip has at least one event, so an empty prediction scores Dice 0.

## Lanes (run in parallel) and timeline

| IST | Lane P: prep (login node / Mac) | Lane A: CPU (Mac, 10 cores, or a CPU node) | Lane G: GPUs (1 GPU per candidate) |
|---|---|---|---|
| 05:45 | P1 bank build, P2 sync to cluster, P2b fetch weights | A1 bank calibrate | — |
| 06:15 | P3 smoke job (≤ 50 min, all candidates) | A2 bank detect on blind | — |
| 06:45 | — | A3 tune legacy + bank → **Submission 1** (~07:30) | G1 launch all candidates (DEADLINE 10:15) |
| 10:15 | P4 pull caches back | — | caches written (training stops 40 min earlier) |
| 10:20 | — | A4 tune with all caches (≤ 45 min) | — |
| 11:10 | — | A5 build + check the final ZIP → **final submission by 11:20** | — |

There are 5 submissions per day. Use 1 for the early check, 1 for the final, and keep the rest in reserve. Codabench slows down near deadlines, so do not leave the final upload later than 11:30.

### P — prep
```bash
# P1 (Mac, ~1 min): build the bank. synthgen needs it on the cluster before training.
cd t1_final && PYTHONPATH=. python -m bank.build_bank
# P2: copy code + data + bank to the cluster (data is ~1.7 GB)
rsync -a --exclude '__pycache__' t1_final/ user@cluster:~/t1_final/
# P2b (login node, internet): weights -> ~/indoml-offline/model/
ssh user@cluster 'cd ~/t1_final && bash models/backbones/fetch_weights.sh'
# P3: smoke all candidates for 30 steps each; fix any FAIL before G1
ssh user@cluster 'cd ~/t1_final && sbatch models/launch/smoke.sbatch'
```

### A — CPU lane (bank + decoding)
```bash
# A0: decoder smoke test (2 minutes)
PYTHONPATH=. python -m decode.tune --name smoke --legacy --no-models --limit 200 --minutes 2 --workers 4
# A1: leave-fold-out calibration. Gate: natural_clips_flagged ~ 0, synthetic F1 high.
PYTHONPATH=. python -m bank.calibrate --workers 10
# A2: blind detection. Gate: ~0.30–0.40 of clips flagged, ~3.2 events per flagged clip.
PYTHONPATH=. python -m bank.detect_blind --workers 10
# A3: tune legacy caches + bank, then Submission 1
PYTHONPATH=. python -m decode.tune --name legacy_bank --legacy --no-models --minutes 30
PYTHONPATH=. python -m assemble.make_submission --config work/tuned/legacy_bank.json --out work/submissions/legacy_bank.zip
python assemble/check_zip.py work/submissions/legacy_bank.zip
```

Submission 1 directly tests the bank on the real test set. The previous leaderboard entry had Event F1 0.47; a large F1 jump confirms the bank transfers.

### G — GPU lane (candidates are independent; launch as many as there are GPUs)
```bash
cd ~/t1_final
export DEADLINE=2026-10-17T10:15:00+05:30          # caches must exist by then
bash models/launch/run_all_omni.sh                    # prints the commands
bash models/launch/run_all_omni.sh --submit           # submits one 1-GPU job per candidate
# 8-GPU box instead: DEADLINE=... bash models/launch/torchrun_node.sh per-gpu m1v_atst_val m2_beats m3_wavlm_large m4_breath m5_w2vbert m1_atst
```

| Candidate | Backbone | Training data | Role |
|---|---|---|---|
| m1v_atst_val | ATST-F strong + mel-CNN | gold/silver/bronze/synth + val natural (fold 4 held out) | primary |
| m2_beats | BEATs strong + mel-CNN | + Gold-event and widened synthetic mixtures | diversity |
| m3_wavlm_large | WavLM-Large (layer mix) + mel-CNN | same as m2 | speech-aware |
| m4_breath | WavLM-base-plus + mel-CNN | human_non_speech ×3, 4 s crops | breath specialist |
| m5_w2vbert | w2v-BERT 2.0 + mel-CNN | same as m2 | multilingual Indic |
| m1_atst | ATST-F strong + mel-CNN | no validation audio | clean reference for CV |

With few GPUs, launch in this order: m1v, m2, m3, m4, m5, m1.

Monitor progress with `~/indoml-work/runs_t1f/<c>/status.json`. Each job writes `work/caches/<c>/{val,blind}.npz` at the end.

### P4 + A4/A5 — fuse everything and submit
```bash
rsync -a user@cluster:~/t1_final/work/caches/ t1_final/work/caches/
# Models trained on val_natural (m1v, m2, m3, m5) saw folds 0-3, so fusion/decoder choice is tuned on fold 4 only:
PYTHONPATH=. python -m decode.tune --name final --legacy --models --minutes 45 --init work/tuned/legacy_bank.json \
    --weight-grid 0.1 --only-folds 4 --cv none
# Honest comparison: score legacy_bank on fold 4 too (re-tune quickly with the same restriction)
PYTHONPATH=. python -m decode.tune --name legacy_bank_f4 --legacy --no-models --minutes 10 \
    --init work/tuned/legacy_bank.json --only-folds 4 --cv none
PYTHONPATH=. python -m assemble.make_submission --config work/tuned/final.json --out work/submissions/final.zip
python assemble/check_zip.py work/submissions/final.zip
```

**Promotion rule:** submit `final.zip` only if its fold-4 score beats `legacy_bank_f4`. Both are restricted to fold 4, about 526 clips, so expect ±0.02 noise; prefer a clear margin. Otherwise the earlier submission stands.

## Failure fallbacks

| What fails | What to do |
|---|---|
| The bank doesn't transfer (A2 flags far from 0.38, or Submission 1 F1 doesn't rise) | Tune with `--no-bank`; the neural models still train on bank-style mixtures |
| A candidate fails the smoke test or diverges | Drop it; the tuner gives zero weight to missing or bad caches |
| No new caches by 10:30 | Keep Submission 1 |
