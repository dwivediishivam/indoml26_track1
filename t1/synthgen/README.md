# synthgen: synthetic noise-event mixtures

This package rebuilds the organisers' synthetic clips: `noisy = clean + sum_i g_i * snippet_i`. The labels are sample-exact (`noisy - clean` is 0 outside the spans), so items get `boundary_weight = 1.0`. The defaults in `config_default.yaml` were measured on 695 validation synthetic clips.

## API
```python
from synthgen import build_mixer, SynthMixer, load_config, ids_in_folds
from synthgen.sources import BankSource, GoldEventSource, FolderSource, ValCleanSource, GoldSpeechSource
from synthgen.dataset import SynthIterableDataset, SynthDataset, OfflineSynthDataset

mixer = build_mixer({"widen_prob": 0.2}, exclude_ids=ids_in_folds([fold]))   # sources from cfg["sources"]
# or explicitly:
mixer = SynthMixer([(BankSource(), 1.0), (GoldEventSource(tag_prior=load_config()["tag_freq"]), 0.3)],
                   [(ValCleanSource(exclude_ids=ids_in_folds([fold])), 1.0)], cfg={"crop": {"prob": 0.2}})
r = mixer.sample(np.random.default_rng(0))   # wav, clean float32[N]; events [(on, off, category, tag)]; info; duration
ds = SynthIterableDataset(mixer, epoch_size=20000, seed=0, max_seconds=10.0, crop_seconds=None, weight=1.0)
```
Each dataset item has these fields:
- `wav`: float32 `[S]`
- `length`: int, number of valid samples
- `target`: `[24, T]`, from `common.targets.frame_targets`
- `clip_cats`: `[7]`
- `tier`: `"synthetic"`
- `weight`
- `boundary_weight`: 1.0

`S` is `crop_seconds` or `max_seconds` (zero-padded) and `T = ceil(S/0.02)`. Clips longer than `S` get a random window. Extra keyword arguments (`sigma_frames`, `clip_edge_boundaries`) are passed through to `frame_targets`.

- **Seeding.** `SynthIterableDataset` seeds by (seed, epoch, DDP rank, worker id). `SynthDataset` / `OfflineSynthDataset` seed by (seed, epoch, index). Call `ds.set_epoch(e)`, and use `persistent_workers=False` for the iterable dataset.
- **Offline set.** `python -m synthgen.write_offline_set --out work/synth/v1 --num 20000 --workers 8 [--set key.sub=value] [--exclude-folds 0]` writes resumable, uncompressed `shard_XXXXX.npz` files. `OfflineSynthDataset` memory-maps them.

## Sources
| class | audio | notes |
|---|---|---|
| `BankSource(bank_dir, uniform, exclude_source_clips)` | `work/bank/snippets.{npz,json}` (written by bank/) | p ∝ `uses`; read-only |
| `GoldEventSource(data_dir, tiers, min_dur, max_dur, isolation, cache, tag_prior, crop_prob)` | Gold spans from the staged blob (`~/indoml-work/data`) | non-overlapping spans, 0.12–9.5 s; JSON index cached in `work/synthgen/`; contains that clip's speech |
| `FolderSource(folder, mapping_csv, trim_db, ...)` | external isolated events | CSV columns `category,tag` and either `file` or `dir` (`.` = all), optional `weight`; silence is trimmed |
| `ValCleanSource(exclude_ids, splice_prob)` | `syntheticCleanRefAudio` | `exclude_ids` keeps held-out folds unseen |
| `GoldSpeechSource(data_dir, tiers, margin, min_len, splice_prob)` | Gold/Silver regions ≥ 50 ms from any event, ≥ 1.5 s long | crops to the target length, or crossfade-splices regions |

`tag_prior` reweights items so that their tag mix follows `tag_freq`. Indexes are built lazily. Before you start DataLoader workers, call `len(src)` (noise) or `src.lengths` (speech) once in the main process. Otherwise every worker scans the manifest. Memmaps are reopened in each worker and are never pickled.

## Config knobs (`config_default.yaml`)
**Distribution specs** (`mixer.Dist`) take one of these forms:
- a number
- `[lo, hi]` (uniform)
- `{probs, values}` (quantile knots, linear inverse CDF)
- `{values, weights}` (discrete)

An override that contains a spec replaces the whole spec.

**Clip length and event count**
- `clip_duration_mode`: `config` draws from `clip_duration`. `speech` uses the natural length of the drawn clean clip, capped at `clip_max_s`.
- `n_events`: events per clip. This is drawn once per clip and is never biased by the rejection loop.

**Placement**
- `gap_min_s`: minimum gap between events, 0.2 s.
- `first_onset`, `last_tail`: the edge margins.
- `edge_slack_weight`: how leftover time is shared between the edges and the gaps. With 0, only the gaps get it (for one-event clips, the edges get it).
- `max_tries`, `edge_tries`: limits for the rejection loop. When they run out, the mixer drops the edge margins, crops events to fit, and drops the longest events only if it still cannot fit them.

**Event length and cropping**
- `event_duration`, `event_min_s`, `event_max_s`: event lengths.
- `crop.prob`, `crop.min_snippet_s`: crop a snippet to a length drawn from `event_duration`. A source's `crop_prob` overrides `crop.prob`. Snippets longer than `event_max_s` are always cropped.

**Level and SNR**
- `snr_db`: per-event SNR, defined as clean power over the whole clip divided by the event's power inside its span.
- `snr_per_clip`: share one SNR across all events in a clip.
- `fade_ms`: raised-cosine fade inside each span edge. The default is 0.
- `speech_gain_db`: gain applied to the clean speech.
- `peak_limit`: when the peak exceeds 0.99, the mix and the clean signal are scaled by the same factor.

**Other keys**
- `overlap_prob`: chance that an event ignores the no-overlap rule. This only applies in widen mode.
- `widen_prob` and `widen:` (robustness for natural clips): with probability `widen_prob`, a sample uses the `widen` overrides (0–6 events, SNR −20 to +15 dB, overlaps, fades, crops, speech gain).
- `sources`: used by `build_mixer`. A source with weight 0 is not built.
- The q0/q100 quantile knots marked "extrap." in `config_default.yaml` are guesses, not measurements.

`mixer.mini_yaml` reads the config on hosts without PyYAML. It handles block mappings with JSON-literal values only.
