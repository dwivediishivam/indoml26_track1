# models/backbones

`from models.backbones import build_backbone` -> `Backbone` with

```
forward(wav [B, N] float32 16 kHz, lengths [B] valid samples) -> [B, T = ceil(N/320), out_dim]
encoder_parameters()  pretrained params      new_parameters()  random-init params (layer mix, norms, mel_cnn)
layer_groups()        encoder params, shallow -> deep (for layer-wise LR decay)
freeze_encoder(bool)  optional helper: requires_grad off + encoder kept in eval() while frozen
```

Output frame i covers [0.02 i, 0.02 (i+1)) s (centre 0.02 i + 0.01). Frames past a clip's valid length
are edge-replicated garbage. Native frames are mapped by `base.resample_to_frames`: linear interpolation
at the output centres using each encoder's true native frame centre (align_corners=False semantics,
edge-clamped to the clip's last valid native frame). Native 20 ms models are padded or trimmed
(`pad_trim_frames`). Padding never enters normalisation or attention (details per backbone below).

Weights are looked up in `$INDOML_MODEL_DIR` (default `~/indoml-offline/model`). `.pt` files are also
searched in `<dir>/pretrainedsed/`, `t1_final/work/weights/` and `../Phase 2/assets/`. Get them with
`bash models/backbones/fetch_weights.sh` on a machine with internet. HF loaders use local dirs first,
otherwise the HF id with `local_files_only` set from `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`.

| name | checkpoint / source | license | expected local path | native frames -> 20 ms | out_dim |
|---|---|---|---|---|---|
| `atst_strong` | `ATST-F_strong_1.pt`, https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/ATST-F_strong_1.pt (sha256 fbf25779...) | PretrainedSED code: MIT (vendor/atst/LICENSE); ATST-Frame upstream: github.com/Audio-WestlakeU/audiossl | `$DIR/ATST-F_strong_1.pt` | 40 ms tokens, centre 0.04k+0.015 s -> linear interp | 768 |
| `beats_strong` | `BEATs_strong_1.pt`, https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/BEATs_strong_1.pt | MIT (PretrainedSED + Microsoft BEATs; vendor/beats/LICENSE) | `$DIR/BEATs_strong_1.pt` | 160 ms steps (8 freq tokens pooled), centre 0.16t+0.0875 s -> linear interp | 768 (6144 concat) |
| `wavlm` | https://huggingface.co/microsoft/wavlm-base-plus | check the HF model card (code: microsoft/unilm, MIT) | `$DIR/wavlm-base-plus/` | 20 ms, centre 20k+12.5 ms -> pad/trim | 768 |
| `wavlm_large` | https://huggingface.co/microsoft/wavlm-large | as above | `$DIR/wavlm-large/` | 20 ms -> pad/trim | 1024 |
| `w2vbert` | https://huggingface.co/facebook/w2v-bert-2.0 | MIT (HF card) | `$DIR/w2v-bert-2.0/` | 20 ms (fbank stride 2), centre 20k+17.5 ms -> pad/trim | 1024 |
| `mel_cnn` | none (random init) | - | - | 10 ms log-mel -> avg-pool x2 (centre 20k+5 ms) -> pad/trim | 256 |
| `fusion` | parts | - | - | parts already at 20 ms | proj_dim |

## Per-backbone notes and kwargs

**atst_strong** (`atst.py`). `checkpoint=None` (auto search; env `ATST_STRONG_CKPT`), `layer_mix=False`
(True: softmax mix of the embedding output + 12 blocks, each parameter-free-LayerNormed, then LayerNorm),
`per_layer_norm=True`, `dropout=0.0`, `topdb="sample"` (`"batch"` = upstream ATSTMel exactly, where
torchaudio's top_db floor is a max over the whole batch), `min_overlap_sec=2.0`, `grad_ckpt=False`,
`pretrained=True`. Frontend = Phase-2/PretrainedSED ATSTMel (64 mels, hop 10 ms, win 1024, dB clamp, MinMax).
Attention gets a key-padding mask (the upstream Block ignores lengths). Absolute positions cap the window at
10 s (250 tokens); longer clips (the val set reaches 18.4 s) are windowed: starts on the 40 ms grid,
>= 2 s overlap, trapezoid overlap-add. ~86M params, 344 MB fp32; 250x250 attention per 10 s window, cheap.
Full fine-tuning at batch 32 x 10 s with AMP needs about 15-20 GB; `grad_ckpt=True` cuts that a lot.

**beats_strong** (`beats.py`). `checkpoint=None` (env `BEATS_STRONG_CKPT`), `layer_mix=False`,
`per_layer_norm=True`, `freq_pool="mean"|"concat"`, `time_shifts=1` (S>1: S shifted passes interleaved ->
160/S ms native hop, S x compute), `chunk_sec=10.0` (None = whole clip in one pass), `min_overlap_sec=2.0`,
`dropout=None` (BEATs cfg default 0.1), `layerdrop=0.0` (cfg default 0.05), `fbank_impl="torch"|"torchaudio"`,
`pretrained=True`. PretrainedSED flattens 62 time x 8 freq patch tokens time-major (496 tokens per 10 s) and
adaptive-avg-pools that axis to 250 frames, so the real resolution is 160 ms. Here the 8 freq tokens of each
step are pooled. Token padding mask is passed to the encoder. ~90M params; 496 tokens per 10 s window (~4x
ATST attention cost). Upstream separate_params puts frontend + pos_conv into the first group.

**wavlm / wavlm_large** (`hf_ssl.py`). `model="microsoft/wavlm-base-plus"` (`wavlm_large` presets
`microsoft/wavlm-large`), `path=None` (explicit local dir), `normalize=None` (auto from
`preprocessor_config.json`: base-plus False, large True; masked over valid samples), `per_layer_norm=True`,
`dropout=None`, `layerdrop=0.0`, `spec_augment=False`, `grad_ckpt=False`, `num_layers=None` (keep only the
first k layers). Output = softmax mix over all hidden states (13 / 25) -> LayerNorm. Base-plus uses a
GroupNorm in conv layer 0 (instance norm over time); it is recomputed over valid frames only, which equals
running each clip unpadded. `attention_mask` from valid lengths. Base-plus 95M params; large 316M (1.2 GB fp32).
Fine-tuning large at 10 s x 16 needs `grad_ckpt=True` or a frozen encoder.

**w2vbert** (`hf_ssl.py`). `model="facebook/w2v-bert-2.0"`, `path=None`, `per_layer_norm=True`, `dropout=None`,
`layerdrop=0.0` (config default 0.1), `spec_augment=False`, `grad_ckpt=False`, `num_layers=None`,
`fbank_impl="torch"`. SeamlessM4TFeatureExtractor is re-implemented in torch on the GPU: kaldi fbank 80 bins
on x*2^15, per-clip per-bin CMVN (ddof=1) over valid frames, stride-2 stacking to 160-d, attention mask =
F_valid // 2. 25 hidden states, 1024-d, about 580M params (2.3 GB fp32). It is the heaviest backbone:
use bf16, `grad_ckpt=True`, `num_layers` truncation, or keep it frozen.

**mel_cnn** (`mel_cnn.py`). `n_mels=128`, `channels=(16, 32, 64, 128, 256)`, `out_dim=256`, `norm="bin"|"global"`,
`f_min=20`, `f_max=8000`, `dropout=0.0`. 128-bin log-mel (25 ms window, 10 ms hop), instance normalisation over
valid frames, 5 double-conv blocks with channel LayerNorm and 2x frequency average pooling, then 2x time
pooling to 20 ms, Linear + LayerNorm. About 1.4M params, all in `new_parameters()`. The 10 ms first block is
the main activation cost.

**fusion** (`fusion.py`). `parts=[{"name": ..., **kwargs}, ...]`, `proj_dim=512`, `norm="per_part"|"joint"`,
`part_dropout=0.0`. Each part's output is LayerNormed (per part by default), concatenated, then a Linear
projection. Encoder and new parameter lists are unions of the parts' lists, plus the norms and projection
in new. Layer groups are aligned at the deep end.

```python
bb = build_backbone("fusion", parts=[{"name": "atst_strong", "layer_mix": True}, {"name": "mel_cnn"}], proj_dim=512)
```

## Vendored code
- `vendor/atst/`: PretrainedSED @1aa47e4 `models/atstframe/*` + `models/transformer_wrapper.py`
  (copied from Phase 2, see PROVENANCE.json), MIT LICENSE.
- `vendor/beats/`: PretrainedSED @1aa47e4 `models/beats/{BEATs,backbone,modules}.py`, MIT LICENSE. Imports
  are package-relative and the torchaudio.compliance import is guarded (PROVENANCE.json).

## Untested
Nothing here has been executed (only `py_compile`). First smoke test on the cluster: for each backbone, build it,
then run `forward(torch.randn(2, 16000*12), torch.tensor([16000*12, 16000*3]))`. Check the shape is
`[2, 600, out_dim]`. Then compare the valid frames of clip 2 with a solo run of `wav[1:, :48000]`. They should match up to float error for wavlm, w2vbert and mel_cnn. For atst and beats they match only approximately, because a batch longer than 10 s is windowed and the windows are blended.
For w2vbert/beats, also compare `fbank_impl="torch"` against `"torchaudio"`.
