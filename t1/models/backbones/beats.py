"""BEATs (iter3+ AS2M), AudioSet-Strong fine-tuned by PretrainedSED (BEATs_strong_1.pt).

How PretrainedSED turns BEATs tokens into a time sequence (models/beats/BEATs_wrapper.py +
models/prediction_wrapper.py @1aa47e4): kaldi fbank (128 bins, 25/10 ms, (x - 15.41663) / (2 * 6.55582))
-> Conv2d 16x16 patches over (time, mel) -> 8 frequency patches per 16-frame (160 ms) time step,
flattened time-major (index = t * 8 + f). A 10 s clip has 998 fbank frames -> 62 time steps -> 496 tokens
("BEATs: batch x 496 x 768"), which the wrapper adaptive-avg-pools along the flattened axis to 250
"40 ms" frames -- i.e. each label frame sees ~2 frequency patches of one 160 ms step. So BEATs' true
time resolution is 160 ms: seq_len = floor((1 + (N - 400) // 160) / 16) * 8 tokens.

Here: tokens are reshaped to [B, steps, 8, 768] and pooled over frequency (freq_pool="mean" -> 768-d,
"concat" -> 6144-d). Time step t covers fbank frames 16t..16t+15 -> centre 0.16 t + 0.0875 s; mapped to
20 ms frames by centre-aligned linear interpolation. time_shifts=S > 1 additionally runs S copies shifted by
160/S ms and interleaves them (native hop 160/S ms, S x compute). Inputs > chunk_sec are windowed like ATST
(the model was fine-tuned on 10 s clips; relative position buckets saturate at 800 tokens).
Padding: fbank is frame-local with fixed normalisation; the encoder gets a token padding mask (zeroed before
the conv positional embedding, key-padding in attention).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import (Backbone, LayerMix, chunked_encode, group_params_by_layer, n_out_frames, prepare_lengths,
                   resample_to_frames, resolve_file)
from .features import fbank, fbank_valid_frames
from .vendor.beats.BEATs import BEATs, BEATsConfig

BEATS_FILES = ["BEATs_strong_1.pt"]
STEP = 2560             # samples per 160 ms time step
FIRST_CENTER = 0.0875   # s, centre of step 0 (fbank frames 0..15, frame j centre 0.01 j + 0.0125)
N_FREQ = 8
FBANK_MEAN, FBANK_STD = 15.41663, 6.55582


def load_beats_strong(beats: BEATs, checkpoint) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict) and "cfg" in state:   # raw microsoft format
        state = {"model.model." + k: v for k, v in state["model"].items()}
    sd, bad = {}, []
    for k, v in state.items():
        if k.startswith(("strong_head.", "weak_head.", "seq_model.")):
            continue
        if k.startswith(("model.model.", "model.beats.")):
            k = k[len("model.model."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        else:
            bad.append(k)
            continue
        if k.startswith("predictor."):
            continue
        k = k.replace("pos_conv.0.weight_g", "pos_conv.0.parametrizations.weight.original0")
        k = k.replace("pos_conv.0.weight_v", "pos_conv.0.parametrizations.weight.original1")
        sd[k] = v
    if bad:
        raise ValueError(f"Unexpected BEATs checkpoint keys: {bad[:5]}")
    missing, extra = beats.load_state_dict(sd, strict=False)
    if missing or extra:
        raise ValueError(f"BEATs load mismatch: missing={missing[:8]}, extra={extra[:8]}")


class BEATsStrong(Backbone):
    """kwargs: checkpoint=None (auto-search), layer_mix=False, per_layer_norm=True, freq_pool="mean"|"concat",
    time_shifts=1, chunk_sec=10.0, min_overlap_sec=2.0, dropout=None (cfg default 0.1), layerdrop=0.0,
    fbank_impl="torch"|"torchaudio", pretrained=True."""

    def __init__(self, checkpoint=None, layer_mix: bool = False, per_layer_norm: bool = True,
                 freq_pool: str = "mean", time_shifts: int = 1, chunk_sec: float | None = 10.0,
                 min_overlap_sec: float = 2.0, dropout: float | None = None, layerdrop: float = 0.0,
                 fbank_impl: str = "torch", pretrained: bool = True):
        super().__init__()
        cfg = BEATsConfig()
        if dropout is not None:
            cfg.dropout = cfg.attention_dropout = cfg.dropout_input = float(dropout)
        cfg.encoder_layerdrop = float(layerdrop)
        self.beats = BEATs(cfg)
        if pretrained:
            path = resolve_file(checkpoint, BEATS_FILES, env_var="BEATS_STRONG_CKPT")
            load_beats_strong(self.beats, path)
            self.checkpoint = str(path)
        if freq_pool not in ("mean", "concat"):
            raise ValueError("freq_pool must be 'mean' or 'concat'")
        if STEP % int(time_shifts) != 0:
            raise ValueError("time_shifts must divide 2560 (1, 2, 4, 5, 8, 10, 16, ...)")
        self.freq_pool, self.time_shifts, self.fbank_impl = freq_pool, int(time_shifts), fbank_impl
        self.win = None if chunk_sec is None else int(round(chunk_sec * 16000 / STEP)) * STEP
        self.min_overlap = int(round(min_overlap_sec * 16000 / STEP)) * STEP
        self.n_layers = cfg.encoder_layers
        D = cfg.encoder_embed_dim
        self.out_dim = D * (N_FREQ if freq_pool == "concat" else 1)
        self.mix = LayerMix(self.n_layers + 1, self.out_dim, per_layer_norm) if layer_mix else None

    def encoder_modules(self):
        return [self.beats]

    def _pool(self, x: Tensor, steps: int) -> Tensor:          # [B, steps*8, D] -> [B, steps, D']
        x = x.reshape(x.shape[0], steps, N_FREQ, x.shape[-1])
        return x.mean(2) if self.freq_pool == "mean" else x.flatten(2)

    def _encode_once(self, wav: Tensor, lengths: Tensor):
        b = self.beats
        fb = fbank(wav * 2 ** 15, 128, self.fbank_impl)
        fb = (fb - FBANK_MEAN) / (2 * FBANK_STD)                                  # [B, F, 128]
        x = b.patch_embedding(fb.unsqueeze(1))                                    # [B, 512, steps, 8]
        steps = x.shape[2]
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)                # [B, steps*8, 512]
        x = b.layer_norm(x)
        if b.post_extract_proj is not None:
            x = b.post_extract_proj(x)
        x = b.dropout_input(x)
        valid = torch.div(fbank_valid_frames(lengths) + 15, 16, rounding_mode="floor").clamp(1, steps)
        pad = torch.arange(steps, device=x.device)[None, :] >= valid[:, None]
        pad = pad[:, :, None].expand(-1, -1, N_FREQ).reshape(x.shape[0], -1)     # time-major like the tokens
        tgt = self.n_layers - 1 if self.mix is not None else None
        x, layer_results = b.encoder.extract_features(x, padding_mask=pad, tgt_layer=tgt)
        if self.mix is not None:
            x = self.mix([self._pool(h.transpose(0, 1), steps) for h, _ in layer_results])
        else:
            x = self._pool(x, steps)
        return x, valid

    def _encode(self, wav: Tensor, lengths: Tensor):
        S = self.time_shifts
        if S == 1:
            return self._encode_once(wav, lengths)
        B, N = wav.shape
        shifts = [s * (STEP // S) for s in range(S)]
        wav_s = torch.cat([F.pad(wav[:, sh:], (0, sh)) for sh in shifts], 0)
        len_s = torch.cat([(lengths - sh).clamp(min=1) for sh in shifts], 0)
        x, valid = self._encode_once(wav_s, len_s)                               # [S*B, steps, D]
        x = x.reshape(S, B, x.shape[1], x.shape[2]).permute(1, 2, 0, 3)            # [B, steps, S, D]
        valid = valid.reshape(S, B).min(0).values * S
        return x.reshape(B, -1, x.shape[-1]), valid

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        lengths = prepare_lengths(wav, lengths)
        T = n_out_frames(wav.shape[-1])
        min_n = 400 + 15 * 160 + STEP                                             # >= one 16-frame step (+ shifts)
        wav = wav.float()
        if wav.shape[-1] < min_n:
            wav = F.pad(wav, (0, min_n - wav.shape[-1]))
        grid = STEP // self.time_shifts
        if self.win is not None:
            feats, valid = chunked_encode(self._encode, wav, lengths, self.win, self.min_overlap, grid)
        else:
            feats, valid = self._encode(wav, lengths)
        return resample_to_frames(feats, T, grid / 16000, FIRST_CENTER, valid)

    def layer_groups(self):
        """12 groups shallow->deep: [frontend/pos_conv/layer_norm + layer 0], layer 1, ..., layer 11.
        The shared relative_attention_bias lives in layer 0 (upstream separate_params convention)."""
        def layer_of(name):
            if ".layers." in name:
                return int(name.split(".layers.")[1].split(".")[0])
            return 0
        return group_params_by_layer(self.beats.named_parameters(), self.n_layers, layer_of, self._unused_ids())
