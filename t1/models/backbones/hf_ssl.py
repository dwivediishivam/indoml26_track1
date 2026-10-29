"""HF transformers SSL speech encoders: WavLM (base-plus / large) and w2v-BERT 2.0.

Both are native 20 ms models; outputs are a learned softmax mix of all hidden states (embedding output +
every layer) followed by LayerNorm, then pad/trim to exactly T = ceil(N / 320) frames.

WavLM: conv feature encoder, receptive field 400 samples / stride 320 -> frame k centre 20 k + 12.5 ms,
L = floor((N - 400) / 320) + 1. Padding never leaks: optional waveform normalisation over valid samples only;
for feat_extract_norm == "group" (base / base-plus) the first conv layer's GroupNorm(512, 512), which is
an instance norm over time in upstream code, is computed over each clip's valid conv frames only (identical
to running the clip unpadded); attention_mask (valid feature frames) goes to the transformer.

w2v-BERT 2.0: SeamlessM4TFeatureExtractor re-implemented in torch on the GPU: kaldi fbank (80 bins, x 2**15)
-> per-clip per-bin mean / std (ddof=1) over valid frames, padding = 0 -> stack 2 frames (stride 2) -> 160-d
at 20 ms. Native frame k = fbank frames 2k, 2k+1 -> centre 20 k + 17.5 ms; valid = F_valid // 2 (the HF
attention mask keeps odd indices).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import (Backbone, LayerMix, group_params_by_layer, length_mask, masked_zero_mean_unit_var,
                   n_out_frames, pad_trim_frames, prepare_lengths, resolve_hf)
from .features import fbank, fbank_valid_frames


def _config_overrides(dropout, layerdrop, spec_augment):
    kw = {"layerdrop": float(layerdrop), "apply_spec_augment": bool(spec_augment)}
    if dropout is not None:
        kw.update(hidden_dropout=dropout, attention_dropout=dropout, activation_dropout=dropout,
                  feat_proj_dropout=dropout)
    return kw


class _HFBase(Backbone):
    def __init__(self, model_cls, model: str, path, dropout, layerdrop, spec_augment, grad_ckpt, num_layers,
                 per_layer_norm):
        super().__init__()
        src, local_only = resolve_hf(model, path)
        self.source = src
        self.model = model_cls.from_pretrained(src, local_files_only=local_only,
                                               **_config_overrides(dropout, layerdrop, spec_augment))
        self.spec_augment = bool(spec_augment)
        if num_layers is not None:                       # keep only the first num_layers transformer layers
            self.model.encoder.layers = self.model.encoder.layers[:int(num_layers)]
            self.model.config.num_hidden_layers = int(num_layers)
        if getattr(self.model, "masked_spec_embed", None) is not None and not self.spec_augment:
            self._mark_unused(self.model.masked_spec_embed)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.n_layers = len(self.model.encoder.layers)
        self.out_dim = self.model.config.hidden_size
        self.mix = LayerMix(self.n_layers + 1, self.out_dim, per_layer_norm)

    def encoder_modules(self):
        return [self.model]

    def layer_groups(self):
        """n_layers + 1 groups: [frontend, feature projection, pos-conv, ...] then one per transformer layer;
        a final (post-stack) encoder LayerNorm joins the last group."""
        n = self.n_layers
        stable = bool(getattr(self.model.config, "do_stable_layer_norm", False))

        def layer_of(name):
            m = re.search(r"encoder\.layers\.(\d+)\.", name)
            if m:
                return int(m.group(1)) + 1
            if stable and name.startswith("encoder.layer_norm."):
                return n
            return 0
        return group_params_by_layer(self.model.named_parameters(), n + 1, layer_of, self._unused_ids())


class WavLMBackbone(_HFBase):
    """kwargs: model="microsoft/wavlm-base-plus" (or "microsoft/wavlm-large" / any HF id), path=None (local dir),
    normalize=None (auto: preprocessor_config do_normalize; large=True, base-plus=False), per_layer_norm=True,
    dropout=None (config default), layerdrop=0.0, spec_augment=False, grad_ckpt=False, num_layers=None."""

    def __init__(self, model: str = "microsoft/wavlm-base-plus", path=None, normalize=None, per_layer_norm=True,
                 dropout=None, layerdrop=0.0, spec_augment=False, grad_ckpt=False, num_layers=None):
        from transformers import WavLMModel
        super().__init__(WavLMModel, model, path, dropout, layerdrop, spec_augment, grad_ckpt, num_layers,
                         per_layer_norm)
        if normalize is None:
            pp = Path(self.source) / "preprocessor_config.json"
            normalize = json.loads(pp.read_text()).get("do_normalize", False) if pp.is_file() \
                else ("large" in model.lower())
        self.normalize = bool(normalize)
        cfg = self.model.config
        self.group_norm_first = cfg.feat_extract_norm == "group"
        self.k0, self.s0 = cfg.conv_kernel[0], cfg.conv_stride[0]

    def _features(self, wav: Tensor, lengths: Tensor) -> Tensor:
        fe = self.model.feature_extractor
        x = wav[:, None]
        if self.training and getattr(fe, "_requires_grad", False) and not x.requires_grad:
            x = x.detach().requires_grad_(True)
        for i, layer in enumerate(fe.conv_layers):
            if i == 0 and self.group_norm_first:
                x = layer.conv(x)                                                    # [B, C, L0]
                gn = layer.layer_norm
                with torch.autocast(device_type=x.device.type, enabled=False):
                    xf = x.float()
                    L0 = xf.shape[-1]
                    v0 = (torch.div(lengths - self.k0, self.s0, rounding_mode="floor") + 1).clamp(1, L0)
                    m = length_mask(v0, L0)[:, None, :].float()
                    mean = (xf * m).sum(-1, keepdim=True) / v0[:, None, None]
                    var = (((xf - mean) * m) ** 2).sum(-1, keepdim=True) / v0[:, None, None]
                    xf = (xf - mean) * torch.rsqrt(var + gn.eps)
                    xf = xf * gn.weight[None, :, None].float() + gn.bias[None, :, None].float()
                x = layer.activation(xf.to(x.dtype))
            else:
                x = layer(x)
        return x.transpose(1, 2)                                                     # [B, L, C]

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        lengths = prepare_lengths(wav, lengths)
        T = n_out_frames(wav.shape[-1])
        wav = wav.float()
        if wav.shape[-1] < 400:
            wav = F.pad(wav, (0, 400 - wav.shape[-1]))
        if self.normalize:
            wav = masked_zero_mean_unit_var(wav, lengths)
        feats = self._features(wav, lengths)
        L = feats.shape[1]
        valid = self.model._get_feat_extract_output_lengths(lengths).long().clamp(1, L)
        attn = length_mask(valid, L)                                                 # bool, True = valid
        hidden, _ = self.model.feature_projection(feats)
        if self.spec_augment:
            hidden = self.model._mask_hidden_states(hidden, attention_mask=attn)
        out = self.model.encoder(hidden, attention_mask=attn, output_hidden_states=True, return_dict=True)
        x = self.mix(list(out.hidden_states))
        return pad_trim_frames(x, T, valid)


class W2VBertBackbone(_HFBase):
    """kwargs: model="facebook/w2v-bert-2.0", path=None, per_layer_norm=True, dropout=None, layerdrop=0.0,
    spec_augment=False, grad_ckpt=False, num_layers=None, fbank_impl="torch"|"torchaudio"."""

    def __init__(self, model: str = "facebook/w2v-bert-2.0", path=None, per_layer_norm=True, dropout=None,
                 layerdrop=0.0, spec_augment=False, grad_ckpt=False, num_layers=None, fbank_impl="torch"):
        from transformers import Wav2Vec2BertModel
        super().__init__(Wav2Vec2BertModel, model, path, dropout, layerdrop, spec_augment, grad_ckpt, num_layers,
                         per_layer_norm)
        self.fbank_impl = fbank_impl
        # adapter / intermediate FFN only touch last_hidden_state; hidden_states come from the encoder.
        for name in ("adapter", "intermediate_ffn"):
            mod = getattr(self.model, name, None)
            if mod is not None:
                self._mark_unused(*mod.parameters())

    def input_features(self, wav: Tensor, lengths: Tensor):
        """SeamlessM4TFeatureExtractor(do_normalize_per_mel_bins=True, stride=2) equivalent, batched."""
        fb = fbank(wav * 2 ** 15, 80, self.fbank_impl)                               # [B, F, 80] float32
        Fn = fb.shape[1]
        fv = fbank_valid_frames(lengths).clamp(1, Fn)
        m = length_mask(fv, Fn)[..., None].float()
        n = fv.float()[:, None, None]
        mean = (fb * m).sum(1, keepdim=True) / n
        var = (((fb - mean) * m) ** 2).sum(1, keepdim=True) / (n - 1).clamp(min=1)
        fb = (fb - mean) / torch.sqrt(var + 1e-7) * m
        if Fn % 2:
            fb = F.pad(fb, (0, 0, 0, 1))
        x = fb.reshape(fb.shape[0], fb.shape[1] // 2, 160)
        valid = (fv // 2).clamp(1, x.shape[1])
        return x, valid

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        lengths = prepare_lengths(wav, lengths)
        T = n_out_frames(wav.shape[-1])
        wav = wav.float()
        if wav.shape[-1] < 720:                                    # >= 2 fbank frames
            wav = F.pad(wav, (0, 720 - wav.shape[-1]))
        x, valid = self.input_features(wav, lengths)
        attn = length_mask(valid, x.shape[1]).long()
        dtype = next(self.model.feature_projection.parameters()).dtype
        out = self.model(input_features=x.to(dtype), attention_mask=attn, output_hidden_states=True,
                         return_dict=True)
        h = self.mix(list(out.hidden_states))
        return pad_trim_frames(h, T, valid)
