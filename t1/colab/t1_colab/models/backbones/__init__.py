"""Frame-level audio backbones for Track 1.  Usage: `from models.backbones import build_backbone`.

Every backbone: forward(wav [B, N] float32 16 kHz, lengths [B] valid samples) -> [B, ceil(N / 320), out_dim]
(20 ms frames, frame i covers [0.02 i, 0.02 (i+1)) s; padding frames are edge-replicated garbage),
plus encoder_parameters() / new_parameters() / layer_groups() (shallow -> deep) and an optional
freeze_encoder(bool) helper. Submodules are imported lazily so e.g. transformers is only needed for
wavlm / w2vbert. See README.md for checkpoints and per-backbone kwargs.
"""
from __future__ import annotations

import importlib

from .base import FRAME, HOP, SR, Backbone, n_out_frames, pad_trim_frames, resample_to_frames

_REGISTRY = {
    "atst_strong": ("atst", "ATSTStrong", {}),
    "atst": ("atst", "ATSTStrong", {}),
    "beats_strong": ("beats", "BEATsStrong", {}),
    "beats": ("beats", "BEATsStrong", {}),
    "wavlm": ("hf_ssl", "WavLMBackbone", {}),
    "wavlm_base_plus": ("hf_ssl", "WavLMBackbone", {"model": "microsoft/wavlm-base-plus"}),
    "wavlm_large": ("hf_ssl", "WavLMBackbone", {"model": "microsoft/wavlm-large"}),
    "w2vbert": ("hf_ssl", "W2VBertBackbone", {}),
    "w2v_bert": ("hf_ssl", "W2VBertBackbone", {}),
    "w2v_bert_2.0": ("hf_ssl", "W2VBertBackbone", {}),
    "mel_cnn": ("mel_cnn", "MelCNN", {}),
    "fusion": ("fusion", "Fusion", {}),
}


def available_backbones() -> list[str]:
    return sorted(_REGISTRY)


def build_backbone(name: str, **kwargs) -> Backbone:
    key = name.strip().lower().replace("-", "_")
    if key not in _REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; available: {available_backbones()}")
    module, cls, defaults = _REGISTRY[key]
    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, cls)(**{**defaults, **kwargs})


__all__ = ["Backbone", "build_backbone", "available_backbones", "n_out_frames", "resample_to_frames",
           "pad_trim_frames", "SR", "FRAME", "HOP"]
