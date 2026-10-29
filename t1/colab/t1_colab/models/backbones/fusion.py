"""Fusion of several backbones at the shared 20 ms grid.

build_backbone("fusion", parts=[{"name": "atst_strong", ...}, {"name": "mel_cnn"}], proj_dim=512)
Each part -> [B, T, D_i]; per-part LayerNorm (norm="per_part", default; "joint" = one LayerNorm over the
concatenation) -> concat -> Linear(sum D_i, proj_dim). part_dropout=p zeroes a whole part's features per
sample with probability p during training (never all parts of a sample).
encoder_parameters / new_parameters = unions of the parts' lists (+ norms and projection in new_parameters).
layer_groups: parts' groups aligned at the deep end (deepest group of every part shares the last group).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import Backbone, _dedupe


class Fusion(Backbone):
    def __init__(self, parts, proj_dim: int = 512, norm: str = "per_part", part_dropout: float = 0.0):
        super().__init__()
        from . import build_backbone
        if not parts:
            raise ValueError("fusion needs at least one part")
        specs = [dict(p) for p in parts]
        self.parts = nn.ModuleList([build_backbone(s.pop("name"), **s) for s in specs])
        dims = [p.out_dim for p in self.parts]
        if norm == "per_part":
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
            self.joint_norm = nn.Identity()
        elif norm == "joint":
            self.norms = nn.ModuleList([nn.Identity() for _ in dims])
            self.joint_norm = nn.LayerNorm(sum(dims))
        else:
            raise ValueError("norm must be 'per_part' or 'joint'")
        self.proj = nn.Linear(sum(dims), proj_dim)
        self.part_dropout = float(part_dropout)
        self.out_dim = proj_dim

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        feats = [n(p(wav, lengths)) for p, n in zip(self.parts, self.norms)]
        if self.training and self.part_dropout > 0 and len(feats) > 1:
            B = wav.shape[0]
            keep = torch.rand(B, len(feats), device=wav.device) >= self.part_dropout
            none = ~keep.any(1)
            keep[none, torch.randint(len(feats), (int(none.sum()),), device=wav.device)] = True
            feats = [f * keep[:, i, None, None].to(f.dtype) for i, f in enumerate(feats)]
        return self.proj(self.joint_norm(torch.cat(feats, -1)))

    def encoder_parameters(self):
        return _dedupe(p for part in self.parts for p in part.encoder_parameters())

    def new_parameters(self):
        own = list(self.norms.parameters()) + list(self.joint_norm.parameters()) + list(self.proj.parameters())
        return _dedupe([p for part in self.parts for p in part.new_parameters()] + own)

    def layer_groups(self):
        groups = [part.layer_groups() for part in self.parts]
        depth = max((len(g) for g in groups), default=0)
        merged = [[] for _ in range(depth)]
        for g in groups:
            off = depth - len(g)
            for i, ps in enumerate(g):
                merged[off + i].extend(ps)
        return [_dedupe(m) for m in merged if m]

    def freeze_encoder(self, frozen: bool = True):
        for part in self.parts:
            part.freeze_encoder(frozen)
        return self

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        for part in self.parts:
            part.train(mode)
        return self
