"""SEDModel: backbone -> masked 2-layer BiGRU -> presence / boundary heads (+ MIL, optional tag head).

Output `logits` [B, 24, T] follows the common.contract channel order:
  0 union presence, 1..7 category presence, 8/9 union onset/offset, 10..16 category onsets, 17..23 offsets.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.contract import CH_CAT, N_CH

HOP = 320


def _build_backbone(cfg: dict):
    from models.backbones import build_backbone       # written by the backbone workstream
    cfg = dict(cfg)
    name = cfg.pop("name")
    return build_backbone(name, **cfg)


class SEDModel(nn.Module):
    def __init__(self, backbone_cfg: dict, hidden: int = 256, gru_layers: int = 2, dropout: float = 0.1,
                 tag_head: bool = False, feat_time_mask: int = 0):
        super().__init__()
        self.backbone = _build_backbone(backbone_cfg)
        dim = int(self.backbone.out_dim)
        self.out_dim_in = dim
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(2 * hidden, hidden, gru_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.presence = nn.Linear(2 * hidden, 8)       # union + 7 categories
        self.boundary = nn.Linear(2 * hidden, 16)      # union on, union off, 7 cat on, 7 cat off
        nn.init.constant_(self.presence.bias, -2.0)
        nn.init.constant_(self.boundary.bias, -4.0)
        self.tag_head = bool(tag_head)
        if self.tag_head:
            self.tag = nn.Linear(2 * hidden, 7)
            self.tag_att = nn.Linear(2 * hidden, 7)
        self.feat_time_mask = int(feat_time_mask)

    # ------------------------------------------------------------------ parameter partitions
    def encoder_parameters(self):
        return list(self.backbone.encoder_parameters())

    def new_parameters(self):
        """Backbone's new params + everything outside the backbone."""
        own = [p for n, p in self.named_parameters() if not n.startswith("backbone.")]
        return list(self.backbone.new_parameters()) + own

    def layer_groups(self):
        return list(self.backbone.layer_groups())

    # ------------------------------------------------------------------ forward
    def forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        b, n = wav.shape
        t = -(-n // HOP)
        feats = self.backbone(wav, lengths)
        if feats.dim() != 3 or feats.shape[0] != b or feats.shape[1] != t or feats.shape[2] != self.out_dim_in:
            raise RuntimeError(f"backbone returned {tuple(feats.shape)}, expected [{b}, {t}, {self.out_dim_in}] "
                               f"(T = ceil(N/320) for N={n})")
        frame_len = torch.div(lengths.long() + HOP - 1, HOP, rounding_mode="floor").clamp(1, t)
        valid = torch.arange(t, device=wav.device)[None] < frame_len[:, None]          # [B, T]
        x = self.proj(feats)
        if self.training and self.feat_time_mask > 0:
            x = x.clone()
            for i in range(b):
                width = min(self.feat_time_mask, int(frame_len[i]) // 10)
                if width > 0:
                    s = int(torch.randint(0, int(frame_len[i]) - width + 1, (1,)))
                    x[i, s:s + width] = 0
        packed = pack_padded_sequence(x, frame_len.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.gru(packed)
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=t)
        y = self.drop(self.norm(y))
        logits = torch.cat([self.presence(y), self.boundary(y)], dim=-1).transpose(1, 2).float()   # [B, 24, T]
        if logits.shape[1] != N_CH:
            raise AssertionError("channel layout mismatch")
        # clip-level MIL head: linear-softmax pooling over the 7 category presence tracks
        p = torch.sigmoid(logits[:, CH_CAT]) * valid[:, None].float()                # [B, 7, T]
        clip_p = (p * p).sum(-1) / p.sum(-1).clamp_min(1e-6)
        out = {"logits": logits, "clip_logits": torch.logit(clip_p.clamp(1e-5, 1 - 1e-5)), "valid": valid}
        if self.tag_head:
            att = self.tag_att(y).float().masked_fill(~valid[:, :, None], -1e4).softmax(dim=1)
            out["tag_logits"] = (self.tag(y).float() * att).sum(1)                   # [B, 7]
        return out


def build_model(cfg: dict) -> SEDModel:
    m = cfg["model"]
    return SEDModel(cfg["backbone"], hidden=m["hidden"], gru_layers=m["gru_layers"], dropout=m["dropout"],
                    tag_head=m["tag_head"], feat_time_mask=m["feat_time_mask"])
