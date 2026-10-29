"""Inference-side copy of the trainer's SED network (no training-only bits)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

CATEGORIES = ("animal", "vehicle_traffic", "baby_child", "singing_music",
              "phone_signal_alarm", "appliance_machine", "human_non_speech")


class SED(nn.Module):
    def __init__(self, backbone="microsoft/wavlm-base-plus", hidden=256, n_out=72,
                 pretrained=False):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if pretrained:
            self.encoder = AutoModel.from_pretrained(backbone)
        else:
            cfg = AutoConfig.from_pretrained(backbone)
            self.encoder = AutoModel.from_config(cfg)
        nl = self.encoder.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(nl))
        d = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 384), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(384, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.15)
        h = 2 * hidden
        self.norm = nn.LayerNorm(h)
        self.frame = nn.Linear(h, n_out)
        self.bound = nn.Linear(h, 2)
        self.att = nn.Linear(h, n_out)

    def forward(self, wav):
        out = self.encoder(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)
        w = torch.softmax(self.layer_w[: hs.shape[0]], 0).view(-1, 1, 1, 1)
        x = (hs * w).sum(0)
        x = self.proj(x)
        x, _ = self.gru(x)
        x = self.norm(x)
        return self.frame(x).transpose(1, 2), self.bound(x).transpose(1, 2)


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    n_out = sd["frame.weight"].shape[0]
    model = SED(ck.get("backbone", "microsoft/wavlm-base-plus"), n_out=n_out)
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model, ck
