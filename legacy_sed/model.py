"""WavLM frame detector with auxiliary event-category targets."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Detector(nn.Module):
    def __init__(self, backbone="microsoft/wavlm-base-plus", hidden=160,
                 drop_last_layers=2, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.backbone_name = backbone
        if pretrained:
            self.encoder = AutoModel.from_pretrained(backbone)
        else:
            try:
                config = AutoConfig.from_pretrained(backbone, local_files_only=True)
            except OSError:
                config = AutoConfig.from_pretrained(backbone)
            self.encoder = AutoModel.from_config(config)
        if drop_last_layers:
            self.encoder.encoder.layers = nn.ModuleList(
                list(self.encoder.encoder.layers)[:-drop_last_layers])
        self.encoder.feature_extractor._freeze_parameters()
        self.temporal = nn.GRU(self.encoder.config.hidden_size, hidden,
                               num_layers=2, batch_first=True, bidirectional=True,
                               dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(0.1),
                                  nn.Linear(2 * hidden, 8))

    def freeze_encoder(self, frozen):
        for param in self.encoder.parameters():
            param.requires_grad_(not frozen)
        self.encoder.feature_extractor._freeze_parameters()

    def forward(self, waveform, output_frames=None):
        features = self.encoder(waveform).last_hidden_state
        features, _ = self.temporal(features)
        logits = self.head(features).transpose(1, 2)
        if output_frames is not None and logits.shape[-1] != output_frames:
            logits = F.interpolate(logits, size=output_frames, mode="linear",
                                   align_corners=False)
        return logits
