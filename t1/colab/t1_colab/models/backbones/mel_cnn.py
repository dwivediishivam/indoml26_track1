"""Small randomly initialised high-resolution branch.

128-bin log-mel, 25 ms Hann window (n_fft 512), 10 ms hop, center=True (frame j centre = 10 j ms),
instance-normalised per clip over its valid frames (norm="bin": per mel bin; "global": one mean/std).
5 conv blocks (2 x [3x3 conv -> channel LayerNorm -> GELU], then 2x average pool over frequency only):
128 -> 4 mel bins. Flatten (C x 4) -> stride-2 average pool over time (frames 2k, 2k+1 -> 20 ms,
centre 20 k + 5 ms) -> Linear -> LayerNorm. Then pad/trim to T. Normalisation is per position (channel
LayerNorm) and padded frames are re-zeroed after every block, so padding never enters any statistic.
All parameters are new_parameters().
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import Backbone, length_mask, n_out_frames, pad_trim_frames, prepare_lengths


class ChannelNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):                      # [B, C, F, T]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), ChannelNorm(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), ChannelNorm(cout), nn.GELU(),
                                 nn.Dropout2d(dropout) if dropout else nn.Identity())

    def forward(self, x, tmask):               # tmask [B, 1, 1, T]
        x = self.net(x * tmask)
        return F.avg_pool2d(x, (2, 1)) * tmask


class MelCNN(Backbone):
    """kwargs: n_mels=128, channels=(16, 32, 64, 128, 256), out_dim=256, norm="bin"|"global",
    f_min=20.0, f_max=8000.0, dropout=0.0."""

    def __init__(self, n_mels: int = 128, channels=(16, 32, 64, 128, 256), out_dim: int = 256,
                 norm: str = "bin", f_min: float = 20.0, f_max: float = 8000.0, dropout: float = 0.0):
        super().__init__()
        import torchaudio
        if norm not in ("bin", "global"):
            raise ValueError("norm must be 'bin' or 'global'")
        self.mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=400, hop_length=160,
                                                        f_min=f_min, f_max=f_max, n_mels=n_mels, power=2.0,
                                                        center=True)
        self.norm_mode = norm
        chans = [1] + list(channels)
        self.blocks = nn.ModuleList([ConvBlock(chans[i], chans[i + 1], dropout) for i in range(len(channels))])
        f_out = n_mels // (2 ** len(channels))
        if f_out < 1:
            raise ValueError("too many blocks for n_mels")
        self.proj = nn.Sequential(nn.Linear(chans[-1] * f_out, out_dim), nn.LayerNorm(out_dim))
        self.out_dim = out_dim

    def logmel(self, wav: Tensor, lengths: Tensor):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = torch.log(self.mel(wav.float()).clamp(min=1e-8))                 # [B, n_mels, M]
            M = x.shape[-1]
            valid = (lengths // 160 + 1).clamp(1, M)
            m = length_mask(valid, M)[:, None, :].float()
            dims = (-1,) if self.norm_mode == "bin" else (-2, -1)
            n = m.sum(dims, keepdim=True) * (1 if self.norm_mode == "bin" else x.shape[1])
            mean = (x * m).sum(dims, keepdim=True) / n
            var = (((x - mean) * m) ** 2).sum(dims, keepdim=True) / n
            x = (x - mean) * torch.rsqrt(var + 1e-5) * m
        return x, valid

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        lengths = prepare_lengths(wav, lengths)
        T = n_out_frames(wav.shape[-1])
        x, valid = self.logmel(wav, lengths)
        tmask = length_mask(valid, x.shape[-1])[:, None, None, :].to(x.dtype)
        x = x.unsqueeze(1)                                                         # [B, 1, F, M]
        for blk in self.blocks:
            x = blk(x, tmask)
        x = x.flatten(1, 2)                                                        # [B, C*F', M]
        x = F.avg_pool1d(x, 2, 2)                                                  # 10 ms -> 20 ms
        x = self.proj(x.transpose(1, 2))                                           # [B, M//2, out_dim]
        v20 = torch.div(valid + 1, 2, rounding_mode="floor").clamp(1, x.shape[1])
        return pad_trim_frames(x, T, v20)

    def encoder_parameters(self):
        return []

    def new_parameters(self):
        return list(self.parameters())

    def layer_groups(self):
        return []
