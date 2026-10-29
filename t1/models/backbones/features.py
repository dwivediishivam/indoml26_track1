"""Batched, GPU-friendly feature frontends in pure torch.

`kaldi_fbank` re-implements torchaudio.compliance.kaldi.fbank (defaults: povey window, dither=0,
remove_dc_offset, preemphasis 0.97, 25 ms / 10 ms, snip_edges=True, power spectrum, 512-pt FFT,
kaldi mel banks low_freq=20 high_freq=nyquist, log with float32-eps floor) for a whole batch at once.
It is the frontend of BEATs (128 bins) and of w2v-BERT 2.0's SeamlessM4TFeatureExtractor (80 bins;
HF builds it to match kaldi fbank). Frame j covers samples [160 j, 160 j + 400): centre 10 j + 12.5 ms,
F = 1 + (N - 400) // 160 frames. Pass impl="torchaudio" to the backbones to use the reference loop instead.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

_MEL_CACHE: dict = {}


def _mel(f):
    return 1127.0 * torch.log1p(f / 700.0)


def kaldi_mel_banks(num_bins: int, padded_window: int, sample_freq: float, low_freq: float, high_freq: float,
                    device, dtype=torch.float32) -> Tensor:
    """[num_bins, padded_window // 2 + 1] (Nyquist column zero), == torchaudio get_mel_banks + pad."""
    key = (num_bins, padded_window, sample_freq, low_freq, high_freq, str(device), dtype)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    num_fft_bins = padded_window // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / padded_window
    mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
    delta = (mel_high - mel_low) / (num_bins + 1)
    b = torch.arange(num_bins, dtype=torch.float64).unsqueeze(1)
    left, center, right = mel_low + b * delta, mel_low + (b + 1.0) * delta, mel_low + (b + 2.0) * delta
    mel = _mel(fft_bin_width * torch.arange(num_fft_bins, dtype=torch.float64)).unsqueeze(0)
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    banks = torch.clamp(torch.minimum(up, down), min=0.0)
    banks = F.pad(banks, (0, 1)).to(device=device, dtype=dtype)
    _MEL_CACHE[key] = banks
    return banks


def kaldi_fbank(wav: Tensor, num_mel_bins: int, sample_frequency: float = 16000.0, frame_length_ms: float = 25.0,
                frame_shift_ms: float = 10.0, preemphasis: float = 0.97, low_freq: float = 20.0,
                high_freq: float = 0.0) -> Tensor:
    """wav [B, N] (already scaled, e.g. * 2**15) -> log mel fbank [B, F, num_mel_bins], float32."""
    with torch.autocast(device_type=wav.device.type, enabled=False):
        wav = wav.float()
        win = int(sample_frequency * frame_length_ms * 0.001)
        shift = int(sample_frequency * frame_shift_ms * 0.001)
        padded = 1 << (win - 1).bit_length()
        if wav.shape[-1] < win:
            wav = F.pad(wav, (0, win - wav.shape[-1]))
        frames = wav.unfold(-1, win, shift)                                    # [B, F, win]
        frames = frames - frames.mean(-1, keepdim=True)                        # remove_dc_offset
        prev = torch.cat([frames[..., :1], frames[..., :-1]], -1)             # replicate-pad preemphasis
        frames = frames - preemphasis * prev
        window = torch.hann_window(win, periodic=False, device=wav.device, dtype=torch.float32).pow(0.85)
        frames = F.pad(frames * window, (0, padded - win))
        power = torch.fft.rfft(frames).abs().pow(2.0)                         # [B, F, padded/2+1]
        banks = kaldi_mel_banks(num_mel_bins, padded, sample_frequency, low_freq, high_freq, wav.device)
        mel = power @ banks.T
        return torch.clamp(mel, min=torch.finfo(torch.float32).eps).log()


def torchaudio_fbank(wav: Tensor, num_mel_bins: int) -> Tensor:
    """Reference per-utterance loop through torchaudio.compliance.kaldi (same output layout)."""
    import torchaudio.compliance.kaldi as ta_kaldi
    with torch.autocast(device_type=wav.device.type, enabled=False):
        return torch.stack([ta_kaldi.fbank(w[None].float(), num_mel_bins=num_mel_bins, sample_frequency=16000,
                                           frame_length=25, frame_shift=10) for w in wav], 0)


def fbank(wav: Tensor, num_mel_bins: int, impl: str = "torch") -> Tensor:
    if impl == "torch":
        return kaldi_fbank(wav, num_mel_bins)
    if impl == "torchaudio":
        return torchaudio_fbank(wav, num_mel_bins)
    raise ValueError(f"unknown fbank impl {impl!r}")


def fbank_valid_frames(lengths: Tensor, win: int = 400, shift: int = 160) -> Tensor:
    return (1 + (lengths - win).div(shift, rounding_mode="floor")).clamp(min=1)
