"""Backbone interface + shared helpers (frame mapping, masking, layer mixing, path resolution).

Frame convention (matches common/contract.py)
---------------------------------------------
Output frame i covers [0.02 i, 0.02 (i + 1)) s, its centre is t_i = 0.02 (i + 0.5) s and
T = ceil(N / 320) for an N-sample 16 kHz input.

Every encoder produces "native" frames whose centres are c_k = first_center + k * hop (seconds).
`resample_to_frames` evaluates the native sequence at t_i by linear interpolation in time
(s_i = (t_i - first_center) / hop, edge-clamped to the sample's last valid native frame). When
first_center == hop / 2 and L * hop == T * 0.02 this is exactly
F.interpolate(mode="linear", align_corners=False); passing the true receptive-field centre removes
the few-ms bias that plain align_corners=False has for STFT/patch frontends.
Native 20 ms models use `pad_trim_frames` (frame i <- native frame min(i, valid - 1)).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SR = 16000          # == common.contract.SR
FRAME = 0.02        # == common.contract.FRAME
HOP = 320           # samples per 20 ms output frame
ROOT = Path(__file__).resolve().parents[2]   # t1_final/
MODEL_DIR = Path(os.environ.get("INDOML_MODEL_DIR", "~/indoml-offline/model")).expanduser()


# --------------------------------------------------------------------------------------------- utils
def n_out_frames(n_samples: int) -> int:
    return max(1, math.ceil(n_samples / HOP))


def prepare_lengths(wav: Tensor, lengths: Tensor | None) -> Tensor:
    if lengths is None:
        return torch.full((wav.shape[0],), wav.shape[-1], dtype=torch.long, device=wav.device)
    return lengths.to(device=wav.device, dtype=torch.long).clamp(1, wav.shape[-1])


def length_mask(valid: Tensor, size: int) -> Tensor:
    """[B] valid counts -> bool [B, size], True = valid."""
    return torch.arange(size, device=valid.device)[None, :] < valid[:, None]


def resample_to_frames(x: Tensor, T: int, hop: float, first_center: float, valid: Tensor | None = None) -> Tensor:
    """x [B, L, D] native frames (centres first_center + k*hop s) -> [B, T, D] at 20 ms centres (see module doc)."""
    B, L, D = x.shape
    t = (torch.arange(T, device=x.device, dtype=torch.float32) + 0.5) * FRAME
    pos = ((t - first_center) / hop).clamp(min=0.0)[None, :].expand(B, T)
    last = (torch.full((B,), L, device=x.device, dtype=torch.long) if valid is None
            else valid.to(x.device).long().clamp(1, L)) - 1
    pos = torch.minimum(pos, last[:, None].float())
    i0 = pos.floor().long()
    i1 = torch.minimum(i0 + 1, last[:, None])
    w = (pos - i0.float()).unsqueeze(-1).to(x.dtype)
    g0 = torch.gather(x, 1, i0.unsqueeze(-1).expand(B, T, D))
    g1 = torch.gather(x, 1, i1.unsqueeze(-1).expand(B, T, D))
    return g0 + (g1 - g0) * w


def pad_trim_frames(x: Tensor, T: int, valid: Tensor | None = None) -> Tensor:
    """Native 20 ms frames -> exactly T frames: frame i <- native min(i, valid-1) (edge replicate)."""
    B, L, D = x.shape
    last = (torch.full((B,), L, device=x.device, dtype=torch.long) if valid is None
            else valid.to(x.device).long().clamp(1, L)) - 1
    idx = torch.minimum(torch.arange(T, device=x.device)[None, :].expand(B, T), last[:, None])
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, T, D))


def masked_zero_mean_unit_var(wav: Tensor, lengths: Tensor, eps: float = 1e-7) -> Tensor:
    """Per-utterance waveform normalisation over valid samples (HF Wav2Vec2FeatureExtractor semantics)."""
    m = length_mask(lengths, wav.shape[-1]).to(wav.dtype)
    n = lengths.to(wav.dtype)[:, None]
    mean = (wav * m).sum(-1, keepdim=True) / n
    var = (((wav - mean) * m) ** 2).sum(-1, keepdim=True) / n
    return (wav - mean) / torch.sqrt(var + eps) * m


def chunked_encode(encode_fn, wav: Tensor, lengths: Tensor, win: int, min_overlap: int, grid: int):
    """Run `encode_fn` on windows of `win` samples and overlap-add the native frames.

    encode_fn(wav [B', win], lengths [B']) -> (feats [B', Lw, D], valid [B']) with native hop = grid samples
    and native frame k of a window starting at sample s being global frame s/grid + k.
    Window starts are multiples of `grid`; overlaps are blended with a trapezoid weight. Returns
    (feats [B, Lg, D], valid [B]) in global native frames, or a single pass if N <= win.
    """
    B, N = wav.shape
    if N <= win:
        return encode_fn(wav, lengths)
    span = N - win
    n = 1 + math.ceil(span / max(grid, win - min_overlap))
    starts = [(j * span // (n - 1)) // grid * grid for j in range(n - 1)] + [math.ceil(span / grid) * grid]
    wav = F.pad(wav, (0, starts[-1] + win - N))
    chunks = torch.stack([wav[:, s:s + win] for s in starts], 1).reshape(B * n, win)
    st = torch.tensor(starts, device=wav.device)
    clen = (lengths[:, None] - st[None, :]).clamp(0, win)                       # [B, n]
    feats, valid = encode_fn(chunks, clen.reshape(-1).clamp(min=1))
    Lw, D = feats.shape[1], feats.shape[2]
    feats = feats.reshape(B, n, Lw, D)
    valid = valid.reshape(B, n).clamp(1, Lw)
    ramp = max(1.0, min_overlap / grid / 2)
    k = torch.arange(Lw, device=wav.device, dtype=torch.float32)
    Lg = starts[-1] // grid + Lw
    out = feats.new_zeros(B, Lg, D, dtype=torch.float32)
    wsum = out.new_zeros(B, Lg)
    gvalid = torch.ones(B, dtype=torch.long, device=wav.device)
    for j, s in enumerate(starts):
        g = s // grid
        vj = valid[:, j]
        taper = torch.minimum(k + 0.5, vj[:, None].float() - k - 0.5).div(ramp).clamp(1e-3, 1.0)
        w = taper * (k[None, :] < vj[:, None]).float() * (clen[:, j] > 0).float()[:, None]
        out[:, g:g + Lw] += feats[:, j].float() * w[..., None]
        wsum[:, g:g + Lw] += w
        gvalid = torch.where(clen[:, j] > 0, torch.maximum(gvalid, g + vj), gvalid)
    out = out / wsum.clamp(min=1e-6)[..., None]
    return out.to(feats.dtype), gvalid


class LayerMix(nn.Module):
    """Learned softmax weighting over hidden states, then LayerNorm.

    per_layer_norm=True applies a parameter-free LayerNorm to each hidden state before mixing, so
    pre-LN residual streams (wavlm-large, w2v-bert, ATST intermediates) have comparable scale.
    """

    def __init__(self, n_layers: int, dim: int, per_layer_norm: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.per_layer_norm = per_layer_norm

    def forward(self, hidden: list[Tensor]) -> Tensor:
        if len(hidden) != self.weights.numel():
            raise ValueError(f"LayerMix expects {self.weights.numel()} layers, got {len(hidden)}")
        w = self.weights.softmax(0)
        out = 0
        for i, h in enumerate(hidden):
            if self.per_layer_norm:
                h = F.layer_norm(h, h.shape[-1:])
            out = out + w[i] * h
        return self.norm(out)


def resolve_file(explicit, filenames: list[str], env_var: str | None = None) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    if env_var and os.environ.get(env_var):
        cands.append(Path(os.environ[env_var]).expanduser())
    for fn in filenames:
        cands += [MODEL_DIR / fn, MODEL_DIR / "pretrainedsed" / fn, ROOT / "work/weights" / fn,
                  ROOT.parent / "Phase 2/assets" / fn]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(f"checkpoint not found; tried {[str(c) for c in cands]} "
                            f"(run models/backbones/fetch_weights.sh on a machine with internet)")


def resolve_hf(model: str, path=None) -> tuple[str, bool]:
    """Return (source, local_files_only). Local directories win; else HF id with the offline flag."""
    base = model.rstrip("/").split("/")[-1]
    cands = ([Path(path).expanduser()] if path else []) + [Path(model).expanduser(), MODEL_DIR / base, MODEL_DIR / model]
    for c in cands:
        if c.is_dir() and (c / "config.json").is_file():
            return str(c), True
    if path:
        raise FileNotFoundError(f"{path} is not a HF model directory (needs config.json)")
    offline = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes") or \
        os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true", "yes")
    return model, offline


def _dedupe(params):
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


# ------------------------------------------------------------------------------------------ interface
class Backbone(nn.Module):
    """forward(wav [B, N] float32 16 kHz, lengths [B] valid samples) -> [B, ceil(N/320), out_dim]."""

    out_dim: int

    def __init__(self):
        super().__init__()
        self._encoder_frozen = False

    # modules holding pretrained weights (used by freeze_encoder / train())
    def encoder_modules(self) -> list[nn.Module]:
        return []

    def encoder_parameters(self) -> list[nn.Parameter]:
        unused = self._unused_ids()
        return _dedupe(p for m in self.encoder_modules() for p in m.parameters() if id(p) not in unused)

    def new_parameters(self) -> list[nn.Parameter]:
        skip = {id(p) for p in self.encoder_parameters()} | self._unused_ids()
        return [p for p in self.parameters() if id(p) not in skip]

    def layer_groups(self) -> list[list[nn.Parameter]]:
        enc = self.encoder_parameters()
        return [enc] if enc else []

    def freeze_encoder(self, frozen: bool = True):
        """Optional helper: requires_grad=False on encoder params and keep encoder modules in eval()."""
        self._encoder_frozen = bool(frozen)
        for p in self.encoder_parameters():
            p.requires_grad_(not frozen)
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            for m in self.encoder_modules():
                m.eval()
        return self

    def _mark_unused(self, *params):
        """Exclude params that never receive gradients (e.g. SSL mask embeddings) from all lists.
        Tracked by name so it survives deepcopy / state_dict round trips."""
        ids = {id(p) for p in params}
        for p in params:
            p.requires_grad_(False)
        names = set(getattr(self, "_unused_names", ()))
        names |= {n for n, p in self.named_parameters() if id(p) in ids}
        self._unused_names = names

    def _unused_ids(self) -> set:
        names = getattr(self, "_unused_names", ())
        return {id(p) for n, p in self.named_parameters() if n in names} if names else set()


def group_params_by_layer(named_params, n_layers: int, layer_of, exclude=()) -> list[list[nn.Parameter]]:
    """layer_of(name) -> group index in [0, n_layers) or None (skip); params whose id is in exclude are skipped."""
    groups = [[] for _ in range(n_layers)]
    for name, p in named_params:
        if id(p) in exclude:
            continue
        g = layer_of(name)
        if g is not None:
            groups[g].append(p)
    return [g for g in groups if g]
