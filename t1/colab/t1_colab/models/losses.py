"""Losses on the 24-channel contract layout. Everything is masked by valid frames.

presence  : BCE or focal on channels 0..7 (union + categories), per-tier weights, Silver background x0.5
dice      : per-clip soft Dice on the union channel
boundary  : focal BCE vs Gaussian targets on channels 8..23, RED-style weight 1 + lambda * target,
            normalised per clip, then per-tier boundary weights
mil       : BCE of the linear-softmax pooled clip probabilities vs clip_cats (Bronze; small weight elsewhere)
tag       : optional clip tag head BCE
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from common.contract import CATEGORIES, N_CH
from models.config import TIERS

BRONZE, SILVER = TIERS.index("bronze"), TIERS.index("silver")


def _tier_table(weights: dict, device):
    return torch.tensor([float(weights[t]) for t in TIERS], device=device)


def channel_weights(loss_cfg, device):
    w = torch.ones(N_CH, device=device)
    w[0] = loss_cfg["union_weight"]
    w[1:8] = loss_cfg["category_weight"]
    w[10:24] = loss_cfg["category_boundary_weight"]
    for cat, mult in loss_cfg["channel_multipliers"].items():
        c = CATEGORIES.index(cat)
        for ch in (1 + c, 10 + c, 17 + c):
            w[ch] = w[ch] * float(mult)
    return w


def _focal_bce(logits, target, gamma):
    """Focal BCE that also works for soft targets: BCE * |target - p|^gamma."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if gamma <= 0:
        return bce
    return bce * (target - torch.sigmoid(logits)).abs().pow(gamma)


def compute_losses(out: dict, batch: dict, loss_cfg: dict):
    logits = out["logits"].float()
    y, m = batch["target"], batch["mask"]
    tier = batch["tier"]
    dev = logits.device
    if logits.shape != y.shape:
        raise RuntimeError(f"logits {tuple(logits.shape)} vs targets {tuple(y.shape)}")
    strong = (tier != BRONZE).float()
    n_strong = strong.sum().clamp_min(1.0)
    pw = _tier_table(loss_cfg["presence_tier_weights"], dev)[tier] * batch["weight"] * strong      # [B]
    bw = _tier_table(loss_cfg["boundary_tier_weights"], dev)[tier] * batch["boundary_weight"] * strong
    chw = channel_weights(loss_cfg, dev)

    # ---- presence (union + categories)
    py, pl = y[:, :8], logits[:, :8]
    if loss_cfg["presence"] == "focal":
        per = _focal_bce(pl, py, loss_cfg["presence_focal_gamma"])
    else:
        per = F.binary_cross_entropy_with_logits(pl, py, reduction="none")
    fw = m[:, None, :] * chw[:8, None] * pw[:, None, None]
    silver_bg = (tier == SILVER)[:, None, None] & (py < 0.5)
    fw = torch.where(silver_bg, fw * loss_cfg["silver_background_weight"], fw)
    # mean over (strong clip, valid frame, channel); multipliers act as absolute up-weights
    denom = ((m.sum(1) * strong).sum() * 8.0).clamp_min(1.0)
    presence = (per * fw).sum() / denom

    # ---- per-clip soft Dice on union
    p = torch.sigmoid(logits[:, 0]) * m
    t = y[:, 0] * m
    s = loss_cfg["dice_smooth"]
    d = 1 - (2 * (p * t).sum(1) + s) / (p.sum(1) + t.sum(1) + s)
    dice = (d * pw).sum() / n_strong

    # ---- boundaries (union + category onsets/offsets)
    by, bl = y[:, 8:], logits[:, 8:]
    bper = _focal_bce(bl, by, loss_cfg["boundary_gamma"])
    w = (1 + loss_cfg["boundary_pos_lambda"] * by) * m[:, None, :] * chw[8:, None]
    per_clip = (bper * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1.0)
    boundary = (per_clip * bw).sum() / n_strong

    # ---- MIL (clip-level linear-softmax pooling)
    mw = _tier_table(loss_cfg["mil_tier_weights"], dev)[tier]
    ml = F.binary_cross_entropy_with_logits(out["clip_logits"].float(), batch["clip_cats"], reduction="none").mean(1)
    mil = (ml * mw).sum() / (mw > 0).float().sum().clamp_min(1.0)

    total = (loss_cfg["presence_weight"] * presence + loss_cfg["dice_weight"] * dice
             + loss_cfg["boundary_weight"] * boundary + loss_cfg["mil_weight"] * mil)
    parts = {"presence": presence, "dice": dice, "boundary": boundary, "mil": mil}

    if "tag_logits" in out:
        tw = _tier_table(loss_cfg["tag_tier_weights"], dev)[tier]
        tl = F.binary_cross_entropy_with_logits(out["tag_logits"].float(), batch["clip_cats"], reduction="none").mean(1)
        tag = (tl * tw).sum() / (tw > 0).float().sum().clamp_min(1.0)
        total = total + loss_cfg["tag_weight"] * tag
        parts["tag"] = tag
    return total, parts
