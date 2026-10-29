"""Strict YAML configuration: every key must exist in DEFAULTS (except free-form sections).

Free-form sections are passed verbatim to other components:
  backbone          -> models.backbones.build_backbone(name, **rest)
  synthetic.kwargs  -> the synthgen Dataset constructor
  loss.channel_multipliers -> {category: multiplier} (validated against CATEGORIES)
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import yaml

from common.contract import CATEGORIES

TIERS = ("gold", "silver", "bronze", "synthetic", "val_natural")
STRONG_TIERS = ("gold", "silver", "synthetic", "val_natural")

DEFAULTS = {
    "name": "",
    "seed": 17,
    "backbone": {},
    "model": {
        "hidden": 256,
        "gru_layers": 2,
        "dropout": 0.1,
        "tag_head": False,
        "feat_time_mask": 0,          # max width (frames) of one feature time-mask per item in training; 0 = off
    },
    "data": {
        "staged_dir": "~/indoml-work/data",   # float16 blob + manifest (Phase 2 staging format)
        "val_dir": None,                       # None -> common.contract.VAL_DIR
        "work_dir": None,                      # None -> common.contract.WORK (folds.json lives here)
        "crop_seconds": 10.0,
        "bronze_max_seconds": 10.0,            # Bronze uses whole clips; longer ones are excluded
        "event_crop_prob": 0.8,
        "background_crop_prob": 0.1,
        "sigma_frames": {"gold": 1.0, "silver": 1.5, "val_natural": 1.0},
        "val_holdout_folds": [4],
        "val_include_synthetic": False,
    },
    "sources": {"gold": 0.0, "silver": 0.0, "bronze": 0.0, "synthetic": 0.0, "val_natural": 0.0},
    # factory(**kwargs) -> map-style Dataset of synthgen items; models.data:build_synthgen wraps synthgen
    "synthetic": {"factory": "models.data:build_synthgen", "kwargs": {}},
    "augment": {
        "gain_db": 6.0,
        "time_shift_prob": 0.3,
        "time_shift_max_frames": 25,          # 0.5 s
        "mixup_prob": 0.15,
        "mixup_lam": [0.4, 0.6],
        "filter_aug_prob": 0.5,
        "filter_aug_db": 6.0,
        "filter_aug_bands": [3, 6],
        "freq_mixstyle_prob": 0.0,
        "freq_mixstyle_alpha": 0.3,
    },
    "loss": {
        "presence": "bce",                     # bce | focal
        "presence_focal_gamma": 2.0,
        "presence_weight": 1.0,
        "union_weight": 1.0,
        "category_weight": 1.0,
        "dice_weight": 0.5,
        "dice_smooth": 1.0,
        "boundary_weight": 1.0,
        "boundary_gamma": 2.0,
        "boundary_pos_lambda": 10.0,           # RED-style positive-heavy weight: w = 1 + lambda * target
        "category_boundary_weight": 1.0,
        "mil_weight": 0.5,
        "tag_weight": 0.2,
        "silver_background_weight": 0.5,
        "channel_multipliers": {},             # {category: multiplier}, applied to presence + boundaries
        "presence_tier_weights": {"gold": 1.0, "silver": 0.5, "bronze": 0.0, "synthetic": 1.0, "val_natural": 1.5},
        "boundary_tier_weights": {"gold": 1.0, "silver": 0.1, "bronze": 0.0, "synthetic": 1.0, "val_natural": 1.5},
        "mil_tier_weights": {"gold": 0.2, "silver": 0.1, "bronze": 1.0, "synthetic": 0.2, "val_natural": 0.2},
        "tag_tier_weights": {"gold": 1.0, "silver": 0.5, "bronze": 1.0, "synthetic": 1.0, "val_natural": 1.0},
    },
    "optim": {
        "batch_size": 16,
        "head_lr": 1.0e-3,
        "encoder_lr": 2.0e-5,
        "layer_decay": 0.85,
        "weight_decay": 0.01,
        "warmup_steps": 200,
        "encoder_warmup_steps": 300,
        "min_lr_ratio": 0.05,
        "freeze_steps": 1500,
        "max_steps": 40000,
        "grad_clip": 5.0,
        "ema_decay": 0.999,
        "max_bad_steps": 20,
    },
    "train": {
        "workers": 12,
        "prefetch_factor": 4,
        "log_every": 25,
        "monitor_every": 1000,
        "first_monitor": 500,
        "save_minutes": 10.0,
        "keep_best": 2,
        "benchmark_warmup": 10,
        "benchmark_steps": 100,
        "stage2_slowdown_guess": 2.5,
        "monitor_seconds_guess": 120.0,
        "reserve_seconds": 240.0,              # kept free before the deadline for final monitor + save
        "ddp_find_unused": True,
    },
    "monitor": {
        "folds": [4],
        "max_clips": 0,                        # 0 = every clip of the monitor folds
        "threshold": 0.5,
        "min_frames": 3,
        "gap_frames": 3,
        "dup_iou": 0.7,
        "batch_size": 16,
        "window_seconds": 10.0,
        "hop_seconds": 5.0,
    },
}

FREEFORM = {("backbone",), ("synthetic", "kwargs"), ("loss", "channel_multipliers")}


def _merge(defaults, user, path=()):
    out = copy.deepcopy(defaults)
    if not isinstance(user, dict):
        raise ValueError(f"config section {'.'.join(path) or '<root>'} must be a mapping")
    for key, value in user.items():
        p = path + (key,)
        if key not in defaults:
            raise ValueError(f"unknown config key: {'.'.join(p)}")
        if isinstance(defaults[key], dict) and p not in FREEFORM:
            out[key] = _merge(defaults[key], value, p)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_path(tree, dotted: str, value):
    parts = dotted.split(".")
    node = tree
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(node, list):
            node = node[int(part)]
            continue
        if part not in node or node[part] is None:
            node[part] = [] if nxt.isdigit() else {}
        node = node[part]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def expand_paths(x):
    """Expand ~ and $VARS in every string (paths in backbone / synthetic kwargs / data)."""
    if isinstance(x, dict):
        return {k: expand_paths(v) for k, v in x.items()}
    if isinstance(x, list):
        return [expand_paths(v) for v in x]
    if isinstance(x, str) and (x.startswith("~") or "$" in x):
        return os.path.expandvars(os.path.expanduser(x))
    return x


def load_config(path, overrides=()):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, text = item.split("=", 1)
        _set_path(raw, key.strip(), yaml.safe_load(text))
    cfg = _merge(DEFAULTS, raw)
    if not cfg["name"]:
        cfg["name"] = Path(path).stem
    for section in ("backbone", "synthetic", "data"):
        cfg[section] = expand_paths(cfg[section])
    validate(cfg)
    return cfg


def validate(cfg):
    if "name" not in cfg["backbone"]:
        raise ValueError("backbone.name is required")
    fr = cfg["sources"]
    if any((not isinstance(v, (int, float))) or v < 0 or not math.isfinite(v) for v in fr.values()):
        raise ValueError(f"sources fractions must be finite and >= 0: {fr}")
    total = sum(fr.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"sources fractions must sum to 1, got {total:.6f}: {fr}")
    if fr["synthetic"] > 0 and ":" not in str(cfg["synthetic"]["factory"]):
        raise ValueError("synthetic.factory must be 'module.path:ClassName'")
    if cfg["loss"]["presence"] not in ("bce", "focal"):
        raise ValueError("loss.presence must be 'bce' or 'focal'")
    bad = set(cfg["loss"]["channel_multipliers"]) - set(CATEGORIES)
    if bad:
        raise ValueError(f"loss.channel_multipliers has unknown categories: {sorted(bad)}")
    for key in ("presence_tier_weights", "boundary_tier_weights", "mil_tier_weights", "tag_tier_weights"):
        if set(cfg["loss"][key]) != set(TIERS):
            raise ValueError(f"loss.{key} must list every tier {TIERS}")
    if set(cfg["data"]["sigma_frames"]) - {"gold", "silver", "val_natural"}:
        raise ValueError("data.sigma_frames keys must be among gold/silver/val_natural")
    o, t = cfg["optim"], cfg["train"]
    for k in ("batch_size", "max_steps"):
        if not isinstance(o[k], int) or o[k] < 1:
            raise ValueError(f"optim.{k} must be a positive int")
    if not isinstance(o["freeze_steps"], int) or o["freeze_steps"] < 0 or o["freeze_steps"] >= o["max_steps"]:
        raise ValueError("optim.freeze_steps must be an int in [0, max_steps)")
    if o["head_lr"] <= 0 or o["encoder_lr"] <= 0 or not 0 < o["layer_decay"] <= 1 or not 0 <= o["ema_decay"] < 1:
        raise ValueError("invalid optimiser settings")
    if not isinstance(t["workers"], int) or t["workers"] < 0:
        raise ValueError("train.workers must be an int >= 0")
    if t["monitor_every"] < 1 or t["keep_best"] < 1 or t["benchmark_steps"] < 10:
        raise ValueError("invalid train.monitor_every / keep_best / benchmark_steps")
    d = cfg["data"]
    if d["crop_seconds"] <= 0.5 or d["bronze_max_seconds"] <= 0.5:
        raise ValueError("data.crop_seconds / bronze_max_seconds too small")
    if fr["synthetic"] > 0 and cfg["synthetic"]["factory"] == "models.data:build_synthgen":
        excl = set(cfg["synthetic"]["kwargs"].get("exclude_folds", [4]))
        if not set(cfg["monitor"]["folds"]) <= excl:
            raise ValueError("synthetic.kwargs.exclude_folds must contain every monitor fold "
                             "(the mixer would otherwise use held-out clean speech)")
    if fr["val_natural"] > 0:
        overlap = set(cfg["monitor"]["folds"]) - set(d["val_holdout_folds"])
        if overlap:
            raise ValueError(f"monitor folds {sorted(overlap)} are used for training (not in data.val_holdout_folds)")
    return cfg
