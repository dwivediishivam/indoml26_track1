#!/usr/bin/env python3
"""Tune fusion weights, decoder family/parameters and merge thresholds on the validation set.

Objective: the exact combined score (event F1 + per-clip macro Dice, common/metric.py semantics,
re-implemented in decode/fastmetric.py) on the FULL validation set, natural and synthetic
pooled exactly as the leaderboard pools them.

Overfitting report (--cv):
  select  (default) for each fold f, pick the best trial by its score on the other 4 folds
          (from the whole trial pool) and report its score on fold f; mean and per-fold.
  retune  run an independent search per fold on 4 folds (cv-frac of the budget), score fold 5.
  none    skip.

Usage (from t1_final/):
  python -m decode.tune --name legacy_bank --legacy --no-models --minutes 30
  python -m decode.tune --name step2 --legacy --models beats atst --minutes 45 --init work/tuned/legacy_bank.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import multiprocessing as mp
import random
import time
import traceback
import zlib

import numpy as np

from common.contract import WORK, folds as load_folds, load_validation
from decode import fusion, pipeline
from decode.fastmetric import aggregate, clip_stats
from decode.legacy import BEST as LEGACY_BEST

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    optuna = None

FAMILIES = ("legacy", "hysteresis_per_category", "csebb", "peaks")
SHORT = {"legacy": "lg", "hysteresis_per_category": "hy", "csebb": "cs", "peaks": "pk"}
N_FOLD = 5
DATA: dict = {}  # filled in the parent before the pool forks (or by the worker initializer)


# ============================================================================= suggesters
def _snap(v, lo, hi, step):
    v = min(max(v, lo), hi)
    if step:
        v = lo + round((v - lo) / step) * step
        v = min(max(v, lo), hi)
    return round(float(v), 6)


class Suggester:
    """Define-by-run interface; records flat params and their distributions."""

    def __init__(self):
        self.params, self.dists = {}, {}

    def float(self, name, lo, hi, step, default):
        self.dists[name] = ("float", lo, hi, step)
        v = self.params[name] = self._float(name, lo, hi, step, default)
        return v

    def int(self, name, lo, hi, default):
        self.dists[name] = ("int", lo, hi, None)
        v = self.params[name] = int(self._int(name, lo, hi, default))
        return v

    def cat(self, name, choices, default):
        choices = list(choices)
        self.dists[name] = ("cat", tuple(choices), None, None)
        v = self.params[name] = self._cat(name, choices, default)
        return v


class FixedSuggester(Suggester):
    """Uses given values, else the space default (warm starts)."""

    def __init__(self, values):
        super().__init__(); self.values = dict(values)

    def _float(self, n, lo, hi, step, d):
        return float(self.values.get(n, d))

    def _int(self, n, lo, hi, d):
        return int(self.values.get(n, d))

    def _cat(self, n, choices, d):
        v = self.values.get(n, d)
        return v if v in choices else d


class RandomSuggester(Suggester):
    """Uniform random; with `base`, keeps base values except names in `mutate` (perturbed)."""

    def __init__(self, rng, base=None, mutate=(), scale=0.15):
        super().__init__(); self.rng, self.base, self.mutate, self.scale = rng, base, set(mutate), scale

    def _float(self, n, lo, hi, step, d):
        if self.base is not None and n in self.base:
            if n not in self.mutate:
                return self.base[n]
            return _snap(self.base[n] + self.rng.gauss(0, self.scale * (hi - lo)), lo, hi, step)
        return _snap(self.rng.uniform(lo, hi), lo, hi, step)

    def _int(self, n, lo, hi, d):
        if self.base is not None and n in self.base:
            if n not in self.mutate:
                return self.base[n]
            return int(min(hi, max(lo, round(self.base[n] + self.rng.gauss(0, max(1.0, self.scale * (hi - lo)))))))
        return self.rng.randint(lo, hi)

    def _cat(self, n, choices, d):
        if self.base is not None and n in self.base:
            if n not in self.mutate and self.base[n] in choices:
                return self.base[n]
            other = [c for c in choices if c != self.base[n]] or choices
            return self.rng.choice(other)
        return self.rng.choice(choices)


class OptunaSuggester(Suggester):
    def __init__(self, trial):
        super().__init__(); self.t = trial

    def _float(self, n, lo, hi, step, d):
        return float(self.t.suggest_float(n, lo, hi, step=step))

    def _int(self, n, lo, hi, d):
        return self.t.suggest_int(n, lo, hi)

    def _cat(self, n, choices, d):
        return self.t.suggest_categorical(n, choices)


def _optuna_dists(dists):
    out = {}
    for n, (kind, a, b, step) in dists.items():
        if kind == "float":
            out[n] = optuna.distributions.FloatDistribution(a, b, step=step)
        elif kind == "int":
            out[n] = optuna.distributions.IntDistribution(a, b)
        else:
            out[n] = optuna.distributions.CategoricalDistribution(a)
    return out


# ============================================================================= search space
def _common_track(s, p, opts, out):
    out["use_union"] = s.cat(p + "use_union", [True, False], True)
    out["use_cat"] = s.cat(p + "use_cat", [True, False], True)
    out["union_cover"] = s.float(p + "union_cover", 0.2, 1.05, 0.05, 0.6)
    out["dedup_iou"] = s.float(p + "dedup_iou", 0.5, 0.95, 0.05, 0.8)
    out["on_bias"] = s.float(p + "on_bias", -0.1, 0.1, 0.01, 0.0)
    out["off_bias"] = s.float(p + "off_bias", -0.1, 0.1, 0.01, 0.0)
    out["union_on_bias"] = s.float(p + "u_on_bias", -0.06, 0.06, 0.01, 0.0)
    out["union_off_bias"] = s.float(p + "u_off_bias", -0.06, 0.06, 0.01, 0.0)
    if opts.get("percat"):
        out["cat_on_bias"] = [s.float(f"{p}con{k}", -0.08, 0.08, 0.01, 0.0) for k in range(7)]
        out["cat_off_bias"] = [s.float(f"{p}coff{k}", -0.08, 0.08, 0.01, 0.0) for k in range(7)]
    return out


def space_legacy(s, p, opts):
    hi = s.float(p + "hi", 0.3, 0.97, 0.01, LEGACY_BEST["hi"])
    out = dict(hi=hi, lo=round(max(0.02, hi - s.float(p + "lo_delta", 0.0, 0.4, 0.01, 0.0)), 4),
               smooth=s.cat(p + "smooth", [1, 3], 1), median=s.cat(p + "median", [1, 3, 5, 7], 3),
               min_dur=s.float(p + "min_dur", 0.0, 0.2, 0.02, 0.0),
               merge_gap=s.float(p + "merge_gap", 0.0, 0.3, 0.02, 0.08),
               pad_on=s.float(p + "pad_on", -0.08, 0.08, 0.01, -0.02),
               pad_off=s.float(p + "pad_off", -0.08, 0.08, 0.01, -0.02),
               split_drop=s.float(p + "split_drop", 0.0, 0.4, 0.05, 0.0),
               track_thr=s.float(p + "track_thr", 0.1, 1.1, 0.05, 0.25),
               track_min=s.float(p + "track_min", 0.04, 0.4, 0.02, 0.24),
               track_cover=s.float(p + "track_cover", 0.2, 0.9, 0.05, 0.45),
               track_iou=s.float(p + "track_iou", 0.2, 0.9, 0.05, 0.45),
               use_cat=s.cat(p + "use_cat", [1.0, 0.0], 1.0), use_tag=0.0,
               bref=s.int(p + "bref", 0, 8, 5), bref_thr=s.float(p + "bref_thr", 0.05, 0.5, 0.05, 0.15))
    if out["split_drop"] > 0:
        out["split_min"] = s.float(p + "split_min", 0.1, 0.5, 0.05, 0.3)
    return out


def space_hysteresis(s, p, opts):
    out = dict(median=s.cat(p + "median", [1, 3, 5, 7], 3),
               hi=s.float(p + "hi", 0.2, 0.95, 0.01, 0.6), lo_delta=s.float(p + "lo_delta", 0.0, 0.4, 0.01, 0.15),
               cat_hi=s.float(p + "cat_hi", 0.2, 0.95, 0.01, 0.5),
               cat_lo_delta=s.float(p + "cat_lo_delta", 0.0, 0.4, 0.01, 0.15),
               gate=s.float(p + "gate", 0.0, 0.5, 0.05, 0.0), gap=s.float(p + "gap", 0.0, 0.3, 0.02, 0.06),
               min_dur=s.float(p + "min_dur", 0.0, 0.3, 0.02, 0.06))
    if opts.get("percat"):
        out["cat_hi_delta"] = [s.float(f"{p}chd{k}", -0.2, 0.2, 0.02, 0.0) for k in range(7)]
    return _common_track(s, p, opts, out)


def space_csebb(s, p, opts):
    out = dict(tau=s.cat(p + "tau", [0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32], 0.12),
               median=s.cat(p + "median", [1, 3, 5], 1), pad=s.cat(p + "pad", ["zero", "edge"], "zero"),
               min_step=s.float(p + "min_step", 0.01, 0.3, 0.01, 0.05),
               seg_thr=s.float(p + "seg_thr", 0.05, 0.7, 0.01, 0.3),
               gamma_abs=s.float(p + "gamma_abs", 0.0, 0.6, 0.02, 0.2),
               gamma_rel=s.float(p + "gamma_rel", 0.0, 0.9, 0.05, 0.0),
               thr=s.float(p + "thr", 0.2, 0.95, 0.01, 0.5), cat_thr=s.float(p + "cat_thr", 0.2, 0.95, 0.01, 0.5),
               min_dur=s.float(p + "min_dur", 0.0, 0.3, 0.02, 0.04), refine=s.cat(p + "refine", [True, False], True))
    return _common_track(s, p, opts, out)


def space_peaks(s, p, opts):
    out = dict(on_thr=s.float(p + "on_thr", 0.05, 0.8, 0.01, 0.2), off_thr=s.float(p + "off_thr", 0.05, 0.8, 0.01, 0.2),
               pres_thr=s.float(p + "pres_thr", 0.1, 0.9, 0.01, 0.4),
               score_thr=s.float(p + "score_thr", 0.0, 0.8, 0.02, 0.0),
               alpha=s.float(p + "alpha", 0.0, 1.0, 0.05, 0.5), max_next=s.cat(p + "max_next", [1, 2, 3], 2),
               min_dur=s.float(p + "min_dur", 0.0, 0.2, 0.02, 0.04),
               max_dur=s.cat(p + "max_dur", [3.0, 6.0, 12.0, 30.0], 30.0),
               nms_iou=s.float(p + "nms_iou", 0.2, 0.95, 0.05, 0.5),
               edge_pres=s.float(p + "edge_pres", 0.2, 1.05, 0.05, 0.6),
               on_shift=s.float(p + "on_shift", -0.06, 0.04, 0.005, 0.0),
               off_shift=s.float(p + "off_shift", -0.04, 0.06, 0.005, 0.0),
               bmedian=s.cat(p + "bmedian", [1, 3], 1), pmedian=s.cat(p + "pmedian", [1, 3, 5], 1),
               refine=s.cat(p + "refine", [True, False], True),
               use_union=s.cat(p + "use_union", [False, True], False), use_cat=True,
               fill_hi=s.float(p + "fill_hi", 0.3, 1.05, 0.05, 1.05),
               union_cover=s.float(p + "union_cover", 0.2, 1.05, 0.05, 0.6),
               dedup_iou=s.float(p + "dedup_iou", 0.5, 0.95, 0.05, 0.8))
    if out["fill_hi"] <= 1.0:
        out["fill_lo_delta"] = s.float(p + "fill_lo_delta", 0.0, 0.3, 0.02, 0.1)
        out["fill_cover"] = s.float(p + "fill_cover", 0.1, 0.9, 0.05, 0.5)
    return out


SPACES = {"legacy": space_legacy, "hysteresis_per_category": space_hysteresis, "csebb": space_csebb,
          "peaks": space_peaks}


def decoder_space(s, pre, opts):
    fams = opts["families"]
    fam = s.cat(pre + "family", fams, fams[0]) if len(fams) > 1 else fams[0]
    params = SPACES[fam](s, f"{pre}{SHORT[fam]}_", opts)
    post = {"edge_tol": s.float(pre + "edge_tol", 0.0, 0.2, 0.01, 0.0),
            "fallback_len": s.cat(pre + "fallback_len", [0.0, 0.5, 1.0, 1.5, 2.0], 1.0)}
    return {"family": fam, "params": params, "post": post}


def merge_space(s, opts):
    if not opts["bank"]:
        return {"bank_on": False}
    m = {"bank_on": s.cat("bank_on", [True, False], True), "gap": 0.2, "max_events": 5}
    if not m["bank_on"]:
        return m
    m["synth_thr"] = s.float("synth_thr", 0.05, 1.0, 0.01, 0.5)
    m["bank_p_thr"] = s.float("bank_p_thr", 0.05, 1.0, 0.01, 0.5)
    if not opts["bank_has_p"]:
        m["ncc_thr"] = s.float("ncc_thr", 0.0, 1.0, 0.01, 0.6)
    m["margin"] = s.float("margin", 0.0, 0.4, 0.02, 0.2)
    m["neural_conf"] = s.float("neural_conf", 0.2, 1.01, 0.01, 0.7)
    m["prior_mode"] = s.cat("prior_mode", ["drop", "merge"], "drop")
    if s.cat("prior_on", [False, True], False):
        m["prior_thr"] = round(m["synth_thr"] * s.float("prior_frac", 0.1, 1.0, 0.05, 0.5), 4)
        m["prior_conf"] = s.float("prior_conf", 0.0, 0.95, 0.05, 0.3)
    else:
        m["prior_thr"] = 1.01
    return m


def build(s, opts):
    names = [c["name"] for c in opts["caches"]]
    if len(names) > 1 and not opts.get("freeze_weights"):
        w = {n: s.float(f"w_{n}", 0.0, 1.0, 0.05, opts["init_weights"].get(n, 0.5)) for n in names}
        mode = s.cat("fusion_mode", ["prob", "logit"], "prob")
    else:
        w = {n: opts["init_weights"].get(n, 1.0) for n in names}
        mode = opts.get("fusion_mode", "prob")
    if sum(w.values()) <= 0:
        w = {n: 1.0 for n in names}
    tot = sum(w.values())
    cfg = {"fusion": {"weights": {k: round(v / tot, 6) for k, v in w.items()}, "mode": mode}}
    cfg["natural"] = decoder_space(s, "nat_", opts)
    same = s.cat("syn_same", [True, False], True) if opts["bank"] else True
    cfg["synthetic"] = None if same else decoder_space(s, "syn_", opts)
    cfg["merge"] = merge_space(s, opts)
    return cfg


def warm_starts(opts, inits):
    """Flat warm-start params: proven legacy decoder with the bank off / on, plus --init configs."""
    base = {"nat_family": "legacy", "syn_same": True, "fusion_mode": "prob"}
    base.update({f"w_{n}": w for n, w in opts["init_weights"].items()})
    out = [dict(base, bank_on=False)]
    if opts["bank"]:
        out.append(dict(base, bank_on=True))
    for fam in opts["families"]:
        if fam != "legacy":
            out.append(dict(base, nat_family=fam, bank_on=bool(opts["bank"])))
    out = [w for w in out if w["nat_family"] in opts["families"]] or [{}]
    return [dict(x) for x in inits] + out


# ============================================================================= data / eval
def load_data(spec, log=print):
    val = load_validation()
    fold_map = load_folds()
    ids = sorted(val)
    if spec.get("only_folds"):
        # Clips a candidate was trained on (val_natural) must not drive fusion/decoder selection.
        keep = set(spec["only_folds"])
        ids = [c for c in ids if fold_map.get(c) in keep]
    if spec.get("limit"):
        ids = ids[:: max(1, len(ids) // int(spec["limit"]))][: int(spec["limit"])]
    durs = pipeline.val_durations({k: val[k] for k in ids}, use_headers=spec.get("headers", True))
    caches = fusion.load_caches(spec["caches"], "val", ids=ids, log=log)
    bank = pipeline.load_bank(spec["bank"], log=log) if spec.get("bank") else {}
    fold = np.array([fold_map.get(c, N_FOLD) for c in ids], int)
    syn = np.array([val[c]["synthetic"] for c in ids], bool)
    return dict(ids=ids, durs=[durs[c] for c in ids], refs=[[(a, b) for a, b, *_ in val[c]["events"]] for c in ids],
                caches=caches, bank=bank, fold=fold, syn=syn, group=fold * 2 + syn.astype(int))


def _init_worker(spec):
    if not DATA:
        DATA.update(load_data(spec, log=lambda *a: None))


def evaluate(cfg, return_preds=False):
    D = DATA
    st = np.zeros((len(D["ids"]), 4), np.float64)
    routes, preds = {}, ({} if return_preds else None)
    for i, cid in enumerate(D["ids"]):
        spans, r = pipeline.predict_clip(D["caches"], cid, D["durs"][i], D["bank"].get(cid), cfg)
        st[i] = clip_stats(D["refs"][i], spans)
        routes[r] = routes.get(r, 0) + 1
        if preds is not None:
            preds[cid] = spans
    return st, routes, preds


def _task(num, cfg):
    t0 = time.time()
    try:
        st, routes, _ = evaluate(cfg)
        return num, st, routes, time.time() - t0, None
    except Exception:
        return num, None, None, time.time() - t0, traceback.format_exc()


G = (N_FOLD + 1) * 2  # (fold or no-fold) x (natural, synthetic)


def group_agg(st):
    agg = np.zeros((G, 4))
    np.add.at(agg, DATA["group"], st)
    return agg


def group_counts():
    if "counts" not in DATA:
        DATA["counts"] = np.bincount(DATA["group"], minlength=G)
    return DATA["counts"]


# T1_SYN_WEIGHT reweights synthetic groups so validation mimics the test mix (test 2080 syn / 3437 nat clips vs
# validation 695 / 1806 -> ~1.57). Pooled event F1 then values natural precision as it will be valued on test.
SYN_W = float(os.environ.get("T1_SYN_WEIGHT", "1.0"))
GROUP_W = np.where(np.arange(G) % 2 == 1, SYN_W, 1.0)


def score_groups(agg, mask):
    if SYN_W == 1.0:
        a = agg[mask].sum(0)
        tp = int(a[0])
        return aggregate(tp, int(a[1]) - tp, int(a[2]) - tp, float(a[3]), int(group_counts()[mask].sum()))
    w = GROUP_W[mask]
    a = (agg[mask] * w[:, None]).sum(0)
    tp = float(a[0])
    return aggregate(tp, float(a[1]) - tp, float(a[2]) - tp, float(a[3]), float((group_counts()[mask] * w).sum()))


def masks():
    g = np.arange(G)
    fold, syn = g // 2, g % 2
    return {"full": np.ones(G, bool), "natural": syn == 0, "synthetic": syn == 1,
            **{f"train{f}": fold != f for f in range(N_FOLD)}, **{f"test{f}": fold == f for f in range(N_FOLD)}}


def count_hist(n, top=8):
    n = np.asarray(n, int)
    return {(str(k) if k < top else f"{top}+"): int(((n == k) if k < top else (n >= top)).sum()) for k in range(0, top + 1)}


# ============================================================================= search
class Trial:
    __slots__ = ("num", "flat", "dists", "cfg", "agg", "score", "routes", "secs", "otrial", "origin")


class Search:
    def __init__(self, label, opts, args, objective_mask, warm, on_improve=None, log_file=None):
        self.label, self.opts, self.args, self.mask = label, opts, args, objective_mask
        self.queue = list(warm)
        self.grid = []
        self.on_improve, self.log_file = on_improve, log_file
        self.rng = random.Random(args.seed * 1000003 + zlib.crc32(label.encode()))
        self.trials, self.best, self.seen, self.num = [], None, set(), 0
        self.best_st = None
        self.study = None
        if optuna is not None and args.sampler in ("auto", "tpe"):
            try:
                sampler = optuna.samplers.TPESampler(seed=args.seed + zlib.crc32(label.encode()) % 997,
                                                     multivariate=True, group=True, constant_liar=True,
                                                     n_startup_trials=max(10, 2 * args.workers))
            except TypeError:
                sampler = optuna.samplers.TPESampler(seed=args.seed)
            self.study = optuna.create_study(direction="maximize", sampler=sampler)

    def _new(self, frac):
        """Next trial (flat params + cfg) or None if nothing new could be generated."""
        for _ in range(50):
            otrial = None
            if self.queue:
                origin, s = "warm", FixedSuggester(self.queue.pop(0))
            elif self.grid:
                origin, s = "grid", FixedSuggester(self.grid.pop(0))
            elif self.best is not None and (frac >= 1 - self.args.refine_frac or
                                            (self.study is None and self.rng.random() < 0.3 and frac > 0.15)):
                keys = [k for k in self.best.flat if self.best.dists[k][0] != "cat" or len(self.best.dists[k][1]) > 1]
                mut = self.rng.sample(keys, k=min(len(keys), self.rng.choice([1, 1, 2, 3])))
                scale = 0.25 * (1 - frac) + 0.04
                origin, s = "refine", RandomSuggester(self.rng, base=self.best.flat, mutate=mut, scale=scale)
            elif self.study is not None:
                otrial = self.study.ask()
                origin, s = "tpe", OptunaSuggester(otrial)
            else:
                origin, s = "random", RandomSuggester(self.rng)
            try:
                cfg = build(s, self.opts)
            except Exception:
                if otrial is not None:
                    self.study.tell(otrial, state=optuna.trial.TrialState.FAIL)
                raise
            key = json.dumps(s.params, sort_keys=True, default=str)
            if key in self.seen:
                if otrial is not None:
                    self.study.tell(otrial, state=optuna.trial.TrialState.PRUNED)
                continue
            self.seen.add(key)
            t = Trial()
            t.num, t.flat, t.dists, t.cfg, t.otrial, t.origin = self.num, s.params, s.dists, cfg, otrial, origin
            self.num += 1
            return t
        return None

    def _record(self, t, st, routes, secs, err):
        if err is not None or st is None:
            print(f"[{self.label}] trial {t.num} ({t.origin}) FAILED:\n{err}", flush=True)
            if t.otrial is not None:
                self.study.tell(t.otrial, state=optuna.trial.TrialState.FAIL)
            return
        t.agg, t.routes, t.secs = group_agg(st), routes, secs
        t.score = score_groups(t.agg, self.mask)["combined"]
        self.trials.append(t)
        if self.study is not None:
            try:
                if t.otrial is not None:
                    self.study.tell(t.otrial, t.score)
                else:
                    self.study.add_trial(optuna.trial.create_trial(params=t.flat, distributions=_optuna_dists(t.dists),
                                                                   value=t.score))
            except Exception:
                pass
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"label": self.label, "num": t.num, "origin": t.origin, "score": t.score,
                                    "secs": round(secs, 2), "routes": routes, "agg": t.agg.tolist(),
                                    "flat": t.flat}, default=str) + "\n")
        if self.best is None or t.score > self.best.score:
            self.best, self.best_st = t, st
            full = score_groups(t.agg, MASKS["full"])
            print(f"[{self.label}] NEW BEST trial {t.num} ({t.origin}, {secs:.1f}s) objective={t.score:.5f} "
                  f"full={full['combined']:.5f} F1={full['event_f1']:.4f} Dice={full['segment_dice']:.4f} "
                  f"nat={score_groups(t.agg, MASKS['natural'])['combined']:.4f} "
                  f"syn={score_groups(t.agg, MASKS['synthetic'])['combined']:.4f} "
                  f"fam={t.cfg['natural']['family']} routes={routes}", flush=True)
            if self.on_improve:
                self.on_improve(self)

    def run(self, pool, seconds):
        t0 = time.time()
        inflight = {}
        nworkers = self.args.workers if pool is not None else 1
        last_report = t0
        exhausted = False
        while True:
            frac = (time.time() - t0) / max(seconds, 1e-6)
            while not exhausted and frac < 1 and len(inflight) < nworkers:
                t = self._new(frac)
                if t is None:
                    exhausted = True  # nothing new could be generated
                    break
                if pool is None:
                    self._record(t, *_task(t.num, t.cfg)[1:])
                    frac = (time.time() - t0) / max(seconds, 1e-6)
                    continue
                inflight[t.num] = (t, pool.apply_async(_task, (t.num, t.cfg)))
            if not inflight:
                if frac >= 1 or exhausted:
                    break
                continue
            done = [k for k, (_, ar) in inflight.items() if ar.ready()]
            if not done:
                time.sleep(0.02)
                continue
            for k in done:
                t, ar = inflight.pop(k)
                _, st, routes, secs, err = ar.get()
                self._record(t, st, routes, secs, err)
            if time.time() - last_report > 60:
                last_report = time.time()
                print(f"[{self.label}] {len(self.trials)} trials, {time.time() - t0:.0f}s, "
                      f"best {self.best.score if self.best else float('nan'):.5f}", flush=True)
        return self.best


# ============================================================================= reports
MASKS: dict = {}


def report_trial(t):
    out = {k: score_groups(t.agg, MASKS[k]) for k in ("full", "natural", "synthetic")}
    out["per_fold"] = [score_groups(t.agg, MASKS[f"test{f}"])["combined"] for f in range(N_FOLD)]
    out["routes"] = t.routes
    return out


def cv_select(trials):
    folds = []
    for f in range(N_FOLD):
        tr, te = MASKS[f"train{f}"], MASKS[f"test{f}"]
        if group_counts()[te].sum() == 0:
            continue
        b = max(trials, key=lambda t: score_groups(t.agg, tr)["combined"])
        folds.append({"fold": f, "trial": b.num, "train": score_groups(b.agg, tr)["combined"],
                      "test": score_groups(b.agg, te)["combined"], "test_detail": score_groups(b.agg, te)})
    tests = [x["test"] for x in folds]
    return {"mode": "select", "mean": float(np.mean(tests)) if tests else None,
            "std": float(np.std(tests)) if tests else None, "per_fold": folds}


def parse_kv(items):
    out = {}
    for it in items or []:
        name, _, rest = it.partition("=")
        if not rest:
            raise SystemExit(f"expected NAME=VAL[,BLIND], got {it!r}")
        val, _, blind = rest.partition(",")
        out[name] = (val, blind or None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output name -> work/tuned/<name>.json")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model caches work/caches/<m>/{val,blind}.npz (default: all discovered)")
    ap.add_argument("--no-models", action="store_true", help="do not use work/caches models")
    ap.add_argument("--cache", action="append", help="extra contract cache NAME=VAL_NPZ[,BLIND_NPZ]")
    ap.add_argument("--legacy", action="store_true", help="include ai2 legacy caches s9000 + s1600")
    ap.add_argument("--legacy-cache", action="append", help="legacy cache NAME=VAL_NPZ[,BLIND_NPZ]")
    ap.add_argument("--weights", default="", help="initial weights NAME=W,NAME=W (default legacy 0.8/0.2)")
    ap.add_argument("--freeze-weights", action="store_true", help="do not search fusion weights")
    ap.add_argument("--fusion-mode", default="prob", choices=["prob", "logit"], help="mode when weights are frozen")
    ap.add_argument("--weight-grid", type=float, default=0.0, help="simplex grid step for an initial weight sweep")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--percat", action="store_true", help="search per-category biases / thresholds")
    ap.add_argument("--bank", default=pipeline.BANK_DEFAULT["val"])
    ap.add_argument("--bank-blind", default=pipeline.BANK_DEFAULT["blind"])
    ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--start-method", default="fork" if os.name == "posix" else "spawn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampler", default="auto", choices=["auto", "tpe", "random"])
    ap.add_argument("--refine-frac", type=float, default=0.3, help="final fraction of time spent on coordinate refinement")
    ap.add_argument("--cv", default="select", choices=["select", "retune", "none"])
    ap.add_argument("--cv-frac", type=float, default=0.4, help="budget fraction for --cv retune")
    ap.add_argument("--init", action="append", default=[], help="warm start from a tuned config JSON")
    ap.add_argument("--limit", type=int, default=0, help="debug: use a subset of N validation clips")
    ap.add_argument("--only-folds", type=int, nargs="+", default=None,
                    help="tune/score only validation clips in these folds, e.g. --only-folds 4 when any cache "
                         "was trained on val_natural folds 0-3 (use with --cv none)")
    ap.add_argument("--no-headers", action="store_true", help="use metadata durations instead of WAV headers")
    ap.add_argument("--no-verify", action="store_true", help="skip the common.metric cross-check of the best")
    args = ap.parse_args(argv)

    # ---------------------------------------------------------------- caches
    specs = []
    if args.legacy:
        specs += [fusion.legacy_spec("s9000"), fusion.legacy_spec("s1600")]
    for n, (v, b) in parse_kv(args.legacy_cache).items():
        specs.append(fusion.legacy_spec(n, v, b))
    if not args.no_models:
        for m in (args.models if args.models is not None else fusion.discover_models()):
            specs.append(fusion.model_spec(m))
    for n, (v, b) in parse_kv(args.cache).items():
        vp = Path(v)
        guess = str(vp.with_name(vp.name.replace("val", "blind", 1))) if "val" in vp.name else None
        specs.append({"name": n, "format": "contract", "val": v, "blind": b or guess})
    if not specs:
        raise SystemExit("no caches: pass --legacy and/or --models/--cache")
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate cache names {names}")
    init_w = {n: fusion.LEGACY_DEFAULT.get(n, 1.0 / len(names)) for n in names}
    for kv in filter(None, args.weights.split(",")):
        k, _, w = kv.partition("=")
        init_w[k] = float(w)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    if "legacy" in families:
        families = ["legacy"] + [f for f in families if f != "legacy"]
    bad = set(families) - set(FAMILIES)
    if bad:
        raise SystemExit(f"unknown families {bad}")

    spec = {"caches": specs, "bank": None if args.no_bank else args.bank, "limit": args.limit,
            "only_folds": args.only_folds,
            "headers": not args.no_headers}
    t_load = time.time()
    DATA.update(load_data(spec))
    MASKS.update(masks())
    bank = DATA["bank"]
    has_p = any(e.get("p") is not None for r in bank.values() for e in r.get("events", []))
    print(f"[tune] {len(DATA['ids'])} clips ({int(DATA['syn'].sum())} synthetic), caches={names}, "
          f"bank={len(bank)} records (p={'yes' if has_p else 'no'}), loaded in {time.time() - t_load:.0f}s, "
          f"sampler={'optuna-tpe' if optuna is not None and args.sampler != 'random' else 'random+coordinate'}",
          flush=True)
    opts = {"caches": specs, "families": families, "bank": bool(bank), "bank_has_p": has_p,
            "percat": args.percat, "init_weights": init_w, "freeze_weights": args.freeze_weights,
            "fusion_mode": args.fusion_mode}
    inits = []
    for p in args.init:
        c = pipeline.load_config(p)
        if c.get("flat_params"):
            inits.append(c["flat_params"])
    warm = warm_starts(opts, inits)

    out_path = WORK / "tuned" / f"{args.name}.json"
    log_path = WORK / "tuned" / f"{args.name}.trials.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    state = {"cv": None}
    counts = group_counts()

    def save(search):
        t = search.best
        cfg = {"name": args.name, **t.cfg, "caches": specs,
               "bank": {"val": spec["bank"], "blind": None if args.no_bank else args.bank_blind},
               "flat_params": t.flat, "metrics": report_trial(t),
               "val_counts": {"pred_hist_natural": count_hist(search.best_st[~DATA["syn"], 1]),
                              "pred_hist_synthetic": count_hist(search.best_st[DATA["syn"], 1]),
                              "ref_hist_natural": count_hist(search.best_st[~DATA["syn"], 2]),
                              "ref_hist_synthetic": count_hist(search.best_st[DATA["syn"], 2]),
                              "pred_events": int(search.best_st[:, 1].sum()),
                              "ref_events": int(search.best_st[:, 2].sum()), "clips": len(DATA["ids"])},
               "cv": state["cv"],
               "search": {"label": search.label, "trial": t.num, "origin": t.origin, "trials": len(search.trials),
                          "elapsed_s": round(time.time() - t_start, 1), "objective": t.score,
                          "limit": args.limit, "families": families, "seed": args.seed}}
        pipeline.atomic_json(out_path, cfg)

    pool = None
    if args.workers > 1:
        ctx = mp.get_context(args.start_method)
        pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=(spec,))
    try:
        budget = args.minutes * 60.0
        if args.cv == "retune":
            per = budget * args.cv_frac / N_FOLD
            folds_out = []
            for f in range(N_FOLD):
                if counts[MASKS[f"test{f}"]].sum() == 0:
                    continue
                s = Search(f"cv{f}", opts, args, MASKS[f"train{f}"], warm, log_file=log_path)
                b = s.run(pool, per)
                if b is not None:
                    folds_out.append({"fold": f, "trial": b.num, "train": b.score,
                                      "test": score_groups(b.agg, MASKS[f"test{f}"])["combined"],
                                      "test_detail": score_groups(b.agg, MASKS[f"test{f}"])})
                    print(f"[cv{f}] train={b.score:.5f} test={folds_out[-1]['test']:.5f}", flush=True)
            tests = [x["test"] for x in folds_out]
            state["cv"] = {"mode": "retune", "mean": float(np.mean(tests)) if tests else None,
                           "std": float(np.std(tests)) if tests else None, "per_fold": folds_out}
            budget -= time.time() - t_start
        main_search = Search("full", opts, args, MASKS["full"], warm, on_improve=save, log_file=log_path)
        if args.weight_grid > 0 and len(names) > 1 and not args.freeze_weights:
            main_search.grid = [dict(w0, **{f"w_{n}": v for n, v in zip(names, pt)})
                                for pt in simplex_grid(len(names), args.weight_grid) for w0 in warm[:1]]
        best = main_search.run(pool, max(budget, 30.0))
    finally:
        if pool is not None:
            pool.close(); pool.join()
    if best is None:
        raise SystemExit("no successful trial")
    if args.cv == "select":
        state["cv"] = cv_select(main_search.trials)
    save(main_search)

    # ---------------------------------------------------------------- final report
    rep = report_trial(best)
    print("\n================ BEST (full validation objective) ================")
    print(f"config: {out_path}\ntrials: {len(main_search.trials)}  best trial {best.num} ({best.origin})")
    for k in ("full", "natural", "synthetic"):
        r = rep[k]
        print(f"  {k:9s} combined={r['combined']:.5f} F1={r['event_f1']:.4f} Dice={r['segment_dice']:.4f} "
              f"tp={r['tp']} fp={r['fp']} fn={r['fn']} clips={r['clips']}")
    print(f"  in-sample per-fold combined: {[round(x, 4) for x in rep['per_fold']]}")
    print(f"  routes: {rep['routes']}")
    cv = state["cv"]
    if cv and cv.get("per_fold"):
        print(f"  {cv['mode']}-CV (tune on 4 folds, score the 5th): mean={cv['mean']:.5f} std={cv['std']:.5f} "
              f"per-fold={[round(x['test'], 4) for x in cv['per_fold']]}")
        print(f"  optimism (in-sample per-fold mean - CV mean): {np.mean(rep['per_fold']) - cv['mean']:+.5f}")
    print(f"  decoder: {best.cfg['natural']['family']}  fusion: {best.cfg['fusion']}  merge: {best.cfg['merge']}")
    if not args.no_verify:
        from common.metric import score as exact_score
        _, _, preds = evaluate(best.cfg, return_preds=True)
        refs = {c: r for c, r in zip(DATA["ids"], DATA["refs"])}
        ex = exact_score(refs, preds)
        ok = abs(ex["combined"] - rep["full"]["combined"]) < 1e-9
        print(f"  verify common.metric.score: combined={ex['combined']:.6f} (fast {rep['full']['combined']:.6f}) "
              f"{'OK' if ok else 'MISMATCH'}")
        cfg = pipeline.load_config(out_path)
        cfg["metrics"]["verified_exact"] = ex
        pipeline.atomic_json(out_path, cfg)


def simplex_grid(m, step):
    k = int(round(1 / step))
    out = []

    def rec(prefix, left):
        if len(prefix) == m - 1:
            out.append(prefix + [left]); return
        for i in range(left + 1):
            rec(prefix + [i], left - i)
    rec([], k)
    return [[round(i / k, 6) for i in pt] for pt in out if any(pt)]


if __name__ == "__main__":
    main()
