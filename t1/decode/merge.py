"""Combine template-matched bank events with neural decodes.

bank record (work/bank/{val,blind}_events.jsonl line):
  {"clip_id", "synthetic_score": max p over events (0 if none),
   "events": [{"onset", "offset", "tag", "category", "ncc", "p", "snippet"}]}  (p >= 0.05 stored)

Routes
  bank     synthetic_score >= synth_thr: bank events with p >= bank_p_thr (ncc >= ncc_thr when
           p is absent), plus neural synthetic-prior events with score >= neural_conf that stay
           `margin` away from the bank events (total <= max_events, gaps >= gap).
  prior    prior_thr <= synthetic_score < synth_thr (or bank route with no surviving bank event):
           neural synthetic decode under the synthetic prior, score >= prior_conf.
  natural  everything else: the neural natural decode.
Never empty: falls back to the natural decode (which itself never returns empty).
"""
from __future__ import annotations

from decode.decoders import synthetic_prior

DEFAULT = dict(bank_on=True, synth_thr=0.5, bank_p_thr=0.5, ncc_thr=0.6, margin=0.2, neural_conf=0.7,
               prior_thr=1.01, prior_conf=0.0, gap=0.2, max_events=5, prior_mode="drop")


def _get(neural_events, key):
    if isinstance(neural_events, dict):
        v = neural_events.get(key)
        if v is None and key == "synthetic":
            v = neural_events.get("natural")
    else:
        v = neural_events
    return v() if callable(v) else (v or [])


def bank_conf(e):
    return float(e["p"]) if e.get("p") is not None else float(e.get("ncc", 0.0))


def route(bank_record, cfg):
    c = {**DEFAULT, **(cfg or {})}
    if not c["bank_on"] or not bank_record:
        return "natural"
    s = float(bank_record.get("synthetic_score", 0.0) or 0.0)
    if s >= c["synth_thr"]:
        return "bank"
    if s >= c["prior_thr"]:
        return "prior"
    return "natural"


def bank_events(bank_record, cfg):
    c = {**DEFAULT, **(cfg or {})}
    out = []
    for e in bank_record.get("events", []) if bank_record else []:
        thr = c["bank_p_thr"] if e.get("p") is not None else c["ncc_thr"]
        if bank_conf(e) >= thr and float(e["offset"]) > float(e["onset"]):
            out.append((float(e["onset"]), float(e["offset"]), e.get("category"), bank_conf(e)))
    # bank events are documented as non-overlapping; enforce the synthetic structure anyway
    return synthetic_prior(out, c["gap"], c["max_events"], mode="drop")


def combine(neural_events, bank_record, cfg):
    """neural_events: list, or {"natural": list|callable, "synthetic": list|callable} (lazy).

    The "synthetic" decode must be raw (no prior applied); the prior is applied here.
    Returns (events [(onset, offset, category, score)], route).
    """
    c = {**DEFAULT, **(cfg or {})}
    r = route(bank_record, c)
    out = []
    memo = {}
    if isinstance(neural_events, dict):  # evaluate each lazy decode at most once
        raw = neural_events
        neural_events = {k: (lambda k=k: memo[k] if k in memo else memo.setdefault(k, _get(raw, k)))
                         for k in ("natural", "synthetic")}
    if r == "bank":
        bank = bank_events(bank_record, c)
        if bank:
            room = int(c["max_events"]) - len(bank)
            neural = [e for e in _get(neural_events, "synthetic") if e[3] >= c["neural_conf"]]
            extra = synthetic_prior(neural, c["gap"], room, exclude=bank, margin=c["margin"],
                                    mode=c["prior_mode"]) if room > 0 and neural else []
            out = sorted(bank + extra, key=lambda e: (e[0], e[1]))
        else:
            r = "prior"
    if r == "prior":
        neural = [e for e in _get(neural_events, "synthetic") if e[3] >= c["prior_conf"]]
        out = synthetic_prior(neural, c["gap"], int(c["max_events"]), mode=c["prior_mode"])
    if not out:
        if r != "natural":
            r = r + "->natural"
        out = list(_get(neural_events, "natural"))
    return out, r
