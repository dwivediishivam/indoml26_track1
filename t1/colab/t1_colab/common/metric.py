"""Faithful Track 1 metric implementation used for local model selection."""
from __future__ import annotations

from collections import defaultdict


def match_events(reference, predicted):
    candidates = []
    for ri, (rs, re) in enumerate(reference):
        tolerance = max(.20 * (re - rs), .05)
        for pi, (ps, pe) in enumerate(predicted):
            if abs(ps - rs) <= tolerance and abs(pe - re) <= tolerance:
                candidates.append((abs(ps-rs) + abs(pe-re), ri, pi))
    used_r, used_p = set(), set()
    for _, ri, pi in sorted(candidates):
        if ri not in used_r and pi not in used_p:
            used_r.add(ri); used_p.add(pi)
    return len(used_r), len(predicted)-len(used_p), len(reference)-len(used_r)


def dice(reference, predicted, frame=.01):
    events = reference + predicted
    if not events:
        return 1.
    n = int((max(end for _, end in events) + .5) / frame) + 1
    ref, pred = bytearray(n), bytearray(n)
    for target, spans in ((ref, reference), (pred, predicted)):
        for start, end in spans:
            for i in range(max(0, int(start/frame)), min(n, int(end/frame)+1)):
                target[i] = 1
    total = sum(ref) + sum(pred)
    return 1. if not total else 2 * sum(a & b for a, b in zip(ref, pred)) / total


def score(reference: dict[str, list[tuple[float, float]]], predicted: dict[str, list[tuple[float, float]]]):
    tp = fp = fn = 0; dice_scores = []
    for clip_id, spans in reference.items():
        a, b, c = match_events(spans, predicted.get(clip_id, []))
        tp += a; fp += b; fn += c
        dice_scores.append(dice(spans, predicted.get(clip_id, [])))
    for clip_id, spans in predicted.items():
        if clip_id not in reference:
            fp += len(spans)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2*precision*recall / (precision+recall) if precision+recall else 0.
    segment_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.
    return {"event_f1": f1, "segment_dice": segment_dice, "combined": f1 + segment_dice,
            "tp": tp, "fp": fp, "fn": fn}
