"""Print per-candidate training progress and monitor scores from RUN_ROOT/<cand>/history.jsonl."""
import json
import os
import sys

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/indoml-work/runs_t1f")
for c in (sys.argv[2:] or ("m1v_atst_val", "m2_beats", "m4_breath", "m1_atst", "m3_wavlm_large", "m5_w2vbert")):
    h = os.path.join(root, c, "history.jsonl")
    if not os.path.exists(h):
        print(f"{c:15s} no history yet")
        continue
    mons, last = [], {}
    for line in open(h):
        d = json.loads(line)
        if d.get("state") == "monitor":
            mons.append(d)
        if "step" in d:
            last = d
    s = f"{c:15s} step {last.get('step', '?')}/{last.get('planned', '?')} stage {last.get('stage', '?')} [{last.get('state', '?')}]"
    if mons:
        cur, best = mons[-1], max(mons, key=lambda d: d["metrics"]["combined"])
        m = cur["metrics"]
        s += f" | last {cur['weights']} {m['combined']:.4f} (nat {m['natural']:.3f}, syn {m['synthetic']:.3f})"
        s += f" | best {best['weights']}@{best['step']} {best['metrics']['combined']:.4f}"
    print(s)
