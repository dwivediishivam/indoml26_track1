"""Write work/folds.json: 5 folds grouped by district, balanced on synthetic and event counts."""
import json, random, collections
from common.contract import load_validation, WORK
val = load_validation()
groups = collections.defaultdict(list)
for k, v in val.items(): groups[(v["district"], v["synthetic"])].append(k)
order = sorted(groups, key=lambda g: -sum(len(val[k]["events"]) for k in groups[g]))
random.Random(0).shuffle(order); order.sort(key=lambda g: -len(groups[g]))
load = {(f, s): 0 for f in range(5) for s in (True, False)}; assign = {}
for g in order:
    f = min(range(5), key=lambda f: load[(f, g[1])])
    for k in groups[g]: assign[k] = f
    load[(f, g[1])] += sum(len(val[k]["events"]) for k in groups[g])
WORK.mkdir(exist_ok=True); (WORK / "folds.json").write_text(json.dumps(assign))
print({f: (sum(1 for k in assign if assign[k] == f and val[k]["synthetic"]), sum(1 for k in assign if assign[k] == f and not val[k]["synthetic"])) for f in range(5)})
