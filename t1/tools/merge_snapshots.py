"""Average each model's best and second-best checkpoint caches: work/caches/<m>_ens2/{val,blind}.npz."""
import sys
import numpy as np
from common.contract import WORK, save_cache, load_cache

for m in sys.argv[1:]:
    a_dir, b_dir, out = WORK / "caches" / m, WORK / "caches" / f"{m}_snap2", WORK / "caches" / f"{m}_ens2"
    if not (b_dir / "blind.npz").exists():
        print(f"skip {m}: no snapshot cache"); continue
    for split in ("val", "blind"):
        a, b = load_cache(a_dir / f"{split}.npz"), load_cache(b_dir / f"{split}.npz")
        merged = {}
        for k, x in a.items():
            y = b.get(k)
            if y is None:
                merged[k] = x; continue
            n = min(x.shape[1], y.shape[1])
            merged[k] = np.concatenate([(x[:, :n] + y[:, :n]) / 2, x[:, n:]], axis=1)
        save_cache(out / f"{split}.npz", merged)
    print(f"merged {m} -> {out.name}")
