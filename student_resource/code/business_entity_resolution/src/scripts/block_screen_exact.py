"""Screen exact-key blockers on the 50k screening S1 set (full target index)."""
import json
import sys

import polars as pl

from ber.blocking.base import eval_candidates
from ber.blocking.exact import KEYS, exact_block
from ber.data import load_gt_pairs, load_queries, load_targets
from ber.experiment import Experiment
from ber.paths import CACHE
from ber.splits import load_folds

Q, T, G = load_queries("train"), load_targets("train"), load_gt_pairs()
n2 = int((T["src"] == 2).sum())
qs = load_folds().filter("screen")["idx"].to_numpy()
Qs = Q.filter(pl.col("q").is_in(qs))
max_block = int(sys.argv[1]) if len(sys.argv) > 1 else 50
(CACHE / "cands").mkdir(exist_ok=True)
for key in KEYS:
    ex = Experiment(f"block_exact_{key}", "blocking", {"blocker": "exact", "key": key, "max_block": max_block},
                    f"exact key {key}", tier="screen")
    c = exact_block(Qs, T, key, max_block)
    c.write_parquet(CACHE / "cands" / f"screen_exact_{key}.parquet")
    m = eval_candidates(c, G, qs, n2)
    ex.log(**m)
    ex.finish()
    print(key, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
