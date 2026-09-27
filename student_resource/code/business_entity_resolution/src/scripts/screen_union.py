"""Union + pruning screen: recall/size of the full union, leave-one-blocker-out
ablation, and the pruner's recall@N curve on the 50k screening S1."""
import json

import numpy as np
import polars as pl

from ber import candidates as cand
from ber.blocking.base import eval_candidates
from ber.data import load_gt_pairs, target_meta
from ber.experiment import Experiment
from ber.paths import CACHE
from ber.splits import load_folds

G = load_gt_pairs()
n2 = int((target_meta("train")["src"] == 2).sum())
folds = load_folds()
qs = folds.filter("screen")["idx"].to_numpy()
qtr = np.random.default_rng(1).choice(folds.filter(pl.col("fold") != 0)["idx"].to_numpy(), 150_000, replace=False)


def show(tag, m):
    print(tag, json.dumps({k: round(v, 4) for k, v in m.items() if k in
                           ("pair_recall", "entity_full_coverage", "cand_mean", "cand_p95", "cand_max")}), flush=True)


U = cand.retrieval_frame("train", qs=np.concatenate([qs, qtr]))
U = U.join(G.with_columns(y=pl.lit(1, pl.Int8)), on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
Us = U.filter(pl.col("q").is_in(qs))
m = eval_candidates(Us, G, qs, n2); show("UNION all", m)
ex = Experiment("block_union_all", "blocking", {"specs": cand.RETR_SPECS}, "union of all 7 blockers", "screen")
ex.log(**m); ex.finish()
for s in cand.RETR_SPECS:
    n = s["name"]
    others = Us.filter(pl.any_horizontal([pl.col(f"{o['name']}_score").is_not_null() for o in cand.RETR_SPECS if o["name"] != n]))
    m = eval_candidates(others, G, qs, n2); show(f"UNION minus {n}", m)
    ex = Experiment(f"block_union_minus_{n}", "blocking", {"dropped": n}, f"union without {n}", "screen")
    ex.log(**m); ex.finish()

Utr = U.filter(pl.col("q").is_in(qtr))
b = cand.train_pruner(Utr, Utr["y"].to_numpy())
b.save_model(str(CACHE / "pruner_screen.txt"))
P = cand.prune(Us, b, 100)
for n in (5, 8, 10, 12, 15, 20, 25, 30, 40):
    m = eval_candidates(P.filter(pl.col("pr_rank") <= n), G, qs, n2); show(f"PRUNED top{n}", m)
    ex = Experiment(f"block_pruned_top{n}", "blocking", {"pruner": "lgb on retrieval feats (150k S1 folds1-4)", "top_n": n},
                    f"union pruned to top{n}", "screen")
    ex.log(**m); ex.finish()
imp = sorted(zip(b.feature_name(), b.feature_importance("gain")), key=lambda x: -x[1])
print("pruner importance:", [(a, int(v)) for a, v in imp[:12]])
