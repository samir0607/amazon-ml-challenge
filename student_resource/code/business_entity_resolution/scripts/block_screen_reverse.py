"""Reverse retrieval screen: every target queries the S1 index (top-k within country);
candidates for S1 q are the targets that ranked q in their top-k. Evaluated on the
50k screening S1 set; also reports the union with forward top-K of the same config."""
import json
import sys
import time

import polars as pl

from ber.blocking.base import eval_candidates
from ber.blocking.tfidf import build_matrices, reverse_topk_block
from ber.data import load_gt_pairs, query_meta, target_meta
from ber.experiment import Experiment
from ber.paths import CACHE
from ber.splits import load_folds

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from block_screen_tfidf import CONFIGS  # noqa: E402  (module guards its own main)

name = sys.argv[1]
K_REV = 5
QM, TM, G = query_meta("train"), target_meta("train"), load_gt_pairs()
n2 = int((TM["src"] == 2).sum())
qs = load_folds().filter("screen")["idx"].to_numpy()
pre = build_matrices("train", **CONFIGS[name])
t0 = time.time()
cache = CACHE / "cands" / f"full_rev_{name}.parquet"
if cache.exists():
    rev = pl.read_parquet(cache)
else:
    rev = reverse_topk_block(pre, QM, k=K_REV)
    rev = rev.with_columns(rrank=pl.col("score").rank("ordinal", descending=True).over("t").cast(pl.Int8))
    rev.write_parquet(cache)
t_rev = time.time() - t0
fwd = pl.read_parquet(CACHE / "cands" / f"screen_tfidf_{name}.parquet")
for kr in (1, 2, 3, 5):
    r = rev.filter(pl.col("rrank") <= kr)
    for kf in (0, 10, 20):
        c = pl.concat([r.select("q", "t"), fwd.filter(pl.col("rank") <= kf).select("q", "t")]).unique()
        ex = Experiment(f"block_rev_{name}_r{kr}_f{kf}", "blocking",
                        {"blocker": "tfidf_reverse+forward", "config": name, "k_rev": kr, "k_fwd": kf},
                        f"reverse top{kr} U forward top{kf} ({name})", tier="screen")
        m = eval_candidates(c, G, qs, n2)
        ex.log(**m, reverse_all_targets_s=round(t_rev, 1))
        ex.finish()
        print(name, f"rev{kr} fwd{kf}", json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in m.items()}), flush=True)
