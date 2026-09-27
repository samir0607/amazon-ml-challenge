"""Screen TF-IDF top-K blockers on the 50k screening S1 set. Retrieves K_MAX per query
once, then reports recall/size at several K cutoffs."""
import json
import sys
import time

import polars as pl

from ber.blocking.base import add_rank, eval_candidates
from ber.blocking.tfidf import build_matrices, topk_block
from ber.data import load_gt_pairs, query_meta, target_meta
from ber.experiment import Experiment
from ber.paths import CACHE
from ber.splits import load_folds

CONFIGS = {
    "name_char3": dict(field="n_core", analyzer="char_wb", ngram=(3, 3)),
    "name_skel_char3": dict(field="n_skel", analyzer="char_wb", ngram=(2, 3)),
    "addr_char3": dict(field="a_norm", analyzer="char_wb", ngram=(3, 3)),
    "nameaddr_word": dict(field="nameaddr_word", analyzer="word", ngram=(1, 1), max_df=0.02,
                          field_expr=pl.concat_str([pl.col("n_core"), pl.col("n_skel").str.replace_all(r"(\S+)", "~$1"),
                                                    pl.col("a_norm")], separator=" ")),
    "nameaddr_char3": dict(field="nameaddr_char3", analyzer="char_wb", ngram=(3, 3),
                           field_expr=pl.concat_str([pl.col("n_core"), pl.col("a_norm")], separator=" | ")),
}
K_MAX, KS = 50, (5, 10, 20, 50)

def main():
    names = sys.argv[1:] or list(CONFIGS)
    QM, TM, G = query_meta("train"), target_meta("train"), load_gt_pairs()
    n2 = int((TM["src"] == 2).sum())
    qs = load_folds().filter("screen")["idx"].to_numpy()
    for name in names:
        cfg = dict(CONFIGS[name])
        t0 = time.time()
        pre = build_matrices("train", **cfg)
        t_build = time.time() - t0
        c = add_rank(topk_block(pre, QM, qs, k=K_MAX))
        t_query = time.time() - t0 - t_build
        c.write_parquet(CACHE / "cands" / f"screen_tfidf_{name}.parquet")
        for k in KS:
            ex = Experiment(f"block_tfidf_{name}_k{k}", "blocking",
                            {"blocker": "tfidf", **{a: str(b) for a, b in cfg.items()}, "k": k},
                            f"tfidf {name} top{k}", tier="screen")
            m = eval_candidates(c.filter(pl.col("rank") <= k), G, qs, n2)
            ex.log(**m, build_s=round(t_build, 1), query_s_50k=round(t_query, 1))
            ex.finish()
            print(name, k, json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in m.items()}),
                  f"build={t_build:.0f}s query={t_query:.0f}s", flush=True)


if __name__ == "__main__":
    main()
