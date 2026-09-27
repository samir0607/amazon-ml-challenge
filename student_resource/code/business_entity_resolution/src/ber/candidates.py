"""Final candidate set = union of blockers, pruned by a light pre-ranker trained only on
retrieval features. The pruned set is exactly what the matcher scores and what is
written to candidate_pairs.tsv.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking.union import target_rev_stats, union_candidates
from .data import target_meta
from .paths import CACHE

RETR_SPECS = [
    {"name": "rev_nameaddr", "kind": "rev", "cfg": "nameaddr_word", "k": 5},
    {"name": "rev_addr", "kind": "rev", "cfg": "addr_word", "k": 3},
    {"name": "rev_name", "kind": "rev", "cfg": "name_word", "k": 3},
    {"name": "fwd_nameaddr", "kind": "fwd", "cfg": "nameaddr_word", "k": 30},
    {"name": "fwd_addr", "kind": "fwd", "cfg": "addr_word", "k": 10},
]
# exact_name_sorted / exact_addr_sorted were dropped: leave-one-out ablation on the
# screening set showed <=0.0001 recall contribution on top of the TF-IDF blockers.


def retrieval_frame(split: str, specs=RETR_SPECS, qs=None) -> pl.DataFrame:
    """Union of blockers (optionally for an S1 subset) with retrieval features plus
    label-free context: gap to the S1's best candidate, and the target's global best /
    second-best reverse scores (how contested the target is among all S1)."""
    u = union_candidates(split, specs, qs)
    tm = target_meta(split).select("t", "src")
    ts = target_rev_stats(split, specs[0])
    u = u.join(tm, on="t", how="left").join(ts, on="t", how="left")
    s = pl.col("rev_nameaddr_score").fill_null(0.0)
    return u.with_columns(
        (s.max().over("q") - s).alias("r_qgap"),
        (pl.col("t_top1") - s).alias("r_tgap"),
        (pl.col("t_top1") - pl.col("t_top2")).alias("r_tmargin"),
        pl.len().over("q").cast(pl.Float32).alias("r_q_n"),
    )


def retr_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(("_score", "_rank")) or c in
            ("n_blockers", "src", "r_qgap", "r_tgap", "r_tmargin", "r_q_n", "t_top1", "t_top2")]


def train_pruner(df: pl.DataFrame, y: np.ndarray, seed: int = 0) -> lgb.Booster:
    cols = retr_feature_cols(df)
    X = df.select(cols).to_numpy().astype(np.float32)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=seed, verbose=-1,
                  num_threads=9)
    b = lgb.train(params, lgb.Dataset(X, y, feature_name=cols), num_boost_round=200)
    return b


def prune(df: pl.DataFrame, booster: lgb.Booster, top_n: int, min_p: float = 0.0) -> pl.DataFrame:
    cols = booster.feature_name()
    p = booster.predict(df.select(cols).to_numpy().astype(np.float32), num_threads=9)
    df = df.with_columns(pr=pl.Series(p.astype(np.float32)))
    df = df.with_columns(pr_rank=pl.col("pr").rank("ordinal", descending=True).over("q").cast(pl.Int16))
    return df.filter((pl.col("pr_rank") <= top_n) & (pl.col("pr") >= min_p))


def save(df: pl.DataFrame, name: str):
    df.write_parquet(CACHE / f"{name}.parquet")


def build_pruned(split: str, booster: lgb.Booster, top_n: int, chunk: int = 200_000) -> pl.DataFrame:
    """Union + prune over all S1 of a split, in S1 chunks to bound memory."""
    from .data import query_meta
    qs = query_meta(split)["q"].to_numpy()
    out = []
    for i in range(0, len(qs), chunk):
        u = retrieval_frame(split, qs=qs[i:i + chunk])
        out.append(prune(u, booster, top_n))
        print(f"  pruned {split} {min(i + chunk, len(qs))}/{len(qs)}", flush=True)
    return pl.concat(out)
