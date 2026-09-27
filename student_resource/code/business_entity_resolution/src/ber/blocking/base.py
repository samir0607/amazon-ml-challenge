"""Shared blocking utilities. A blocker returns a polars frame (q, t, score) with at
most one row per pair; `rank` is added per q by descending score."""
import numpy as np
import polars as pl


def add_rank(c: pl.DataFrame, score="score", name="rank") -> pl.DataFrame:
    return c.with_columns(pl.col(score).rank("ordinal", descending=True).over("q").cast(pl.Int32).alias(name))


def eval_candidates(cand: pl.DataFrame, gt: pl.DataFrame, qs: np.ndarray, n2: int) -> dict:
    """Int-space candidate stats for the query set `qs` (numpy array of q)."""
    uq = pl.DataFrame({"q": np.unique(qs).astype(np.int32)})
    g = gt.join(uq, on="q", how="semi")
    c = cand.select("q", "t").join(uq, on="q", how="semi").unique()
    hit = g.join(c, on=["q", "t"], how="semi")
    sizes = uq.join(c.group_by("q").agg(n=pl.len()), on="q", how="left").fill_null(0)["n"].to_numpy()
    full = g.join(hit.with_columns(h=pl.lit(True)), on=["q", "t"], how="left").group_by("q").agg(pl.col("h").is_not_null().all())["h"].mean()
    n_s2 = g.filter(pl.col("t") < n2).height
    return {
        "pair_recall": hit.height / max(g.height, 1),
        "recall_s2": hit.filter(pl.col("t") < n2).height / max(n_s2, 1),
        "recall_s3": hit.filter(pl.col("t") >= n2).height / max(g.height - n_s2, 1),
        "entity_full_coverage": float(full),
        "cand_mean": float(sizes.mean()),
        "cand_median": float(np.median(sizes)),
        "cand_p95": float(np.percentile(sizes, 95)),
        "cand_max": int(sizes.max()),
        "pct_q_no_cand": float((sizes == 0).mean()),
        "n_cand_pairs": c.height,
    }
