"""Stage-2 feature assembly shared by training and inference, so both paths build
exactly the same features from (q, t, p1)."""
import polars as pl

from .features import add_context_features
from .siblings import SIB_COLS, sibling_features

CTX_COLS = ["p1", "c1_qrank", "c1_qgap", "c1_trank", "c1_tgap_best", "c1_t_ncomp", "c1_q_ncand", "c1_tmargin",
            "c1_q_sum", "c1_q_n50", "c1_q_n20", "c1_q_max", "c1_t_sum_other"]
STAGE2_EXTRA = CTX_COLS + SIB_COLS


def context_frame(p1: pl.DataFrame) -> pl.DataFrame:
    """(q, t, p1) over ALL candidates of a split -> (q, t, CTX_COLS)."""
    c = add_context_features(p1.select("q", "t", "p1"), "p1", "c1")
    c = c.with_columns(
        c1_q_sum=pl.col("p1").sum().over("q"),
        c1_q_n50=(pl.col("p1") > 0.5).sum().over("q").cast(pl.Float32),
        c1_q_n20=(pl.col("p1") > 0.2).sum().over("q").cast(pl.Float32),
        c1_q_max=pl.col("p1").max().over("q"),
        c1_t_sum_other=(pl.col("p1").sum().over("t") - pl.col("p1")),
    )
    return c.select(["q", "t"] + CTX_COLS)


def stage2_frame(split: str, p1: pl.DataFrame, sib: pl.DataFrame | None = None) -> pl.DataFrame:
    """Context + sibling features for every candidate pair of the split."""
    if sib is None:
        sib = sibling_features(split, p1)
    return context_frame(p1).join(sib, on=["q", "t"], how="left")
