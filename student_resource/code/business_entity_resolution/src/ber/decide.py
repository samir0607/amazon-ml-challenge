"""Decision layer: turn calibrated pair probabilities into per-S1 match sets.

Rules (all operate on a frame with q, t, p):
- threshold: keep pairs with p >= thr (optionally per source)
- exclusive: each target keeps only its best S1 (ground truth has <=1 S1 per target)
- expected_f: per S1, choose the top-k prefix maximizing plug-in expected F0.5, where
  k = 0 (predict singleton) scores prod(1 - p_i).
"""
import numpy as np
import polars as pl

BETA2 = 0.25


def exclusive(df: pl.DataFrame, p="p") -> pl.DataFrame:
    return df.filter(pl.col(p) == pl.col(p).max().over("t")).unique("t", keep="first")


def threshold(df: pl.DataFrame, thr: float, p="p", thr_s3: float | None = None, n2: int | None = None) -> pl.DataFrame:
    if thr_s3 is None:
        return df.filter(pl.col(p) >= thr)
    return df.filter(pl.when(pl.col("t") < n2).then(pl.col(p) >= thr).otherwise(pl.col(p) >= thr_s3))


def expected_f(df: pl.DataFrame, p="p", min_p: float = 0.0, extra_mass: float = 0.0) -> pl.DataFrame:
    """Plug-in expected-F0.5 subset selection per S1. `extra_mass` adds expected true
    matches outside the candidate list (blocking misses) to the denominator."""
    d = df.select("q", "t", p).sort(["q", p], descending=[False, True])
    q = d["q"].to_numpy()
    pv = d[p].to_numpy().astype(np.float64)
    keep = np.zeros(len(d), dtype=bool)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    for s, e in zip(starts, ends):
        ps = pv[s:e]
        exp_g = ps.sum() + extra_mass
        cs = np.cumsum(ps)
        k = np.arange(1, e - s + 1)
        ef = (1 + BETA2) * cs / (BETA2 * exp_g + k)
        best = int(np.argmax(ef))
        p_empty = float(np.prod(1 - ps))
        if ef[best] > p_empty and ps[0] >= min_p:
            keep[s:s + best + 1] = True
    return d.filter(pl.Series(keep))
