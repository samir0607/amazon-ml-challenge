"""Cross-source / sibling consistency features.

An S1 entity usually has several S2/S3 records. For a candidate (q, t) we compare t with
the S1's *other* plausible candidates t' (stage-1 p1 >= min_p): if t looks like records
that are confidently matched to q, that supports the match; if it does not, that argues
against it. This is used only as features — no transitivity is enforced.
"""
from multiprocessing import get_context

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .features import load_pair_inputs
from .paths import CACHE

SIB_COLS = ["sib_max_w", "sib_top_sim", "sib_n_strong", "sib_wmean", "sib_same_src_max", "sib_n"]


def _sim(a):
    (nc1, nk1, a1, nu1), (nc2, nk2, a2, nu2) = a
    s_name = max(fuzz.token_set_ratio(nc1, nc2), fuzz.token_set_ratio(nk1, nk2)) / 100
    if a1 and a2:
        s_addr = fuzz.token_set_ratio(a1, a2) / 100
        return 0.5 * s_name + 0.5 * s_addr
    return s_name


def _chunk(rows):
    return np.asarray([_sim(r) for r in rows], dtype=np.float32)


def _pair_sims(pairs_ab: pl.DataFrame, tdf: pl.DataFrame, pool) -> pl.DataFrame:
    fa = [f"{c}_a" for c in ("n_core", "n_skel", "a_norm", "a_nums")]
    fb = [f"{c}_b" for c in ("n_core", "n_skel", "a_norm", "a_nums")]
    j = (pairs_ab.join(tdf.rename({c: f"{c}_a" for c in tdf.columns if c != "t"}), left_on="a", right_on="t", how="left")
         .join(tdf.rename({c: f"{c}_b" for c in tdf.columns if c != "t"}), left_on="b", right_on="t", how="left"))
    j = j.with_columns([pl.col(c).fill_null("") for c in fa + fb])
    rows = list(zip(zip(*[j[c].to_list() for c in fa]), zip(*[j[c].to_list() for c in fb])))
    chunks = [rows[i:i + 50_000] for i in range(0, len(rows), 50_000)]
    sim = np.concatenate(pool.map(_chunk, chunks, chunksize=1)) if chunks else np.zeros(0, np.float32)
    return j.select("a", "b").with_columns(sim=pl.Series(sim, dtype=pl.Float32))


def sibling_features(split: str, p1: pl.DataFrame, min_p: float = 0.05, procs: int = 9,
                     q_chunk: int = 150_000) -> pl.DataFrame:
    """p1: (q, t, p1) over all candidates of the split. Returns (q, t, SIB_COLS) for
    candidates with p1 >= min_p (others get nulls when joined). Processed in S1 chunks
    to bound memory."""
    _, tdf = load_pair_inputs(split)
    tdf = tdf.select("t", "n_core", "n_skel", "a_norm", "a_nums")
    n2 = _n2(split)
    s = p1.filter(pl.col("p1") >= min_p).select("q", "t", "p1")
    qs = np.unique(s["q"].to_numpy())
    out = []
    with get_context("fork").Pool(procs) as pool:
        for i in range(0, len(qs), q_chunk):
            sc = s.filter(pl.col("q").is_in(qs[i:i + q_chunk]))
            pairs = sc.join(sc.rename({"t": "t2", "p1": "p2"}), on="q").filter(pl.col("t") != pl.col("t2"))
            pairs = pairs.with_columns(pl.min_horizontal("t", "t2").alias("a"), pl.max_horizontal("t", "t2").alias("b"))
            simdf = _pair_sims(pairs.select("a", "b").unique(), tdf, pool)
            pairs = pairs.join(simdf, on=["a", "b"], how="left")
            same = (pl.col("t") < n2) == (pl.col("t2") < n2)
            top = pairs.sort("p2", descending=True).group_by(["q", "t"]).agg(sib_top_sim=pl.col("sim").first())
            agg = pairs.group_by(["q", "t"]).agg(
                sib_max_w=(pl.col("p2") * pl.col("sim")).max(),
                sib_n_strong=((pl.col("p2") > 0.5) & (pl.col("sim") > 0.8)).sum().cast(pl.Float32),
                sib_wmean=(pl.col("p2") * pl.col("sim")).sum() / pl.col("p2").sum(),
                sib_same_src_max=pl.when(same).then(pl.col("sim")).otherwise(None).max(),
                sib_n=pl.len().cast(pl.Float32),
            )
            out.append(agg.join(top, on=["q", "t"]).select(["q", "t"] + SIB_COLS))
            print(f"  siblings {split}: {min(i + q_chunk, len(qs))}/{len(qs)} S1", flush=True)
    return pl.concat(out)


def _n2(split):
    from .data import target_meta
    return int((target_meta(split)["src"] == 2).sum())


def cached_sibling_features(split: str, p1: pl.DataFrame, name: str) -> pl.DataFrame:
    path = CACHE / f"extra_sib_{split}_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = sibling_features(split, p1)
    d.write_parquet(path)
    return d
