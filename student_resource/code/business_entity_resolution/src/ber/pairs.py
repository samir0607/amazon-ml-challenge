"""Chunked pair-feature computation over a candidate frame, written as parquet parts.

Adds string/numeric pair features (features.py) and generic-name frequency features.
Row identity is (q, t); retrieval feature columns from the candidate frame are kept.
"""
import gc
import time

import polars as pl

from .features import FEATURE_NAMES, load_pair_inputs, pair_features
from .paths import CACHE


def name_freq_tables(split: str, qdf: pl.DataFrame, tdf: pl.DataFrame):
    """How many targets / S1 records share a sorted core name (generic-name signal)."""
    ft = tdf.group_by("n_sorted").agg(name_freq_t=pl.len())
    fq = qdf.group_by("n_sorted").agg(name_freq_q=pl.len())
    return ft, fq


def compute_pair_features(split: str, cands: pl.DataFrame, name: str, chunk: int = 500_000) -> pl.DataFrame:
    out_dir = CACHE / "pairs" / f"{split}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    qdf, tdf = load_pair_inputs(split)
    ft, fq = name_freq_tables(split, qdf, tdf)
    cands = cands.sort("q")
    n = cands.height
    for i, start in enumerate(range(0, n, chunk)):
        part = out_dir / f"part_{i:04d}.parquet"
        if part.exists():
            continue
        t0 = time.time()
        c = cands.slice(start, chunk)
        f = pair_features(c, qdf, tdf, split)
        f = (f.join(qdf.select("q", "n_sorted"), on="q", how="left")
             .join(ft, on="n_sorted", how="left").join(fq, on="n_sorted", how="left").drop("n_sorted")
             .join(tdf.select("t", t_sorted="n_sorted"), on="t", how="left")
             .join(ft.rename({"n_sorted": "t_sorted", "name_freq_t": "t_name_freq_t"}), on="t_sorted", how="left")
             .drop("t_sorted")
             .with_columns([pl.col(c).fill_null(0).log1p().cast(pl.Float32)
                            for c in ("name_freq_t", "name_freq_q", "t_name_freq_t")]))
        f.write_parquet(part)
        print(f"  pairs {name} part {i}: {c.height} rows {time.time() - t0:.0f}s", flush=True)
        del f
        gc.collect()
    return pl.read_parquet(out_dir / "*.parquet")


MODEL_FEATURES_BASE = FEATURE_NAMES + ["name_freq_t", "name_freq_q", "t_name_freq_t"]
