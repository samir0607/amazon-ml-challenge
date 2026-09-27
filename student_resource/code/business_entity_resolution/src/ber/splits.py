"""Leakage-safe grouped split: folds are assigned per Source 1 entity, never per pair.

fold 0 is the fixed primary validation fold. `screen` is a fixed 50k-S1 subset of
fold 0 used for fast screening. The assignment is a seeded permutation of the sorted
S1 ids, so it is reproducible and independent of file order.
"""
import numpy as np
import polars as pl

from .io import load_source
from .paths import CACHE

N_FOLDS = 5
SEED = 20260927
SCREEN_N = 50_000


def load_folds() -> pl.DataFrame:
    """Columns: entity_id, idx (S1 row idx), fold (0..4), screen (bool)."""
    cache = CACHE / "folds.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    s1 = load_source("train", "source1").select("entity_id", "idx").sort("entity_id")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(s1.height)
    fold = np.empty(s1.height, dtype=np.int8)
    fold[perm] = np.arange(s1.height) % N_FOLDS
    s1 = s1.with_columns(fold=pl.Series(fold))
    f0 = np.flatnonzero(fold == 0)
    screen = np.zeros(s1.height, dtype=bool)
    screen[rng.choice(f0, SCREEN_N, replace=False)] = True
    s1 = s1.with_columns(screen=pl.Series(screen))
    s1.write_parquet(cache)
    return s1
