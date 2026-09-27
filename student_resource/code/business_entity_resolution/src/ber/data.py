"""Integer-indexed views used by blocking/features.

q  : S1 row index (int32)
t  : global target index over S2 ++ S3 (S2 rows first), int32
"""
import polars as pl

from .io import load_ground_truth
from .normalize import NORM_VERSION, load_normalized
from .paths import CACHE


def _cols(split, source, columns):
    load_normalized(split, source)  # ensure cache exists
    return pl.read_parquet(CACHE / f"norm_{NORM_VERSION}_{split}_{source}.parquet", columns=columns)


def load_queries(split: str, columns: list | None = None) -> pl.DataFrame:
    cols = None if columns is None else ["idx"] + [c for c in columns if c not in ("q", "idx")]
    return _cols(split, "source1", cols).rename({"idx": "q"})


def load_targets(split: str, columns: list | None = None) -> pl.DataFrame:
    """Not cached in-process on purpose: the full frame is ~5 GB. Pass `columns`."""
    cols = None if columns is None else [c for c in columns if c not in ("t", "src")]
    s2 = _cols(split, "source2", cols).with_columns(src=pl.lit(2, pl.Int8))
    s3 = _cols(split, "source3", cols).with_columns(src=pl.lit(3, pl.Int8))
    t = pl.concat([s2, s3])
    if "idx" in t.columns:
        t = t.drop("idx")
    return t.with_row_index("t").with_columns(pl.col("t").cast(pl.Int32))


def target_meta(split: str) -> pl.DataFrame:
    """Light (t, src, country) view — what blocking needs without the text columns."""
    cache = CACHE / f"tmeta_{split}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    m = load_targets(split, ["country"]).select("t", "src", "country")
    m.write_parquet(cache)
    return m


def query_meta(split: str) -> pl.DataFrame:
    return load_normalized(split, "source1").select(q="idx", country="country")


def load_gt_pairs() -> pl.DataFrame:
    """(q, t) int pairs of true matches for the train split."""
    cache = CACHE / "gt_pairs.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    q = load_queries("train", ["entity_id"]).select(s1_id="entity_id", q="q")
    t = load_targets("train", ["entity_id"]).select(match_id="entity_id", t="t")
    g = load_ground_truth().drop_nulls("match_id").join(q, on="s1_id").join(t, on="match_id").select("q", "t")
    g.write_parquet(cache)
    return g
