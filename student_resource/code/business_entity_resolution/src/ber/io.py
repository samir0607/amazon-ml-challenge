"""Raw TSV loading with a parquet cache.

Raw text is never modified here; normalization lives in `normalize.py`.
Every record gets a dense int32 row index (`idx`) within its (split, source) table,
which is what all downstream arrays use.
"""
import polars as pl

from .paths import CACHE, DATA

SOURCES = ("source1", "source2", "source3")


def _read_tsv(path) -> pl.DataFrame:
    # quote_char=None: business names legitimately contain quotes.
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=False, truncate_ragged_lines=False,
    )


def load_source(split: str, source: str) -> pl.DataFrame:
    """Columns: idx (int32), entity_id, business_name, business_address, country."""
    cache = CACHE / f"raw_{split}_{source}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    df = _read_tsv(DATA / split / f"{split}_{source}.tsv")
    df = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    df.write_parquet(cache)
    return df


def load_ground_truth() -> pl.DataFrame:
    """Long format: one row per (s1_id, match_id). S1 entities with no match appear
    once with match_id = null so singletons are never lost."""
    cache = CACHE / "gt_long.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    gt = _read_tsv(DATA / "train" / "train_ground_truth.tsv")
    gt = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("match_id"))
        .explode("match_id")
        .with_columns(pl.when(pl.col("match_id") == "").then(None)
                      .otherwise(pl.col("match_id")).alias("match_id"))
        .select(s1_id="source1_entity_id", match_id="match_id")
    )
    gt.write_parquet(cache)
    return gt
