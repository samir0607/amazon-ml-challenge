"""Exact-key blocking: join queries and targets on a derived key (always scoped by
country). Keys shared by more than `max_block` targets are skipped as non-selective."""
import polars as pl

KEYS = {
    "name_sorted": pl.col("n_sorted"),
    "name_skel": pl.col("n_skel"),
    "name_nospace": pl.col("n_core").str.replace_all(" ", ""),
    "addr_sorted": pl.col("a_sorted"),
    # numbers + region: "618 0 | stusmd" catches reordered / reformatted addresses
    "addr_nums_region": pl.when(pl.col("a_nums") != "").then(
        pl.concat_str([pl.col("a_nums").str.split(" ").list.sort().list.join(" "), pl.col("a_region")], separator="|")),
}


def exact_block(queries: pl.DataFrame, targets: pl.DataFrame, key: str, max_block: int = 50) -> pl.DataFrame:
    expr = KEYS[key]
    qk = queries.select("q", k=pl.concat_str([pl.col("country"), expr], separator="#")).filter(
        pl.col("k").is_not_null() & ~pl.col("k").str.ends_with("#"))
    tk = targets.select("t", k=pl.concat_str([pl.col("country"), expr], separator="#")).filter(
        pl.col("k").is_not_null() & ~pl.col("k").str.ends_with("#"))
    tk = tk.join(qk.select("k").unique(), on="k", how="semi")
    sizes = tk.group_by("k").agg(n=pl.len()).filter(pl.col("n") <= max_block)
    tk = tk.join(sizes, on="k", how="semi")
    c = qk.join(tk, on="k").select("q", "t")
    # score = selectivity of the key (1/n targets sharing it)
    return c.join(c.group_by("q").agg(n=pl.len()), on="q").with_columns(
        score=(1.0 / pl.col("n")).cast(pl.Float32)).select("q", "t", "score")
