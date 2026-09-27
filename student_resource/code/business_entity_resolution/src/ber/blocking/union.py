"""Candidate generation for a whole split: run each configured blocker, cache its raw
output, and outer-join them into one (q, t) table carrying every blocker's score and
rank as retrieval features.

Blocker spec: {"name": str, "kind": "fwd" | "rev" | "exact", "cfg": {...}, "k": int}
 - fwd  : S1 -> targets TF-IDF top-k (per country)
 - rev  : targets -> S1 TF-IDF top-k; optional "subset" polars expr on targets
 - exact: exact key join (cfg: {"key": ..., "max_block": ...})
"""
import json

import numpy as np
import polars as pl

from ..data import load_targets, query_meta
from ..normalize import NORM_VERSION
from ..paths import CACHE
from .exact import exact_block
from .tfidf import build_matrices, reverse_topk_block, topk_block

TFIDF_CONFIGS = {
    "name_char3": dict(field="n_core", analyzer="char_wb", ngram=(3, 3)),
    "nameaddr_word": dict(field="nameaddr_word", analyzer="word", ngram=(1, 1), max_df=0.02),
    "addr_word": dict(field="a_norm", analyzer="word", ngram=(1, 1), max_df=0.02),
    "name_word": dict(field="name_word", analyzer="word", ngram=(1, 1), max_df=0.02),
    "skel_char": dict(field="n_skel", analyzer="char_wb", ngram=(2, 3), max_df=0.05),
}


def field_expr(field: str):
    if field == "nameaddr_word":
        return pl.concat_str([pl.col("n_core"), pl.col("n_skel").str.replace_all(r"(\S+)", "~$1"),
                              pl.col("a_norm")], separator=" ")
    if field == "name_word":
        return pl.concat_str([pl.col("n_core"), pl.col("n_skel").str.replace_all(r"(\S+)", "~$1")], separator=" ")
    return pl.col(field)


def run_blocker(split: str, spec: dict, qs: np.ndarray | None = None) -> pl.DataFrame:
    """Raw (q, t, score) for one blocker over all S1 of the split (cached)."""
    path = blocker_path(split, spec)
    if path.exists():
        return pl.read_parquet(path)
    qm = query_meta(split)
    if qs is None:
        qs = qm["q"].to_numpy()
    kind = spec["kind"]
    if kind in ("fwd", "rev"):
        cfg = dict(TFIDF_CONFIGS[spec["cfg"]])
        pre = build_matrices(split, field_expr=field_expr(cfg["field"]), **cfg)
        if kind == "fwd":
            c = topk_block(pre, qm, qs, k=spec["k"])
        else:
            keep = None
            if spec.get("subset") == "nonascii_name":
                keep = load_targets(split, ["business_name"]).filter(
                    ~pl.col("business_name").str.contains(r"^[\x00-\x7F]*$"))["t"].to_numpy()
            c = reverse_topk_block(pre, qm, k=spec["k"], t_subset=keep)
    elif kind == "exact":
        from ..data import load_queries
        from .exact import KEYS
        need = sorted(set(KEYS[spec["cfg"]["key"]].meta.root_names()) | {"country"})
        c = exact_block(load_queries(split, need), load_targets(split, need), spec["cfg"]["key"],
                        spec["cfg"].get("max_block", 50))
    else:
        raise ValueError(kind)
    path.parent.mkdir(exist_ok=True)
    c.write_parquet(path)
    return c


def blocker_path(split: str, spec: dict):
    import hashlib
    tag = json.dumps({k: v for k, v in spec.items() if k != "k_use"}, sort_keys=True)
    h = hashlib.md5((tag + NORM_VERSION).encode()).hexdigest()[:8]
    return CACHE / "cands" / f"{split}_{spec['name']}_{h}.parquet"


def union_candidates(split: str, specs: list[dict], qs: np.ndarray | None = None) -> pl.DataFrame:
    """Outer-join all blockers, optionally restricted to S1 subset `qs`. Rank for fwd is
    per q; for rev it is per t (how highly the target ranked this S1) and is computed on
    the full file before any q filtering. Blocker outputs must already be cached."""
    qdf = None if qs is None else pl.DataFrame({"q": np.asarray(qs, dtype=np.int32)}).lazy()
    out = None
    for spec in specs:
        n = spec["name"]
        path = blocker_path(split, spec)
        if not path.exists():
            run_blocker(split, spec)
        lz = pl.scan_parquet(path)
        over = "t" if spec["kind"] == "rev" else "q"
        lz = lz.with_columns(pl.col("score").rank("ordinal", descending=True).over(over).cast(pl.Int16).alias("r"))
        lz = lz.filter(pl.col("r") <= spec.get("k_use", spec.get("k", 10**4)))
        if qdf is not None:
            lz = lz.join(qdf, on="q", how="semi")
        c = lz.select("q", "t", pl.col("score").alias(f"{n}_score"), pl.col("r").alias(f"{n}_rank")).collect()
        out = c if out is None else out.join(c, on=["q", "t"], how="full", coalesce=True)
    score_cols = [f"{s['name']}_score" for s in specs]
    return out.with_columns(n_blockers=pl.sum_horizontal([pl.col(c).is_not_null() for c in score_cols]).cast(pl.Int8))


def target_rev_stats(split: str, spec: dict) -> pl.DataFrame:
    """Per-target best / second-best reverse score (global, label-free)."""
    path = CACHE / "cands" / f"tstats_{blocker_path(split, spec).stem}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = (pl.scan_parquet(blocker_path(split, spec)).group_by("t")
         .agg(t_top1=pl.col("score").max(), t_top2=pl.col("score").sort(descending=True).slice(1, 1).first())
         .with_columns(pl.col("t_top2").fill_null(0.0)).collect())
    d.write_parquet(path)
    return d
