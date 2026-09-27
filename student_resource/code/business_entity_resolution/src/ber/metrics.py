"""Challenge metric (macro F0.5 over S1 entities, singletons included) and
candidate-set statistics. All inputs are long-format polars frames keyed by S1 id."""
import numpy as np
import polars as pl

BETA2 = 0.25


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred: (s1_id, match_id) predicted pairs; gt: (s1_id, match_id) with null match_id
    rows allowed; s1_ids: the evaluated S1 universe (every entity counts, even if absent
    from pred). Returns overall and breakdown metrics."""
    universe = pl.DataFrame({"s1_id": s1_ids.unique()})
    gt = gt.drop_nulls("match_id").join(universe, on="s1_id", how="semi")
    pred = pred.drop_nulls("match_id").join(universe, on="s1_id", how="semi").unique()
    tp = pred.join(gt, on=["s1_id", "match_id"], how="inner").group_by("s1_id").agg(tp=pl.len())
    npred = pred.group_by("s1_id").agg(npred=pl.len())
    ngt = gt.group_by("s1_id").agg(ngt=pl.len())
    d = (universe.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
         .join(ngt, on="s1_id", how="left").fill_null(0))
    tp_, np_, ng_ = (d[c].to_numpy().astype(np.float64) for c in ("tp", "npred", "ngt"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(np_ > 0, tp_ / np_, 0.0)
        r = np.where(ng_ > 0, tp_ / ng_, 0.0)
        f = np.where(tp_ > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    f = np.where(ng_ == 0, (np_ == 0).astype(np.float64), f)
    single = ng_ == 0
    return {
        "macro_f05": float(f.mean()),
        "n_s1": int(len(f)),
        "micro_precision": float(tp_.sum() / max(np_.sum(), 1)),
        "micro_recall": float(tp_.sum() / max(ng_.sum(), 1)),
        "f05_singletons": float(f[single].mean()) if single.any() else None,
        "f05_nonsingletons": float(f[~single].mean()) if (~single).any() else None,
        "singleton_false_match_rate": float((np_[single] > 0).mean()) if single.any() else None,
        "empty_pred_on_nonsingleton_rate": float((np_[~single] == 0).mean()) if (~single).any() else None,
        "n_pred_pairs": int(np_.sum()),
    }


def candidate_stats(cand: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series, n_targets: int | None = None) -> dict:
    """cand: (s1_id, match_id) candidate pairs. Recall is pair-level over true pairs of
    the evaluated S1 universe; also reports per-entity full-coverage and size stats."""
    universe = pl.DataFrame({"s1_id": s1_ids.unique()})
    gt = gt.drop_nulls("match_id").join(universe, on="s1_id", how="semi")
    cand = cand.join(universe, on="s1_id", how="semi").unique(["s1_id", "match_id"])
    hit = gt.join(cand, on=["s1_id", "match_id"], how="semi")
    sizes = universe.join(cand.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").fill_null(0)["n"].to_numpy()
    per = gt.with_columns(h=pl.struct("s1_id", "match_id").is_in(hit.select(pl.struct("s1_id", "match_id")).to_series().implode()))
    full = per.group_by("s1_id").agg(pl.col("h").all())["h"].mean()
    out = {
        "pair_recall": hit.height / max(gt.height, 1),
        "recall_s2": _src_recall(gt, hit, "S2-"),
        "recall_s3": _src_recall(gt, hit, "S3-"),
        "entity_full_coverage": float(full) if full is not None else None,
        "n_true_pairs": gt.height,
        "n_cand_pairs": cand.height,
        "cand_mean": float(sizes.mean()),
        "cand_median": float(np.median(sizes)),
        "cand_p95": float(np.percentile(sizes, 95)),
        "cand_max": int(sizes.max()) if len(sizes) else 0,
        "pct_s1_no_cand": float((sizes == 0).mean()),
    }
    if n_targets:
        out["reduction_ratio"] = 1.0 - cand.height / (len(universe) * n_targets)
    return out


def _src_recall(gt, hit, prefix):
    g = gt.filter(pl.col("match_id").str.starts_with(prefix)).height
    h = hit.filter(pl.col("match_id").str.starts_with(prefix)).height
    return h / g if g else None
