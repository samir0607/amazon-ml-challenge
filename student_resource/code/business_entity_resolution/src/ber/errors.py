"""Error analysis: per-S1 F0.5 decomposition and categorization of FP / FN pairs."""
import numpy as np
import polars as pl


def per_s1(pred: pl.DataFrame, gt: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """pred/gt: (q, t); universe: (q). Returns q, ngt, tp, npred, F."""
    tp = pred.join(gt, on=["q", "t"]).group_by("q").agg(tp=pl.len())
    npred = pred.group_by("q").agg(npred=pl.len())
    ngt = gt.group_by("q").agg(ngt=pl.len())
    d = universe.join(ngt, on="q", how="left").join(tp, on="q", how="left").join(npred, on="q", how="left").fill_null(0)
    tp_, np_, ng_ = (d[c].to_numpy().astype(float) for c in ("tp", "npred", "ngt"))
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(np_ > 0, tp_ / np_, 0.0)
        R = np.where(ng_ > 0, tp_ / ng_, 0.0)
        F = np.where(tp_ > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    F = np.where(ng_ == 0, (np_ == 0).astype(float), F)
    return d.with_columns(F=pl.Series(F))


def loss_breakdown(pred: pl.DataFrame, cands: pl.DataFrame, gt: pl.DataFrame, universe: pl.DataFrame) -> list[dict]:
    """Share of (1 - macro F) attributable to each S1-level error type, plus the part
    that is unavoidable given the candidate set (oracle decisions inside candidates)."""
    d = per_s1(pred, gt, universe)
    oracle = per_s1(cands.join(gt, on=["q", "t"], how="semi"), gt, universe).select("q", Fo="F")
    d = d.join(oracle, on="q")
    n = d.height
    cats = {
        "singleton_false_match": (pl.col("ngt") == 0) & (pl.col("npred") > 0),
        "nonsingleton_predicted_empty": (pl.col("ngt") > 0) & (pl.col("npred") == 0),
        "nonsingleton_with_fp": (pl.col("ngt") > 0) & (pl.col("npred") > pl.col("tp")) & (pl.col("npred") > 0),
        "nonsingleton_fn_only": (pl.col("ngt") > 0) & (pl.col("npred") == pl.col("tp")) & (pl.col("tp") < pl.col("ngt")) & (pl.col("npred") > 0),
    }
    out = [{"category": "TOTAL", "n_s1": n, "loss": float((1 - d["F"]).sum() / n),
            "blocking_ceiling_loss": float((1 - d["Fo"]).sum() / n)}]
    for k, e in cats.items():
        x = d.filter(e)
        out.append({"category": k, "n_s1": x.height, "loss": float((1 - x["F"]).sum() / n),
                    "blocking_ceiling_loss": float((1 - x["Fo"]).sum() / n)})
    return out


def categorize_pairs(df: pl.DataFrame) -> pl.DataFrame:
    """Rule-based category for FP (y=0, predicted) and FN (y=1, not predicted) pairs.
    df needs pair features + y + pred + is_singleton."""
    name_hi = pl.max_horizontal("n_tset", "n_skel_tset") >= 0.9
    name_lo = pl.max_horizontal("n_tset", "n_skel_tset") < 0.6
    addr_hi = pl.col("a_tset") >= 0.85
    addr_lo = (pl.col("a_tset") >= 0) & (pl.col("a_tset") < 0.6)
    fp = pl.when(pl.col("is_singleton")).then(pl.lit("FP singleton false match: ") ).otherwise(pl.lit("FP: "))
    fp_type = (pl.when(name_hi & addr_hi & (pl.col("num_conflict") == 1)).then(pl.lit("same name+street, number conflict"))
               .when(name_hi & addr_lo).then(pl.lit("same brand, different location"))
               .when(name_hi & (pl.col("t_addr_missing") == 1)).then(pl.lit("same name, target address missing"))
               .when(name_lo & addr_hi).then(pl.lit("same address, different business"))
               .when((pl.col("name_freq_t") > 3) & name_hi).then(pl.lit("generic name collision"))
               .when(name_hi & addr_hi).then(pl.lit("near-duplicate record (likely sibling entity)"))
               .otherwise(pl.lit("weak evidence both fields")))
    fn_type = (pl.when(pl.col("t_nonascii_name") == 1).then(pl.lit("transliteration (non-Latin name)"))
               .when(pl.col("n_domain_flag") == 1).then(pl.lit("domain-style name"))
               .when(name_lo & addr_hi).then(pl.lit("DBA/alias or replaced name"))
               .when(pl.col("t_addr_missing") == 1).then(pl.lit("missing target address"))
               .when(name_hi & addr_lo).then(pl.lit("address heavily corrupted/partial"))
               .when(name_lo & addr_lo).then(pl.lit("both fields heavily corrupted"))
               .otherwise(pl.lit("threshold/competition (moderate evidence)")))
    return df.with_columns(
        category=pl.when((pl.col("y") == 0) & pl.col("pred")).then(pl.concat_str([fp, fp_type]))
        .when((pl.col("y") == 1) & ~pl.col("pred")).then(pl.concat_str([pl.lit("FN: "), fn_type]))
        .otherwise(None))
