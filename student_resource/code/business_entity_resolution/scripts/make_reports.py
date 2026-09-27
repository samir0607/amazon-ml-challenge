"""Generate reports/ artifacts from cached data + experiments/experiment_log.csv:
 - reports/dataset_report.md
 - reports/candidate_analysis.csv  (blocking experiments)
 - reports/ablation_results.csv    (matcher experiments + ablations)
 - reports/error_analysis.csv / .html (FP/FN categories of the best validation run)
"""
import argparse
import json

import polars as pl

from ber.data import load_gt_pairs
from ber.decide import exclusive, expected_f
from ber.errors import categorize_pairs, loss_breakdown
from ber.io import load_ground_truth, load_source
from ber.paths import CACHE, EXPERIMENTS, REPORTS
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--val-pred", default="val_pred_v2_top12.parquet")
ap.add_argument("--feat-name", default="v1_top12")
ap.add_argument("--skip-dataset", action="store_true")
args = ap.parse_args()


ASCII = r"^[\x00-\x7F]*$"


def dataset_report():
    lines = ["# Dataset report", ""]
    for split in ("train", "test"):
        lines += [f"## {split}", "", "| source | rows | countries | addr missing | non-ASCII name | non-ASCII addr | name len (median) | addr len (median) |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in ("source1", "source2", "source3"):
            d = load_source(split, s)
            n = d["business_name"]
            a = d["business_address"]
            ctry = ", ".join(f"{r['country']} {r['count']:,}" for r in d["country"].value_counts().sort("count", descending=True).to_dicts())
            na_name = (~n.str.contains(ASCII)).mean()
            na_addr = (~a.fill_null("").str.contains(ASCII)).mean()
            lines.append(f"| {s} | {d.height:,} | {ctry} | {a.is_null().mean():.2%} | {na_name:.2%} | {na_addr:.2%} | "
                         f"{n.str.len_chars().median():.0f} | {a.str.len_chars().median():.0f} |")
        lines.append("")
    g = load_ground_truth()
    cnt = g.group_by("s1_id").agg(n=pl.col("match_id").count())
    lines += ["## Ground truth (train)", "",
              f"- S1 entities: {cnt.height:,}; singletons (no match): {(cnt['n'] == 0).sum():,} ({(cnt['n'] == 0).mean():.2%})",
              f"- true pairs: {g.drop_nulls().height:,}; mean matches per S1: {cnt['n'].mean():.2f}; max {cnt['n'].max()}",
              "- every S2/S3 record matches at most one S1 (exclusivity holds for 100% of labelled records)",
              f"- matched share of S2 / S3 records: {g.drop_nulls().filter(pl.col('match_id').str.starts_with('S2'))['match_id'].n_unique() / load_source('train', 'source2').height:.1%} / "
              f"{g.drop_nulls().filter(pl.col('match_id').str.starts_with('S3'))['match_id'].n_unique() / load_source('train', 'source3').height:.1%}",
              "- country label agrees on 100% of true pairs (used to scope blocking; open set: France appears only in test)", "",
              "### Matches per S1", "", "| matches | S1 count |", "|---|---|"]
    for r in cnt["n"].value_counts().sort("n").to_dicts():
        lines.append(f"| {r['n']} | {r['count']:,} |")
    (REPORTS / "dataset_report.md").write_text("\n".join(lines) + "\n")


def experiment_tables():
    log = pl.read_csv(EXPERIMENTS / "experiment_log.csv", infer_schema_length=0)
    num = ["pair_recall", "cand_mean", "cand_p95", "macro_f05", "micro_precision", "micro_recall", "f05_singletons", "runtime_s"]
    log = log.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    log.filter(pl.col("stage") == "blocking").select(
        "exp_id", "description", "tier", "pair_recall", "cand_mean", "cand_p95", "runtime_s", "decision", "notes"
    ).write_csv(REPORTS / "candidate_analysis.csv")
    log.filter(pl.col("stage") != "blocking").select(
        "exp_id", "stage", "description", "tier", "macro_f05", "micro_precision", "micro_recall", "f05_singletons", "decision", "notes"
    ).write_csv(REPORTS / "ablation_results.csv")


def error_analysis():
    V = pl.read_parquet(CACHE / args.val_pred)
    u = load_folds().filter(pl.col("fold") == 0).select(q="idx")
    G = load_gt_pairs().join(u, on="q", how="semi")
    pred = expected_f(exclusive(V.select("q", "t", "p")), min_p=0.6, extra_mass=0.2).select("q", "t")
    bd = loss_breakdown(pred, V.select("q", "t"), G, u)
    single = u.join(G.group_by("q").len(), on="q", how="left").select("q", is_singleton=pl.col("len").is_null())
    F = (pl.scan_parquet(CACHE / "pairs" / f"train_{args.feat_name}" / "*.parquet").join(u.lazy(), on="q", how="semi")
         .select("q", "t", "n_tset", "n_skel_tset", "a_tset", "num_conflict", "t_addr_missing", "name_freq_t",
                 "t_nonascii_name", "n_nospace_partial").collect()
         .with_columns(n_domain_flag=(pl.col("n_nospace_partial") >= 0).cast(pl.Int8)))
    D = (V.join(F, on=["q", "t"]).join(pred.with_columns(pred=pl.lit(True)), on=["q", "t"], how="left")
         .with_columns(pl.col("pred").fill_null(False)).join(single, on="q"))
    D = categorize_pairs(D).filter(pl.col("category").is_not_null())
    # impact: each FP/FN pair's share of its S1's F loss is approximated by count; also report mean p
    s = (D.group_by("category").agg(n_pairs=pl.len(), n_s1=pl.col("q").n_unique(), mean_p=pl.col("p").mean())
         .with_columns(share_of_errors=pl.col("n_pairs") / pl.col("n_pairs").sum()).sort("n_pairs", descending=True))
    # blocking misses (true pairs not in candidates) are FN too
    miss = G.join(V.select("q", "t"), on=["q", "t"], how="anti")
    s = pl.concat([s, pl.DataFrame({"category": ["FN: candidate blocker failure (not in top-12)"], "n_pairs": [miss.height],
                                    "n_s1": [miss["q"].n_unique()], "mean_p": [None], "share_of_errors": [None]},
                                   schema=s.schema)]).sort("n_pairs", descending=True)
    s.write_csv(REPORTS / "error_analysis.csv")
    ex = D.join(load_source("train", "source1").select(q="idx", s1_name="business_name", s1_addr="business_address"), on="q")
    T = pl.concat([load_source("train", "source2"), load_source("train", "source3")]).with_row_index("t").select(
        pl.col("t").cast(pl.Int32), t_name="business_name", t_addr="business_address")
    ex = ex.join(T, on="t")
    html = ["<html><head><meta charset='utf-8'><title>Error analysis</title><style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}"
            "th{background:#f3f3f3}</style></head><body><h1>Validation error analysis (fold 0)</h1>",
            "<h2>Loss decomposition (share of 1 - macro F0.5)</h2><table><tr><th>S1 error type</th><th>#S1</th><th>loss</th><th>of which blocking ceiling</th></tr>"]
    for r in bd:
        html.append(f"<tr><td>{r['category']}</td><td>{r['n_s1']:,}</td><td>{r['loss']:.5f}</td><td>{r['blocking_ceiling_loss']:.5f}</td></tr>")
    html.append("</table><h2>Pair error categories</h2><table><tr><th>category</th><th>#pairs</th><th>#S1</th><th>mean p</th></tr>")
    for r in s.to_dicts():
        mp = "" if r["mean_p"] is None else f"{r['mean_p']:.3f}"
        html.append(f"<tr><td>{r['category']}</td><td>{r['n_pairs']:,}</td><td>{r['n_s1']:,}</td><td>{mp}</td></tr>")
    html.append("</table><h2>Examples</h2>")
    for cat in s["category"].to_list()[:12]:
        sub = ex.filter(pl.col("category") == cat).sample(n=min(6, ex.filter(pl.col("category") == cat).height), seed=0)
        if sub.height == 0:
            continue
        html.append(f"<h3>{cat}</h3><table><tr><th>S1 name</th><th>S1 address</th><th>target name</th><th>target address</th><th>p</th></tr>")
        for r in sub.to_dicts():
            html.append("<tr>" + "".join(f"<td>{(r[k] or '')}</td>" for k in ("s1_name", "s1_addr", "t_name", "t_addr")) + f"<td>{r['p']:.3f}</td></tr>")
        html.append("</table>")
    (REPORTS / "error_analysis.html").write_text("\n".join(html) + "</body></html>")
    (REPORTS / "loss_breakdown.json").write_text(json.dumps(bd, indent=2))


if not args.skip_dataset:
    dataset_report()
experiment_tables()
error_analysis()
print("reports written", flush=True)
