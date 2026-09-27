"""Write output/matching_results.tsv from cross-encoder-fused test scores
(cache/test_scores_ce_lite.parquet, produced by ce_lite.py). The candidate set is
unchanged: the cross-encoder only re-scores pairs that are already candidates."""
import argparse
import json

import polars as pl

from ber.decide import exclusive, expected_f
from ber.io import load_source
from ber.paths import CACHE, OUTPUT

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="full", choices=["addonly", "full"],
                help="full: use the fused probability (best on the leaderboard). addonly: p = max(stage-2 p, "
                     "fused p); scored lower on the leaderboard than full fusion.")
ap.add_argument("--prior-odds", type=float, default=1.0,
                help="multiply every match probability's odds by this factor (<1 = more conservative), to "
                     "account for the test set's higher distractor density")
ap.add_argument("--out", default=str(OUTPUT / "matching_results.tsv"))
ap.add_argument("--scores", default="test_scores_ce_lite.parquet", help="fused test scores in cache/")
args = ap.parse_args()
cfg = json.loads((CACHE / "final_v2_top12.json").read_text())
rule = cfg["rule"]
F = pl.read_parquet(CACHE / args.scores).select("q", "t", "p")
base = pl.read_parquet(CACHE / "test_scores_v2_top12.parquet").select("q", "t", pb="p")
assert F.height == base.height, "fused scores must cover exactly the candidate set"
if args.mode == "addonly":
    F = F.join(base, on=["q", "t"]).select("q", "t", p=pl.max_horizontal("p", "pb"))
if args.prior_odds != 1.0:
    k = args.prior_odds
    F = F.with_columns(p=(pl.col("p") * k / (pl.col("p") * k + (1 - pl.col("p")))).cast(pl.Float32))
d = expected_f(exclusive(F), min_p=rule["min_p"], extra_mass=rule["extra_mass"])
s1 = load_source("test", "source1").select(q="idx", s1_id="entity_id")
tid = pl.concat([load_source("test", "source2").select("entity_id"),
                 load_source("test", "source3").select("entity_id")]).with_row_index("t").with_columns(pl.col("t").cast(pl.Int32))
lists = d.select("q", "t").join(tid, on="t").group_by("q").agg(pl.col("entity_id").sort().str.join(","))
out = s1.join(lists, on="q", how="left").with_columns(pl.col("entity_id").fill_null("")).sort("q")
out.select(source1_entity_id="s1_id", matched_entity_ids="entity_id").write_csv(
    args.out, separator="\t", quote_style="never")
print(f"matches: {d.height} pairs; S1 with >=1 match: {d['q'].n_unique()} / {s1.height}", flush=True)
