"""Write output/matching_results.tsv from cross-encoder-fused test scores
(cache/test_scores_ce_lite.parquet, produced by ce_lite.py). The candidate set is
unchanged: the cross-encoder only re-scores pairs that are already candidates."""
import json

import polars as pl

from ber.decide import exclusive, expected_f
from ber.io import load_source
from ber.paths import CACHE, OUTPUT

cfg = json.loads((CACHE / "final_v2_top12.json").read_text())
rule = cfg["rule"]
F = pl.read_parquet(CACHE / "test_scores_ce_lite.parquet").select("q", "t", "p")
cands = pl.read_parquet(CACHE / "test_scores_v2_top12.parquet").select("q", "t")
assert F.height == cands.height, "fused scores must cover exactly the candidate set"
d = expected_f(exclusive(F), min_p=rule["min_p"], extra_mass=rule["extra_mass"])
s1 = load_source("test", "source1").select(q="idx", s1_id="entity_id")
tid = pl.concat([load_source("test", "source2").select("entity_id"),
                 load_source("test", "source3").select("entity_id")]).with_row_index("t").with_columns(pl.col("t").cast(pl.Int32))
lists = d.select("q", "t").join(tid, on="t").group_by("q").agg(pl.col("entity_id").sort().str.join(","))
out = s1.join(lists, on="q", how="left").with_columns(pl.col("entity_id").fill_null("")).sort("q")
out.select(source1_entity_id="s1_id", matched_entity_ids="entity_id").write_csv(
    OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
print(f"matches: {d.height} pairs; S1 with >=1 match: {d['q'].n_unique()} / {s1.height}", flush=True)
