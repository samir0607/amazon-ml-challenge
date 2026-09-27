"""Frozen-pipeline inference on the test split -> output/candidate_pairs.tsv and
output/matching_results.tsv. Uses models produced by train_matcher.py --final."""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import candidates as cand
from ber.decide import exclusive, expected_f, threshold
from ber.io import load_source
from ber.pairs import compute_pair_features
from ber.paths import CACHE, OUTPUT
from ber.stage2 import stage2_frame

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v2_top12", help="<tag>_top<N> of the trained pipeline")
ap.add_argument("--from-scores", action="store_true", help="reuse cached test scores (decision/output only)")
args = ap.parse_args()
cfg = json.loads((CACHE / f"final_{args.name}.json").read_text())
feat_tag = cfg.get("feat_tag", cfg["tag"])
feat_name = f"{feat_tag}_top{cfg['top_n']}"

# ---- candidates (union -> prune): exactly the set the matcher scores
path = CACHE / f"cands_test_{feat_name}.parquet"
if path.exists():
    C = pl.read_parquet(path)
else:
    pruner = lgb.Booster(model_file=str(CACHE / f"pruner_{feat_tag}.txt"))
    C = cand.build_pruned("test", pruner, cfg["top_n"])
    C.write_parquet(path)

s1 = load_source("test", "source1").select(q="idx", s1_id="entity_id")
tid = pl.concat([load_source("test", "source2").select("entity_id"),
                 load_source("test", "source3").select("entity_id")]).with_row_index("t").with_columns(pl.col("t").cast(pl.Int32))


def write_lists(pairs: pl.DataFrame, col: str, out):
    lists = pairs.select("q", "t").unique().join(tid, on="t").group_by("q").agg(pl.col("entity_id").sort().str.join(","))
    d = s1.join(lists.rename({"entity_id": col}), on="q", how="left").with_columns(pl.col(col).fill_null("")).sort("q")
    d.select(pl.col("s1_id").alias("source1_entity_id"), pl.col(col)).write_csv(out, separator="\t", quote_style="never")


OUTPUT.mkdir(exist_ok=True)
tau = cfg.get("cascade_tau", 0.0)

score_path = CACHE / f"test_scores_{args.name}.parquet"
if args.from_scores and score_path.exists():
    F = pl.read_parquet(score_path)
else:
    # ---- features + two-stage scoring (streamed by part to bound memory)
    compute_pair_features("test", C, f"test_{feat_name}")
    parts = sorted((CACHE / "pairs" / f"test_test_{feat_name}").glob("*.parquet"))
    BASE, FEATS2 = cfg["base"], cfg["feats2"]
    p1_path = CACHE / f"test_p1_{args.name}.parquet"
    if p1_path.exists():
        P1 = pl.read_parquet(p1_path)
    else:
        s1_models = [lgb.Booster(model_file=str(CACHE / f"stage1_{args.name}_fold{f}.txt")) for f in range(5)]
        p1 = []
        for pth in parts:
            d = pl.read_parquet(pth)
            X = d.select(BASE).to_numpy().astype(np.float32)
            p1.append(d.select("q", "t").with_columns(
                p1=pl.Series(np.mean([m.predict(X, num_threads=9) for m in s1_models], axis=0).astype(np.float32))))
        P1 = pl.concat(p1)
        P1.write_parquet(p1_path)
    P1 = P1.filter(pl.col("p1") >= tau) if tau > 0 else P1  # learned-blocking cascade
    S2X = stage2_frame("test", P1)
    b2 = lgb.Booster(model_file=str(CACHE / f"stage2_{args.name}_final.txt"))
    wcat = cfg.get("blend_cat", 0.0)
    cat = None
    if wcat > 0:
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier()
        cat.load_model(str(CACHE / f"stage2_{args.name}_final_cat.cbm"))
    scored = []
    for pth in parts:
        d = pl.read_parquet(pth).join(S2X, on=["q", "t"], how="inner")
        X = d.select(FEATS2).to_numpy().astype(np.float32)
        p = b2.predict(X, num_threads=9)
        if cat is not None:
            p = (1 - wcat) * p + wcat * cat.predict_proba(X)[:, 1]
        scored.append(d.select("q", "t", "p1").with_columns(p=pl.Series(p.astype(np.float32))))
    F = pl.concat(scored)
    F.write_parquet(CACHE / f"test_scores_{args.name}.parquet")

# ---- final candidate set = exactly the pairs stage 2 scored (after the cascade)
write_lists(F, "candidate_entity_ids", OUTPUT / "candidate_pairs.tsv")
print(f"candidates after cascade (p1 >= {tau}): {F.height} pairs, {F.height / s1.height:.2f} per S1", flush=True)

# ---- decision rule chosen on validation
rule = cfg["rule"]
d = F.select("q", "t", p=pl.col(rule["score"]))
if rule["excl"]:
    d = exclusive(d)
if rule["rule"] == "expF":
    d = expected_f(d, min_p=rule.get("min_p") or 0.0, extra_mass=rule.get("extra_mass") or 0.0)
else:
    d = threshold(d, rule["thr"])
M = d.join(F.select("q", "t"), on=["q", "t"], how="semi")  # matches must be candidates
assert M.height == d.height
write_lists(M, "matched_entity_ids", OUTPUT / "matching_results.tsv")
by_c = M.join(load_source("test", "source1").select(q="idx", country="country"), on="q").group_by("country").agg(
    n_pairs=pl.len(), n_s1=pl.col("q").n_unique())
print("matches:", M.height, "pairs; S1 with >=1 match:", M["q"].n_unique(), "/", s1.height, flush=True)
print(by_c, flush=True)
