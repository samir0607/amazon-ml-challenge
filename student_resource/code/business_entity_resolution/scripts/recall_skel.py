"""Recall study: what does the reverse skeleton-char blocker on non-ASCII target names
(rev_skel_nonascii, k_use=1..3) add on top of the 5-blocker union?

Screening set only (50k S1 of fold 0), full target index. Steps (cached under cache/recall/):
 1. 6-blocker retrieval frame for screen S1 and for the pruner sample (150k S1 of folds
    1-4, same seed as train_matcher) -> variants k=0 (baseline) / 1 / 2 / 3.
 2. union recall per variant; pruner per variant (k=0 reuses pruner_v1.txt, which was
    trained on the same sample); recall at top-12/13/14.
 3. pair features for pairs in any variant's top-14 that are NOT in the current
    top-12 set; score with the OOF fold-0 stage-1 model.
 4. cascade counts (p1>=0.01 survivors) and an F0.5 delta estimate on top of the
    current fold-0 stage-2 decisions.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import candidates as cand
from ber.blocking.base import eval_candidates
from ber.data import load_gt_pairs, target_meta
from ber.decide import exclusive, expected_f
from ber.pairs import compute_pair_features
from ber.paths import CACHE
from ber.splits import load_folds

OUT = CACHE / "recall"
OUT.mkdir(exist_ok=True)
SKEL = {"name": "rev_skel_nonascii", "kind": "rev", "cfg": "skel_char", "k": 3, "subset": "nonascii_name"}
SPEC6 = cand.RETR_SPECS + [SKEL]
N5 = [f"{s['name']}_score" for s in cand.RETR_SPECS]
KS = (0, 1, 2, 3)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


folds = load_folds().select(q="idx", fold="fold", screen="screen")
G = load_gt_pairs()
GY = G.with_columns(y=pl.lit(1, pl.Int8))
n2 = int((target_meta("train")["src"] == 2).sum())
qs_screen = folds.filter("screen")["q"].to_numpy()
qs_pr = np.random.default_rng(1).choice(folds.filter(pl.col("fold") != 0)["q"].to_numpy(), 150_000, replace=False)


def label(df):
    return df.join(GY, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))


def frame6(tag, qs):
    p = OUT / f"u6_{tag}.parquet"
    if not p.exists():
        cand.retrieval_frame("train", SPEC6, qs=qs).write_parquet(p)
    return pl.read_parquet(p)


def variant(u6: pl.DataFrame, k: int) -> pl.DataFrame:
    """5-blocker union + skel pairs with per-target rank <= k (k=0: no skel)."""
    n5 = pl.sum_horizontal([pl.col(c).is_not_null() for c in N5]).cast(pl.Int8)
    ms = pl.col("rev_skel_nonascii_rank").fill_null(99) <= k
    d = u6.with_columns(_n5=n5, _ms=ms).filter((pl.col("_n5") > 0) | pl.col("_ms"))
    if k == 0:
        d = d.drop("rev_skel_nonascii_score", "rev_skel_nonascii_rank")
    else:
        d = d.with_columns([pl.when(pl.col("_ms")).then(pl.col(c)).otherwise(None).alias(c)
                            for c in ("rev_skel_nonascii_score", "rev_skel_nonascii_rank")])
    d = d.with_columns(n_blockers=(pl.col("_n5") + pl.col("_ms").cast(pl.Int8)).cast(pl.Int8),
                       n5=pl.col("_n5"), r_q_n=pl.len().over("q").cast(pl.Float32)).drop("_n5", "_ms")
    return d


def pruner(k, utr):
    if k == 0:
        return lgb.Booster(model_file=str(CACHE / "pruner_v1.txt"))
    p = OUT / f"pruner_skel_k{k}.txt"
    if p.exists():
        return lgb.Booster(model_file=str(p))
    d = label(variant(utr, k))
    b = cand.train_pruner(d.drop("n5"), d["y"].to_numpy())
    b.save_model(str(p))
    return b


res = {}
# ------------------------------------------------------------------ 1-2 union + prune
old = pl.read_parquet(CACHE / "cands_train_v1_top12.parquet", columns=["q", "t"]).join(
    pl.DataFrame({"q": qs_screen.astype(np.int32)}), on="q", how="semi")
res["old_top12"] = eval_candidates(old, G, qs_screen, n2)
log("old top12 screen:", res["old_top12"])
us = frame6("screen", qs_screen)
log("screen u6", us.shape)
pr_paths = {k: OUT / f"pruned_screen_k{k}.parquet" for k in KS}
if not all(p.exists() for p in pr_paths.values()):
    utr = frame6("prtrain", qs_pr)
    log("prtrain u6", utr.shape)
    boosters = {k: pruner(k, utr) for k in KS}
    del utr
    for k in KS:
        v = variant(us, k)
        cols = boosters[k].feature_name()
        p = boosters[k].predict(v.select(cols).to_numpy().astype(np.float32), num_threads=8)
        v = v.with_columns(pr=pl.Series(p.astype(np.float32)))
        v = v.with_columns(pr_rank=pl.col("pr").rank("ordinal", descending=True).over("q").cast(pl.Int16))
        v.filter(pl.col("pr_rank") <= 14).write_parquet(pr_paths[k])
        res[f"union_k{k}"] = eval_candidates(v, G, qs_screen, n2)
        log(f"k={k} union:", res[f"union_k{k}"])
PR = {k: pl.read_parquet(pr_paths[k]) for k in KS}
for k in KS:
    for n in (12, 13, 14):
        res[f"pruned_k{k}_top{n}"] = eval_candidates(PR[k].filter(pl.col("pr_rank") <= n), G, qs_screen, n2)
        log(f"k={k} top{n}:", {a: round(b, 5) for a, b in res[f"pruned_k{k}_top{n}"].items() if a in ("pair_recall", "entity_full_coverage", "cand_mean")})
if "union_k0" not in res:
    for k in KS:
        res[f"union_k{k}"] = eval_candidates(variant(us, k), G, qs_screen, n2)
        log(f"k={k} union:", {a: round(b, 5) for a, b in res[f"union_k{k}"].items() if a in ("pair_recall", "cand_mean")})
# union misses (5-blocker) that skel recovers
miss5 = G.join(pl.DataFrame({"q": qs_screen.astype(np.int32)}), on="q", how="semi").join(
    variant(us, 0).select("q", "t"), on=["q", "t"], how="anti")
res["n_true_screen"] = G.join(pl.DataFrame({"q": qs_screen.astype(np.int32)}), on="q", how="semi").height
res["n_union5_miss"] = miss5.height
for k in (1, 2, 3):
    res[f"union5_miss_recovered_k{k}"] = miss5.join(variant(us, k).filter(pl.col("n5") == 0).select("q", "t"), on=["q", "t"], how="semi").height
del us

# ------------------------------------------------------------------ 3 features for new pairs
new = pl.concat([PR[k].select("q", "t") for k in (1, 2, 3)]).unique().join(old, on=["q", "t"], how="anti")
log("new pairs (any variant, top14, not in old top12):", new.height)
F = compute_pair_features("train", new.sort(["q", "t"]), "recall_skel_screen_new")
F = F.select([c for c in F.columns if c == "q" or c == "t" or c not in PR[1].columns])
m0 = lgb.Booster(model_file=str(CACHE / "stage1_v2_top12_fold0.txt"))
BASE = json.loads((CACHE / "final_v2_top12.json").read_text())["base"]

# baseline decisions on screen: fold-0 stage-2 p + final rule
V = pl.read_parquet(CACHE / "val_pred_v2_top12.parquet").join(pl.DataFrame({"q": qs_screen.astype(np.int32)}), on="q", how="semi")
P1old = pl.read_parquet(CACHE / "p1_train_v2_top12.parquet").join(pl.DataFrame({"q": qs_screen.astype(np.int32)}), on="q", how="semi")
dec0 = expected_f(exclusive(V.select("q", "t", "p")), min_p=0.6, extra_mass=0.2)
GS = G.join(pl.DataFrame({"q": qs_screen.astype(np.int32)}), on="q", how="semi")


def macro_f(dec):
    tp = dec.join(GS, on=["q", "t"], how="semi").group_by("q").agg(tp=pl.len())
    d = (pl.DataFrame({"q": qs_screen.astype(np.int32)}).join(tp, on="q", how="left")
         .join(dec.group_by("q").agg(np_=pl.len()), on="q", how="left")
         .join(GS.group_by("q").agg(ng=pl.len()), on="q", how="left").fill_null(0))
    tp_, np_, ng_ = (d[c].to_numpy().astype(float) for c in ("tp", "np_", "ng"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(np_ > 0, tp_ / np_, 0)
        r = np.where(ng_ > 0, tp_ / ng_, 0)
        f = np.where(tp_ > 0, 1.25 * p * r / (0.25 * p + r), 0)
    return float(np.where(ng_ == 0, (np_ == 0).astype(float), f).mean())


res["F_base_screen"] = macro_f(dec0)
log("baseline F0.5 screen:", res["F_base_screen"])

for k in (1, 2, 3):
    for n in (12, 13, 14):
        tag = f"k{k}_top{n}"
        cur = PR[k].filter(pl.col("pr_rank") <= n)
        nk = cur.join(old, on=["q", "t"], how="anti").join(F, on=["q", "t"], how="left")
        nk = nk.with_columns(n_blockers=pl.col("n5"))  # stage-1 knows only the 5 blockers
        nk = nk.with_columns(p1=pl.Series(m0.predict(nk.select(BASE).to_numpy().astype(np.float32), num_threads=8).astype(np.float32)))
        nk = label(nk)
        dropped = old.join(cur.select("q", "t"), on=["q", "t"], how="anti").join(P1old, on=["q", "t"], how="left")
        dropped = label(dropped)
        r = {"new_pairs": nk.height, "new_true": int(nk["y"].sum()),
             "new_skel_only": int((nk["n5"] == 0).sum()), "new_true_skel_only": int(nk.filter(pl.col("n5") == 0)["y"].sum())}
        for th in (0.01, 0.02, 0.03, 0.1, 0.5, 0.6):
            r[f"true_p1>={th}"] = int(nk.filter((pl.col("y") == 1) & (pl.col("p1") >= th)).height)
            r[f"false_p1>={th}"] = int(nk.filter((pl.col("y") == 0) & (pl.col("p1") >= th)).height)
        r["dropped_pairs"] = dropped.height
        r["dropped_true"] = int(dropped["y"].sum())
        for th in (0.01, 0.5):
            r[f"dropped_true_p1>={th}"] = int(dropped.filter((pl.col("y") == 1) & (pl.col("p1") >= th)).height)
            r[f"dropped_false_p1>={th}"] = int(dropped.filter((pl.col("y") == 0) & (pl.col("p1") >= th)).height)
        r["s1_recovered_p1>=0.6"] = int(nk.filter((pl.col("y") == 1) & (pl.col("p1") >= 0.6))["q"].n_unique())
        # F estimate: drop decisions on dropped pairs, add new pairs with p1 >= thr, re-apply exclusivity
        base = dec0.join(dropped.select("q", "t"), on=["q", "t"], how="anti").select("q", "t", "p")
        for th in (0.5, 0.6, 0.7, 0.8):
            add = nk.filter(pl.col("p1") >= th).select("q", "t", p=pl.col("p1"))
            r[f"dF_add_p1>={th}"] = macro_f(exclusive(pl.concat([base, add]))) - res["F_base_screen"]
        r["dF_drop_only"] = macro_f(base) - res["F_base_screen"]
        res[tag] = r
        log(tag, r)
        if n == 12:
            nk.select("q", "t", "y", "p1", "pr", "pr_rank", "n5", "rev_skel_nonascii_rank").write_parquet(OUT / f"new_scored_{tag}.parquet")

(OUT / "recall_skel_results.json").write_text(json.dumps(res, indent=1, default=float))
log("DONE")
