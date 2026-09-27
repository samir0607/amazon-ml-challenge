"""Train/evaluate the two-stage GBDT matcher on the train split.

Steps (each cached):
 1. retrieval union + label
 2. pruner (retrieval features only) trained on folds 1-4 sample; choose top-N
 3. pair features for the pruned candidates of all train S1
 4. stage-1 LightGBM, cross-fitted by fold -> OOF p1 for every train pair
 5. context features from p1 (per-S1 and per-target competition)
 6. stage-2 LightGBM on folds 1-4, evaluated on fold 0 (screen + full)
 7. decision-rule sweep on fold 0 (macro F0.5)
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import candidates as cand
from ber.blocking.base import eval_candidates
from ber.data import load_gt_pairs, target_meta
from ber.decide import exclusive, expected_f, threshold
from ber.experiment import Experiment
from ber.siblings import sibling_features
from ber.stage2 import STAGE2_EXTRA, context_frame
from ber.metrics import macro_f05
from ber.pairs import MODEL_FEATURES_BASE, compute_pair_features
from ber.paths import CACHE
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--top-n", type=int, default=12)
ap.add_argument("--pruner-s1", type=int, default=150_000)
ap.add_argument("--train-s1", type=int, default=300_000, help="stage-1 S1 sample per fold model")
ap.add_argument("--s2-train-s1", type=int, default=1_200_000)
ap.add_argument("--easy-neg-keep", type=float, default=0.25)
ap.add_argument("--tag", default="v1")
ap.add_argument("--final", action="store_true")
ap.add_argument("--rule-min-p", type=float, default=0.6, help="min_p fixed for the final expected-F rule (0.6 chosen for singleton robustness)")
ap.add_argument("--feat-tag", default=None, help="reuse candidates/pair features of this tag")
args = ap.parse_args()
args.feat_tag = args.feat_tag or args.tag

folds = load_folds().select(q="idx", fold="fold", screen="screen")
G = load_gt_pairs().with_columns(y=pl.lit(1, pl.Int8))
n2 = int((target_meta("train")["src"] == 2).sum())
rng = np.random.default_rng(0)


def label(df):
    return df.join(G, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))


def sample_q(fold_mask, n, seed):
    qs = folds.filter(fold_mask)["q"].to_numpy()
    return np.random.default_rng(seed).choice(qs, min(n, len(qs)), replace=False)


def cached_stage2(p1):
    path = CACHE / f"s2x_train_{args.tag}_top{args.top_n}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    d = context_frame(p1).join(sibling_features("train", p1), on=["q", "t"], how="left")
    d.write_parquet(path)
    return d


# ---------------------------------------------------------------- 1-2 retrieval + pruning
pruned_path = CACHE / f"cands_train_{args.feat_tag}_top{args.top_n}.parquet"
if pruned_path.exists():
    C = pl.read_parquet(pruned_path)
else:
    qtr = sample_q(pl.col("fold") != 0, args.pruner_s1, 1)
    Utr = label(cand.retrieval_frame("train", qs=qtr))
    booster = cand.train_pruner(Utr, Utr["y"].to_numpy())
    del Utr
    booster.save_model(str(CACHE / f"pruner_{args.feat_tag}.txt"))
    C = cand.build_pruned("train", booster, args.top_n)
    qs_screen = folds.filter("screen")["q"].to_numpy()
    ex = Experiment(f"block_final_{args.tag}_top{args.top_n}", "blocking",
                    {"specs": cand.RETR_SPECS, "top_n": args.top_n, "pruner_s1": args.pruner_s1},
                    f"final candidate set (union pruned top{args.top_n})", "fold0")
    ex.log(**eval_candidates(C, G, folds.filter(pl.col("fold") == 0)["q"].to_numpy(), n2)); ex.finish()
    print("final candidates fold0:", ex.metrics, flush=True)
    C = label(C)
    C.write_parquet(pruned_path)

# ---------------------------------------------------------------- 3 pair features
feat_name = f"{args.feat_tag}_top{args.top_n}"
pair_name = f"{args.tag}_top{args.top_n}"  # model / score artifacts
compute_pair_features("train", C.drop("y"), feat_name)
del C
PARTS = CACHE / "pairs" / f"train_{feat_name}" / "*.parquet"
lazy = pl.scan_parquet(PARTS)
cols = lazy.collect_schema().names()
retr = list(dict.fromkeys(cand.retr_feature_cols(pl.DataFrame(schema=cols)) + ["pr", "pr_rank"]))
BASE = MODEL_FEATURES_BASE + retr
print(f"{len(BASE)} base features", flush=True)


def rows_for(qs):
    """Labelled feature rows for a set of S1 q (materialized)."""
    qdf = pl.DataFrame({"q": np.asarray(qs, dtype=np.int32)})
    return label(lazy.join(qdf.lazy(), on="q", how="semi").collect())


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=9)


def fit(df, feats, seed, rounds=3000):
    qs = df["q"].unique().to_numpy()
    es_q = np.random.default_rng(seed).choice(qs, len(qs) // 10, replace=False)
    is_es = df["q"].is_in(es_q).to_numpy()
    X, y = df.select(feats).to_numpy().astype(np.float32), df["y"].to_numpy()
    w = df["w"].to_numpy() if "w" in df.columns else np.ones(len(y), np.float32)
    dtr = lgb.Dataset(X[~is_es], y[~is_es], weight=w[~is_es], feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(X[is_es], y[is_es], weight=w[is_es], reference=dtr)
    return lgb.train({**PARAMS, "seed": seed}, dtr, rounds, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def predict(b, df, feats):
    return b.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=9)


# ---------------------------------------------------------------- 4 stage-1 OOF
p1_path = CACHE / f"p1_train_{pair_name}.parquet"
if p1_path.exists():
    P1 = pl.read_parquet(p1_path)
else:
    models = []
    for f in range(5):
        mp = CACHE / f"stage1_{pair_name}_fold{f}.txt"
        if mp.exists():
            models.append(lgb.Booster(model_file=str(mp)))
            continue
        t0 = time.time()
        tr = rows_for(sample_q(pl.col("fold") != f, args.train_s1, 10 + f))
        b = fit(tr, BASE, f)
        del tr
        b.save_model(str(mp))
        models.append(b)
        print(f"stage1 fold {f}: {b.best_iteration} it, {time.time() - t0:.0f}s", flush=True)
    parts = []
    fmap = folds.select("q", "fold")
    for path in sorted((CACHE / "pairs" / f"train_{feat_name}").glob("*.parquet")):
        d = pl.read_parquet(path).join(fmap, on="q", how="left")
        p = np.zeros(d.height, dtype=np.float32)
        fo = d["fold"].to_numpy()
        for f in range(5):
            m = fo == f
            if m.any():
                p[m] = predict(models[f], d.filter(pl.Series(m)), BASE)
        parts.append(d.select("q", "t").with_columns(p1=pl.Series(p)))
    P1 = pl.concat(parts)
    P1.write_parquet(p1_path)

# ---------------------------------------------------------------- 5 context + siblings (ALL train pairs)
S2X = cached_stage2(P1)
FEATS2 = BASE + STAGE2_EXTRA


def rows2(qs, seed):
    """Stage-2 rows; easy negatives (p1 < 0.005) subsampled with inverse-prob weights."""
    d = rows_for(qs).join(S2X, on=["q", "t"], how="left")
    easy = (d["y"] == 0) & (d["p1"] < 0.005)
    keep = ~easy | pl.Series(np.random.default_rng(seed).random(d.height) < args.easy_neg_keep)
    return d.filter(keep).with_columns(w=pl.when(easy.filter(keep)).then(1.0 / args.easy_neg_keep).otherwise(1.0).cast(pl.Float32))


# ---------------------------------------------------------------- 6 stage-2
tr = rows2(sample_q(pl.col("fold") != 0, args.s2_train_s1, 99), 0)
b2 = fit(tr, FEATS2, 0)
del tr
b2.save_model(str(CACHE / f"stage2_{pair_name}_fold0.txt"))
V = rows_for(folds.filter(pl.col("fold") == 0)["q"].to_numpy()).join(S2X, on=["q", "t"], how="left")
V = V.with_columns(p=pl.Series(predict(b2, V, FEATS2).astype(np.float32)))
V.select("q", "t", "y", "p", "p1", "src").write_parquet(CACHE / f"val_pred_{pair_name}.parquet")
imp = sorted(zip(FEATS2, b2.feature_importance("gain")), key=lambda x: -x[1])[:25]
print("top features:", [(a, int(b)) for a, b in imp], flush=True)

# ---------------------------------------------------------------- 7 decisions on fold 0
s1_id = pl.read_parquet(CACHE / "folds.parquet").select(q="idx", s1_id="entity_id", fold="fold")
val_ids = s1_id.filter(pl.col("fold") == 0)
gt_val = val_ids.join(G, on="q", how="left").select("s1_id", match_id=pl.col("t"))


def score(pred, tag, extra=None):
    pr = pred.join(val_ids, on="q").select("s1_id", match_id=pl.col("t"))
    m = macro_f05(pr, gt_val, val_ids["s1_id"])
    return m


results = []
for which in ("p1", "p"):
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for excl in (False, True):
            d = V.select("q", "t", p=pl.col(which))
            d = threshold(d, thr)
            if excl:
                d = exclusive(d)
            m = score(d, "")
            results.append({"score": which, "rule": "thr", "thr": thr, "excl": excl, **m})
    for excl in (False, True):
        for mp, em in ((0.0, 0.0), (0.5, 0.2), (0.6, 0.2)):
            d = V.select("q", "t", p=pl.col(which))
            if excl:
                d = exclusive(d)
            m = score(expected_f(d, min_p=mp, extra_mass=em), "")
            results.append({"score": which, "rule": "expF", "thr": None, "excl": excl, "min_p": mp, "extra_mass": em, **m})
res = pl.DataFrame(results).sort("macro_f05", descending=True)
print(res.select("score", "rule", "thr", "excl", "macro_f05", "micro_precision", "micro_recall",
                 "f05_singletons", "singleton_false_match_rate").head(12), flush=True)
best = res.row(0, named=True)
best = {**best, "min_p": best.get("min_p") or 0.0, "extra_mass": best.get("extra_mass") or 0.0}
if best["rule"] == "expF" and args.rule_min_p is not None:
    # min_p 0.5 and 0.6 tie on F0.5 (0.97207 vs 0.97205) but 0.6 is much better on
    # singletons (0.969 vs 0.959), so the final rule fixes it for robustness.
    best = {**best, "min_p": args.rule_min_p, "extra_mass": 0.2}
ex = Experiment(f"matcher_{args.tag}", "matching",
                {**vars(args), "params": PARAMS, "features": FEATS2, "best_rule": {k: best[k] for k in ("score", "rule", "thr", "excl", "min_p", "extra_mass")}},
                "two-stage LightGBM + context features", tier="fold0")
ex.log(**{k: v for k, v in best.items() if isinstance(v, (int, float))})
ex.save_json("decision_sweep.json", results)
ex.save_json("feature_importance.json", imp)
ex.finish()

# ---------------------------------------------------------------- 8 final stage-2 on all folds
if args.final:
    tr = rows2(sample_q(pl.col("fold") >= 0, args.s2_train_s1, 123), 1)
    bf = fit(tr, FEATS2, 7)
    bf.save_model(str(CACHE / f"stage2_{pair_name}_final.txt"))
    (CACHE / f"final_{pair_name}.json").write_text(json.dumps(
        {"base": BASE, "feats2": FEATS2, "rule": {k: best[k] for k in ("score", "rule", "thr", "excl", "min_p", "extra_mass")},
         "top_n": args.top_n, "tag": args.tag, "feat_tag": args.feat_tag}))
    print("saved final stage-2", flush=True)
