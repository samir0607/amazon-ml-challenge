"""Cross-validated stage-2 predictions for decision-rule validation.

For each fold f: train stage-2 (same params / features / easy-negative subsampling as
train_matcher.py stage 2) on a sample of S1 from folds != f, using the cached pair
features + OOF p1 + s2x context/sibling features, then predict every candidate pair of
fold f. Writes cache/cv_pred_v2_fold{f}.parquet (q, t, y, p, p1, src).

Run through heavy_lock.py (peak RAM ~4 GB, ~10 min per fold).
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.data import load_gt_pairs
from ber.paths import CACHE
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--s2-train-s1", type=int, default=300_000)
ap.add_argument("--easy-neg-keep", type=float, default=0.25)
ap.add_argument("--name", default="v2_top12")
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()

cfg = json.loads((CACHE / f"final_{args.name}.json").read_text())
BASE, FEATS2 = cfg["base"], cfg["feats2"]
feat_name = f"{cfg['feat_tag']}_top{cfg['top_n']}"
S2X_PATH = CACHE / f"s2x_train_{args.name}.parquet"
s2x_cols = [c for c in pl.scan_parquet(S2X_PATH).collect_schema().names() if c not in ("q", "t")]
pair_cols = [c for c in BASE if c not in s2x_cols]
assert all(c in BASE or c in s2x_cols for c in FEATS2)
PAIRS = pl.scan_parquet(CACHE / "pairs" / f"train_{feat_name}" / "*.parquet").select(["q", "t"] + pair_cols)
S2X = pl.scan_parquet(S2X_PATH)
folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs().with_columns(y=pl.lit(1, pl.Int8))

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=args.threads)


def sample_q(fold_mask, n, seed):
    qs = folds.filter(fold_mask)["q"].to_numpy()
    return np.random.default_rng(seed).choice(qs, min(n, len(qs)), replace=False)


def rows(qs):
    qdf = pl.DataFrame({"q": np.asarray(qs, dtype=np.int32)}).lazy()
    d = (PAIRS.join(qdf, on="q", how="semi")
         .join(S2X.join(qdf, on="q", how="semi"), on=["q", "t"], how="left").collect())
    return d.join(G, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))


def rows2(qs, seed):
    d = rows(qs)
    easy = (d["y"] == 0) & (d["p1"] < 0.005)
    keep = ~easy | pl.Series(np.random.default_rng(seed).random(d.height) < args.easy_neg_keep)
    return d.filter(keep).with_columns(
        w=pl.when(easy.filter(keep)).then(1.0 / args.easy_neg_keep).otherwise(1.0).cast(pl.Float32))


def fit(df, feats, seed, rounds=3000):
    qs = df["q"].unique().to_numpy()
    es_q = np.random.default_rng(seed).choice(qs, len(qs) // 10, replace=False)
    is_es = df["q"].is_in(es_q).to_numpy()
    X, y = df.select(feats).to_numpy().astype(np.float32), df["y"].to_numpy()
    w = df["w"].to_numpy()
    del df
    dtr = lgb.Dataset(X[~is_es], y[~is_es], weight=w[~is_es], feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(X[is_es], y[is_es], weight=w[is_es], reference=dtr)
    return lgb.train({**PARAMS, "seed": seed}, dtr, rounds, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


for f in [int(x) for x in args.folds.split(",")]:
    out = CACHE / f"cv_pred_v2_fold{f}.parquet"
    if out.exists():
        print(f"fold {f}: exists", flush=True)
        continue
    t0 = time.time()
    mp = CACHE / f"cv_stage2_v2_fold{f}.txt"
    if mp.exists():
        b = lgb.Booster(model_file=str(mp))
    else:
        tr = rows2(sample_q(pl.col("fold") != f, args.s2_train_s1, 99 + f), f)
        print(f"fold {f}: train rows {tr.height}, pos {int(tr['y'].sum())}", flush=True)
        b = fit(tr, FEATS2, f)
        del tr
        b.save_model(str(mp))
        print(f"fold {f}: {b.best_iteration} it, {time.time() - t0:.0f}s", flush=True)
    qv = folds.filter(pl.col("fold") == f)["q"].to_numpy()
    parts = []
    for i in range(0, len(qv), 100_000):
        d = rows(qv[i:i + 100_000])
        p = b.predict(d.select(FEATS2).to_numpy().astype(np.float32), num_threads=args.threads)
        parts.append(d.select("q", "t", "y", "p1", "src").with_columns(p=pl.Series(p.astype(np.float32))))
        del d
    V = pl.concat(parts)
    V.write_parquet(out)
    print(f"fold {f}: predicted {V.height} pairs, {time.time() - t0:.0f}s", flush=True)
