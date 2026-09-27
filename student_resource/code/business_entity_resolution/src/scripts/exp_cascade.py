"""Cascade experiment: candidate set = pairs with OOF stage-1 p1 >= tau (learned blocking).
Stage-2 context is recomputed within the surviving candidates only, stage-2 retrained on
survivors of folds 1-4, and evaluated on fold 0 (macro F0.5 + candidate statistics)."""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import candidates as cand
from ber.blocking.base import eval_candidates
from ber.data import load_gt_pairs, target_meta
from ber.decide import exclusive, expected_f
from ber.errors import per_s1
from ber.experiment import Experiment
from ber.pairs import MODEL_FEATURES_BASE
from ber.paths import CACHE
from ber.siblings import SIB_COLS
from ber.splits import load_folds
from ber.stage2 import STAGE2_EXTRA, context_frame

ap = argparse.ArgumentParser()
ap.add_argument("--taus", default="0,0.01,0.03")
ap.add_argument("--train-s1", type=int, default=600_000)
ap.add_argument("--name", default="v2_top12")
ap.add_argument("--feat", default="v1_top12")
args = ap.parse_args()

folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs()
n2 = int((target_meta("train")["src"] == 2).sum())
lazy = pl.scan_parquet(CACHE / "pairs" / f"train_{args.feat}" / "*.parquet")
cols = lazy.collect_schema().names()
BASE = MODEL_FEATURES_BASE + list(dict.fromkeys(cand.retr_feature_cols(pl.DataFrame(schema=cols)) + ["pr", "pr_rank"]))
FEATS2 = BASE + STAGE2_EXTRA
P1 = pl.read_parquet(CACHE / f"p1_train_{args.name}.parquet")
SIB = pl.read_parquet(CACHE / f"s2x_train_{args.name}.parquet", columns=["q", "t"] + SIB_COLS)
u = folds.filter(pl.col("fold") == 0).select("q")
g0 = G.join(u, on="q", how="semi")
qtr = np.random.default_rng(99).choice(folds.filter(pl.col("fold") != 0)["q"].to_numpy(), args.train_s1, replace=False)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=9, seed=0)


def rows(qs, S2X):
    qdf = pl.DataFrame({"q": np.asarray(qs, dtype=np.int32)})
    d = lazy.join(qdf.lazy(), on="q", how="semi").collect().join(S2X, on=["q", "t"], how="inner")
    return d.join(G.with_columns(y=pl.lit(1, pl.Int8)), on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))


for tau in [float(x) for x in args.taus.split(",")]:
    surv = P1.filter(pl.col("p1") >= tau) if tau > 0 else P1
    S2X = context_frame(surv).join(SIB, on=["q", "t"], how="left")
    tr = rows(qtr, S2X)
    easy = (tr["y"] == 0) & (tr["p1"] < 0.005)
    keep = ~easy | pl.Series(np.random.default_rng(0).random(tr.height) < 0.25)
    tr = tr.filter(keep).with_columns(w=pl.when(easy.filter(keep)).then(4.0).otherwise(1.0))
    es = tr["q"].is_in(np.random.default_rng(0).choice(qtr, len(qtr) // 10, replace=False)).to_numpy()
    X, y, w = tr.select(FEATS2).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["w"].to_numpy()
    del tr
    dtr = lgb.Dataset(X[~es], y[~es], weight=w[~es], feature_name=FEATS2)
    b = lgb.train(PARAMS, dtr, 3000, valid_sets=[lgb.Dataset(X[es], y[es], weight=w[es], reference=dtr)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    del X, y, w, dtr
    V = rows(u["q"].to_numpy(), S2X)
    V = V.select("q", "t", "y").with_columns(p=pl.Series(b.predict(V.select(FEATS2).to_numpy().astype(np.float32), num_threads=9)))
    d = per_s1(expected_f(exclusive(V.select("q", "t", "p")), min_p=0.6, extra_mass=0.2).select("q", "t"), g0, u)
    cs = eval_candidates(V.select("q", "t"), G, u["q"].to_numpy(), n2)
    m = {"tau": tau, "macro_f05": float(d["F"].mean()), "f05_singletons": float(d.filter(pl.col("ngt") == 0)["F"].mean()),
         "pair_recall": cs["pair_recall"], "cand_mean": cs["cand_mean"], "cand_p95": cs["cand_p95"], "cand_max": cs["cand_max"],
         "best_iter": b.best_iteration}
    ex = Experiment(f"cascade_tau{tau}", "matching", {**vars(args), "tau": tau}, f"cascade p1>={tau} + stage-2 retrained ({args.train_s1} S1)", "fold0")
    ex.log(**m)
    ex.finish()
    V.write_parquet(CACHE / f"val_pred_cascade_tau{tau}.parquet")
    print(json.dumps(m), flush=True)
