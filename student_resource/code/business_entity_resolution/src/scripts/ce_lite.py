"""Time-boxed cross-encoder fusion on borderline pairs only.

1. train one multilingual-e5-small cross-encoder on N borderline train pairs (folds 1-4,
   OOF stage-1 p1 in (lo, hi))
2. score borderline fold-0 pairs (stage-2 p in (lo, hi)) and borderline test pairs
3. fuse with logistic regression on [logit p, ce logit, product], fit on half A of fold-0 S1,
   report macro F0.5 on half B (base vs fused); then refit on all fold 0 for test
Pairs outside the band keep the stage-2 probability."""
import argparse
import json
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from ber.data import load_gt_pairs
from ber.decide import exclusive, expected_f
from ber.errors import per_s1
from ber.experiment import Experiment
from ber.models.crossencoder import CrossEncoder, pair_texts
from ber.paths import CACHE
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=100_000)
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.98)
ap.add_argument("--val", default="val_pred_v2_top12.parquet", help="fold-0 stage-2 preds (q,t,y,p_lgb)")
ap.add_argument("--val-col", default="p_lgb")
ap.add_argument("--test", default="test_scores_v2_top12.parquet")
args = ap.parse_args()

folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs().with_columns(y=pl.lit(1, pl.Int8))
MODEL = CACHE / "ce_lite"
band = (pl.col("p") > args.lo) & (pl.col("p") < args.hi)

# 1 ---- train
if not (MODEL / "trained.ok").exists():
    P1 = pl.read_parquet(CACHE / "p1_train_v2_top12.parquet").rename({"p1": "p"}).filter(band)
    tr = P1.join(folds.filter(pl.col("fold") != 0), on="q").sample(args.n_train, seed=7)
    tr = tr.join(G, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    a, b = pair_texts(tr, "train")
    t0 = time.time()
    ce = CrossEncoder()
    ce.train(a, b, tr["y"].to_numpy(), save_to=str(MODEL))
    (MODEL / "trained.ok").write_text(str(tr.height))
    print(f"CE trained on {tr.height} pairs in {time.time() - t0:.0f}s", flush=True)
ce = CrossEncoder(str(MODEL))


def ce_scores(df, split, name):
    path = CACHE / f"ce_lite_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    a, b = pair_texts(df, split)
    out = df.select("q", "t").with_columns(ce=pl.Series(ce.predict(a, b)))
    out.write_parquet(path)
    return out


def feats(d):
    lp = np.log(np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6) / np.clip(1 - d["p"].to_numpy(), 1e-6, 1))
    c = d["ce"].to_numpy()
    return np.c_[lp, c, lp * c]


# 2 ---- fold 0
V = pl.read_parquet(CACHE / args.val).select("q", "t", "y", p=pl.col(args.val_col))
Vb = V.filter(band)
print(f"fold-0 borderline pairs: {Vb.height} / {V.height}", flush=True)
Vb = Vb.join(ce_scores(Vb, "train", "val"), on=["q", "t"])
u = folds.filter(pl.col("fold") == 0).select("q")
half = u.with_columns(h=(pl.col("q").hash(11) % 2).cast(pl.Int8))
A, B = half.filter(pl.col("h") == 0).select("q"), half.filter(pl.col("h") == 1).select("q")
g0 = G.select("q", "t").join(u, on="q", how="semi")


def fuse(model, Vall, Vband):
    pf = Vband.select("q", "t").with_columns(pf=pl.Series(model.predict_proba(feats(Vband))[:, 1].astype(np.float32)))
    return Vall.join(pf, on=["q", "t"], how="left").with_columns(p=pl.coalesce("pf", "p")).drop("pf")


def f05(d, uu):
    pred = expected_f(exclusive(d.select("q", "t", "p")), min_p=0.6, extra_mass=0.2).select("q", "t")
    r = per_s1(pred, g0.join(uu, on="q", how="semi"), uu)
    return float(r["F"].mean()), float(r.filter(pl.col("ngt") == 0)["F"].mean())


VbA = Vb.join(A, on="q", how="semi")
lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(VbA), VbA["y"].to_numpy())
VB = V.join(B, on="q", how="semi")
base_B, fused_B = f05(VB, B), f05(fuse(lr, VB, Vb.join(B, on="q", how="semi")), B)
from sklearn.metrics import roc_auc_score
auc_p = roc_auc_score(Vb["y"], Vb["p"]); auc_ce = roc_auc_score(Vb["y"], Vb["ce"])
res = {"half_B_base": base_B, "half_B_fused": fused_B, "gain": fused_B[0] - base_B[0], "auc_band_p": auc_p,
       "auc_band_ce": auc_ce, "coef": lr.coef_.tolist(), "n_band": Vb.height}
print(json.dumps(res), flush=True)
ex = Experiment("ce_lite_fusion", "matching", vars(args), "e5-small cross-encoder on borderline pairs + LR fusion (half-B holdout)", "fold0-halfB")
ex.log(macro_f05=fused_B[0], f05_singletons=fused_B[1], base_macro_f05=base_B[0], gain=res["gain"], auc_band_p=auc_p, auc_band_ce=auc_ce)
ex.finish(decision="keep" if res["gain"] > 0.0005 else "reject")

# 3 ---- test (only if it helps)
if res["gain"] > 0.0005:
    lr_all = LogisticRegression(C=1.0, max_iter=1000).fit(feats(Vb), Vb["y"].to_numpy())
    T = pl.read_parquet(CACHE / args.test).select("q", "t", "p")
    Tb = T.filter(band)
    print(f"test borderline pairs: {Tb.height} / {T.height}", flush=True)
    Tb = Tb.join(ce_scores(Tb, "test", "test"), on=["q", "t"])
    fuse(lr_all, T, Tb).write_parquet(CACHE / "test_scores_ce_lite.parquet")
    (CACHE / "ce_lite_fusion.json").write_text(json.dumps({"coef": lr_all.coef_.tolist(), "intercept": lr_all.intercept_.tolist(), **vars(args)}))
    print("TEST FUSED WRITTEN", flush=True)
print("DONE", flush=True)
