"""Extend the cross-encoder band upward (to --hi2): score the extra fold-0 / test pairs with
the trained ce_lite model, refit the logistic fusion on half A of fold 0, compare on half
B against the current band, and write fused test scores only if it helps."""
import argparse
import json

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
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.98)
ap.add_argument("--hi2", type=float, default=0.995)
ap.add_argument("--val", default="val_pred_cascade_tau0.01.parquet")
ap.add_argument("--test", default="test_scores_v2_top12.parquet")
args = ap.parse_args()

ce = CrossEncoder(str(CACHE / "ce_lite"))
folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs()
extra = (pl.col("p") >= args.hi) & (pl.col("p") < args.hi2)
band2 = (pl.col("p") > args.lo) & (pl.col("p") < args.hi2)


def scores(df, split, name):
    path = CACHE / f"ce_lite_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    a, b = pair_texts(df, split)
    out = df.select("q", "t").with_columns(ce=pl.Series(ce.predict(a, b)))
    out.write_parquet(path)
    return out


def feats(d):
    p = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    c = d["ce"].to_numpy()
    return np.c_[lp, c, lp * c]


V = pl.read_parquet(CACHE / args.val).select("q", "t", "y", "p")
ce_val = pl.concat([pl.read_parquet(CACHE / "ce_lite_val.parquet"), scores(V.filter(extra), "train", f"val_extra_{args.hi2}")])
Vb = V.filter(band2).join(ce_val, on=["q", "t"])
u = folds.filter(pl.col("fold") == 0).select("q")
half = u.with_columns(h=(pl.col("q").hash(11) % 2).cast(pl.Int8))
A, B = half.filter(pl.col("h") == 0).select("q"), half.filter(pl.col("h") == 1).select("q")
g0 = G.join(u, on="q", how="semi")


def fuse(model, Vall, Vband):
    pf = Vband.select("q", "t").with_columns(pf=pl.Series(model.predict_proba(feats(Vband))[:, 1].astype(np.float32)))
    return Vall.join(pf, on=["q", "t"], how="left").with_columns(p=pl.coalesce("pf", "p")).drop("pf")


def f05(d, uu):
    pred = expected_f(exclusive(d.select("q", "t", "p")), min_p=0.6, extra_mass=0.2).select("q", "t")
    r = per_s1(pred, g0.join(uu, on="q", how="semi"), uu)
    return float(r["F"].mean()), float(r.filter(pl.col("ngt") == 0)["F"].mean())


def run(bandexpr):
    vb = Vb.filter(bandexpr)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(feats(vb.join(A, on="q", how="semi")), vb.join(A, on="q", how="semi")["y"].to_numpy())
    VB = V.join(B, on="q", how="semi")
    return f05(fuse(lr, VB, vb.join(B, on="q", how="semi")), B), vb


old = run((pl.col("p") > args.lo) & (pl.col("p") < args.hi))
new = run(band2)
gain = new[0][0] - old[0][0]
print(json.dumps({"half_B_old_band": old[0], "half_B_new_band": new[0], "gain": gain}), flush=True)
ex = Experiment(f"ce_band_hi{args.hi2}", "matching", vars(args), f"cross-encoder band extended to {args.hi2}", "fold0-halfB")
ex.log(macro_f05=new[0][0], f05_singletons=new[0][1], base_macro_f05=old[0][0], gain=gain)
ex.finish(decision="keep" if gain > 0.0001 else "reject")

if gain > 0.0001:
    lr_all = LogisticRegression(C=1.0, max_iter=1000).fit(feats(new[1]), new[1]["y"].to_numpy())
    T = pl.read_parquet(CACHE / args.test).select("q", "t", "p")
    ce_test = pl.concat([pl.read_parquet(CACHE / "ce_lite_test.parquet"), scores(T.filter(extra), "test", f"test_extra_{args.hi2}")])
    Tb = T.filter(band2).join(ce_test, on=["q", "t"])
    fuse(lr_all, T, Tb).write_parquet(CACHE / "test_scores_ce_lite.parquet")
    print("TEST FUSED WRITTEN (extended band)", flush=True)
print("DONE", flush=True)
