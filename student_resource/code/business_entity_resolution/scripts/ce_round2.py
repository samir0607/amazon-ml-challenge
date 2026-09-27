"""Round 2: continue training the ce_lite cross-encoder on fresh borderline train pairs,
score the fold-0 band, fuse [logit p, ce1, ce2, interactions] on half A of fold 0, and
compare with the round-1 fusion on held-out half B. Test is scored only if round 2 wins."""
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
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.995)
ap.add_argument("--val", default="val_pred_cascade_tau0.01.parquet")
ap.add_argument("--test", default="test_scores_v2_top12.parquet")
ap.add_argument("--stage", default="val", choices=["val", "test"])
args = ap.parse_args()

folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs().with_columns(y=pl.lit(1, pl.Int8))
M2 = CACHE / "ce_lite2"
band = (pl.col("p") > args.lo) & (pl.col("p") < args.hi)

if not (M2 / "trained.ok").exists():
    P1 = pl.read_parquet(CACHE / "p1_train_v2_top12.parquet").rename({"p1": "p"}).filter((pl.col("p") > 0.02) & (pl.col("p") < 0.98))
    tr = P1.join(folds.filter(pl.col("fold") != 0), on="q").sample(args.n_train, seed=8)
    tr = tr.join(G, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    a, b = pair_texts(tr, "train")
    t0 = time.time()
    ce = CrossEncoder(str(CACHE / "ce_lite"))  # continue from round 1
    ce.train(a, b, tr["y"].to_numpy(), lr=args.lr, save_to=str(M2), seed=1)
    (M2 / "trained.ok").write_text(str(tr.height))
    print(f"CE round 2 trained on {tr.height} more pairs in {time.time() - t0:.0f}s", flush=True)
ce2 = CrossEncoder(str(M2))


def ce2_scores(df, split, name):
    path = CACHE / f"ce_lite2_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    a, b = pair_texts(df, split)
    out = df.select("q", "t").with_columns(ce2=pl.Series(ce2.predict(a, b)))
    out.write_parquet(path)
    return out


def ce1(name):
    return pl.concat([pl.read_parquet(CACHE / f"ce_lite_{name}.parquet"),
                      pl.read_parquet(CACHE / f"ce_lite_{name}_extra_0.995.parquet")])


def lp_of(d):
    p = np.clip(d["p"].to_numpy(), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def f1(d):  # round-1 features
    lp, c = lp_of(d), d["ce"].to_numpy()
    return np.c_[lp, c, lp * c]


def f2(d):  # round-2 features
    lp, c, c2 = lp_of(d), d["ce"].to_numpy(), d["ce2"].to_numpy()
    return np.c_[lp, c, c2, lp * c, lp * c2]


def fuse(model, feats, Vall, Vband):
    pf = Vband.select("q", "t").with_columns(pf=pl.Series(model.predict_proba(feats(Vband))[:, 1].astype(np.float32)))
    return Vall.join(pf, on=["q", "t"], how="left").with_columns(p=pl.coalesce("pf", "p")).drop("pf")


V = pl.read_parquet(CACHE / args.val).select("q", "t", "y", "p")
Vb = V.filter(band).join(ce1("val"), on=["q", "t"])
Vb = Vb.join(ce2_scores(Vb.select("q", "t"), "train", "val"), on=["q", "t"])
u = folds.filter(pl.col("fold") == 0).select("q")
half = u.with_columns(h=(pl.col("q").hash(11) % 2).cast(pl.Int8))
A, B = half.filter(pl.col("h") == 0).select("q"), half.filter(pl.col("h") == 1).select("q")
g0 = G.select("q", "t").join(u, on="q", how="semi")


def f05(d, uu):
    pred = expected_f(exclusive(d.select("q", "t", "p")), min_p=0.6, extra_mass=0.2).select("q", "t")
    r = per_s1(pred, g0.join(uu, on="q", how="semi"), uu)
    return float(r["F"].mean()), float(r.filter(pl.col("ngt") == 0)["F"].mean())


VbA, VbB, VB = Vb.join(A, on="q", how="semi"), Vb.join(B, on="q", how="semi"), V.join(B, on="q", how="semi")
lr1 = LogisticRegression(C=1.0, max_iter=2000).fit(f1(VbA), VbA["y"].to_numpy())
lr2 = LogisticRegression(C=1.0, max_iter=2000).fit(f2(VbA), VbA["y"].to_numpy())
r1, r2 = f05(fuse(lr1, f1, VB, VbB), B), f05(fuse(lr2, f2, VB, VbB), B)
from sklearn.metrics import roc_auc_score
res = {"half_B_round1": r1, "half_B_round2": r2, "gain": r2[0] - r1[0],
       "auc_ce1": roc_auc_score(Vb["y"], Vb["ce"]), "auc_ce2": roc_auc_score(Vb["y"], Vb["ce2"])}
print(json.dumps(res), flush=True)
if args.stage == "val":
    ex = Experiment("ce_round2", "matching", vars(args), "cross-encoder round 2 (+100k pairs) fused with round 1", "fold0-halfB")
    ex.log(macro_f05=r2[0], f05_singletons=r2[1], base_macro_f05=r1[0], gain=res["gain"])
    ex.finish(decision="keep" if res["gain"] > 0.0001 else "reject")

if args.stage == "test":
    lr_all = LogisticRegression(C=1.0, max_iter=2000).fit(f2(Vb), Vb["y"].to_numpy())
    T = pl.read_parquet(CACHE / args.test).select("q", "t", "p")
    Tb = T.filter(band).join(ce1("test"), on=["q", "t"])
    print(f"test band pairs: {Tb.height}", flush=True)
    Tb = Tb.join(ce2_scores(Tb.select("q", "t"), "test", "test"), on=["q", "t"])
    fuse(lr_all, f2, T, Tb).write_parquet(CACHE / "test_scores_ce_lite.parquet")
    print("TEST FUSED WRITTEN (round 2)", flush=True)
print("DONE", flush=True)
