"""Fast stage-2 experiments on cached pair features + OOF stage-1 scores.

Trains stage-2 on folds 1-4 (sampled S1), evaluates macro F0.5 on fold 0 with the
tuned expected-F rule. Options toggle extra feature groups and training size."""
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import candidates as cand
from ber.data import load_gt_pairs
from ber.decide import exclusive, expected_f
from ber.errors import loss_breakdown, per_s1
from ber.experiment import Experiment
from ber.features import add_context_features
from ber.pairs import MODEL_FEATURES_BASE
from ber.paths import CACHE
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v1_top12")
ap.add_argument("--train-s1", type=int, default=300_000)
ap.add_argument("--setfeats", action="store_true")
ap.add_argument("--extra", default="", help="comma list of extra feature parquet names in cache/extra_<name>.parquet")
ap.add_argument("--leaves", type=int, default=255)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--tag", default="")
args = ap.parse_args()

folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs()
lazy = pl.scan_parquet(CACHE / "pairs" / f"train_{args.name}" / "*.parquet")
cols = lazy.collect_schema().names()
BASE = MODEL_FEATURES_BASE + list(dict.fromkeys(cand.retr_feature_cols(pl.DataFrame(schema=cols)) + ["pr", "pr_rank"]))
P1 = pl.read_parquet(CACHE / f"p1_train_{args.name}.parquet")
CTX = add_context_features(P1, "p1", "c1")
FEATS = BASE + ["p1", "c1_qrank", "c1_qgap", "c1_trank", "c1_tgap_best", "c1_t_ncomp", "c1_q_ncand", "c1_tmargin"]
if args.setfeats:
    CTX = CTX.with_columns(
        c1_q_sum=pl.col("p1").sum().over("q"),
        c1_q_n50=(pl.col("p1") > 0.5).sum().over("q").cast(pl.Float32),
        c1_q_n20=(pl.col("p1") > 0.2).sum().over("q").cast(pl.Float32),
        c1_q_max=pl.col("p1").max().over("q"),
        c1_t_sum_other=(pl.col("p1").sum().over("t") - pl.col("p1")),
        c1_src_rank=pl.col("p1").rank("ordinal", descending=True).over(["q", pl.col("t") < 0]).cast(pl.Float32),
    )
    FEATS += ["c1_q_sum", "c1_q_n50", "c1_q_n20", "c1_q_max", "c1_t_sum_other"]
for e in [x for x in args.extra.split(",") if x]:
    ex_df = pl.read_parquet(CACHE / f"extra_{e}.parquet")
    CTX = CTX.join(ex_df, on=["q", "t"], how="left")
    FEATS += [c for c in ex_df.columns if c not in ("q", "t")]


def rows_for(qs):
    qdf = pl.DataFrame({"q": np.asarray(qs, dtype=np.int32)})
    d = lazy.join(qdf.lazy(), on="q", how="semi").collect().join(CTX, on=["q", "t"], how="left")
    return d.join(G.with_columns(y=pl.lit(1, pl.Int8)), on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))


qtr = np.random.default_rng(99).choice(folds.filter(pl.col("fold") != 0)["q"].to_numpy(), args.train_s1, replace=False)
tr = rows_for(qtr)
es = tr["q"].is_in(np.random.default_rng(0).choice(qtr, len(qtr) // 10, replace=False)).to_numpy()
X, y = tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy()
del tr
params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=9, seed=0)
dtr = lgb.Dataset(X[~es], y[~es], feature_name=FEATS)
b = lgb.train(params, dtr, 3000, valid_sets=[lgb.Dataset(X[es], y[es], reference=dtr)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
del X, y, dtr
u = folds.filter(pl.col("fold") == 0).select("q")
V = rows_for(u["q"].to_numpy())
V = V.select("q", "t", "y").with_columns(p=pl.Series(b.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=9)))
g = G.join(u, on="q", how="semi")
pred = expected_f(exclusive(V.select("q", "t", "p")), min_p=0.6, extra_mass=0.2).select("q", "t")
d = per_s1(pred, g, u)
single = d["ngt"] == 0
m = {"macro_f05": float(d["F"].mean()), "f05_singletons": float(d.filter(single)["F"].mean()),
     "f05_nonsingletons": float(d.filter(~single)["F"].mean()), "best_iter": b.best_iteration}
tag = args.tag or f"s2_n{args.train_s1}{'_set' if args.setfeats else ''}{'_' + args.extra.replace(',', '+') if args.extra else ''}"
ex = Experiment(f"matcher_{tag}", "matching", {**vars(args), "features": FEATS}, f"stage-2 experiment {tag}", "fold0")
ex.log(**m)
ex.save_json("loss_breakdown.json", loss_breakdown(pred, V.select("q", "t"), g, u))
ex.save_json("feature_importance.json", sorted([(a, float(v)) for a, v in zip(FEATS, b.feature_importance("gain"))], key=lambda x: -x[1])[:30])
V.select("q", "t", "y", "p").write_parquet(CACHE / f"val_pred_{tag}.parquet")
ex.finish()
print(tag, json.dumps(m), flush=True)
