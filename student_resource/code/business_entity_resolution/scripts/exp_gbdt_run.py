"""Stage-2 GBDT screening on cached matrices (exp_gbdt_build.py): LightGBM parameter
variants, CatBoost, and blends. Each variant is trained on the same 400k-S1 sample and
evaluated on all of fold 0 with the production decision rule."""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.data import load_gt_pairs
from ber.decide import exclusive, expected_f
from ber.errors import per_s1
from ber.experiment import Experiment
from ber.paths import CACHE
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="base,lr03,leaves511,leaves127,mdl300,ff06,l2_5,catboost")
args = ap.parse_args()

D = CACHE / "gbdt_exp"
meta = json.loads((D / "meta.json").read_text())
FEATS = meta["feats2"]
Xtr, ytr, wtr, es = (np.load(D / f) for f in ("Xtr.npy", "ytr.npy", "wtr.npy", "estr.npy"))
es = es.astype(bool)
Xva = np.load(D / "Xva.npy", mmap_mode="r")
keys = pl.read_parquet(D / "va_keys.parquet")
u = load_folds().filter(pl.col("fold") == 0).select(q="idx")
g0 = load_gt_pairs().join(u, on="q", how="semi")

BASE = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=9, seed=0)
LGB = {
    "base": {}, "lr03": {"learning_rate": 0.03}, "leaves511": {"num_leaves": 511}, "leaves127": {"num_leaves": 127},
    "mdl300": {"min_data_in_leaf": 300}, "ff06": {"feature_fraction": 0.6}, "l2_5": {"lambda_l2": 5.0},
    "seed1": {"seed": 1}, "seed2": {"seed": 2},
}


def evaluate(p):
    d = keys.select("q", "t").with_columns(p=pl.Series(p.astype(np.float32)))
    r = per_s1(expected_f(exclusive(d), min_p=0.6, extra_mass=0.2).select("q", "t"), g0, u)
    return float(r["F"].mean()), float(r.filter(pl.col("ngt") == 0)["F"].mean())


def predict_lgb(b):
    out = np.empty(Xva.shape[0], dtype=np.float32)
    for i in range(0, Xva.shape[0], 1_000_000):
        out[i:i + 1_000_000] = b.predict(np.asarray(Xva[i:i + 1_000_000]), num_threads=9)
    return out


preds = {}
for v in args.variants.split(","):
    t0 = time.time()
    if v == "catboost":
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(depth=8, learning_rate=0.08, iterations=3000, l2_leaf_reg=3, thread_count=9,
                               od_type="Iter", od_wait=100, verbose=0, random_seed=0)
        m.fit(Pool(Xtr[~es], ytr[~es], weight=wtr[~es]), eval_set=Pool(Xtr[es], ytr[es], weight=wtr[es]), use_best_model=True)
        p = np.concatenate([m.predict_proba(np.asarray(Xva[i:i + 1_000_000]))[:, 1] for i in range(0, Xva.shape[0], 1_000_000)])
        it = m.get_best_iteration()
    else:
        params = {**BASE, **LGB[v]}
        dtr = lgb.Dataset(Xtr[~es], ytr[~es], weight=wtr[~es], feature_name=FEATS)
        b = lgb.train(params, dtr, 4000, valid_sets=[lgb.Dataset(Xtr[es], ytr[es], weight=wtr[es], reference=dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p, it = predict_lgb(b), b.best_iteration
    preds[v] = p.astype(np.float32)
    np.save(D / f"pred_{v}.npy", preds[v])
    f, fs = evaluate(p)
    ex = Experiment(f"gbdt_{v}", "matching", {"variant": v, "train_s1": meta["train_s1"]}, f"stage-2 variant {v} (400k S1)", "fold0")
    ex.log(macro_f05=f, f05_singletons=fs, best_iter=it)
    ex.finish()
    print(json.dumps({"variant": v, "macro_f05": round(f, 6), "single": round(fs, 5), "iter": it, "s": round(time.time() - t0)}), flush=True)

# blends of everything trained in this run
names = list(preds)
if "catboost" in preds and len(names) > 1:
    best_lgb = max((n for n in names if n != "catboost"), key=lambda n: evaluate(preds[n])[0])
    for w in (0.3, 0.5, 0.7):
        f, fs = evaluate(w * preds[best_lgb] + (1 - w) * preds["catboost"])
        print(json.dumps({"variant": f"blend_{best_lgb}_{w}_cat", "macro_f05": round(f, 6), "single": round(fs, 5)}), flush=True)
lgbs = [n for n in names if n != "catboost"]
if len(lgbs) >= 3:
    f, fs = evaluate(np.mean([preds[n] for n in lgbs], axis=0))
    print(json.dumps({"variant": "avg_all_lgb", "macro_f05": round(f, 6), "single": round(fs, 5)}), flush=True)
