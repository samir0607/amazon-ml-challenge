"""Build cached float32 matrices for stage-2 GBDT screening (exp_gbdt_run.py).

Train: fixed 400k-S1 sample of folds 1-4 (rng 99), easy negatives (y=0, p1<0.005)
kept at 25% with weight 4 (rng 0), 10% of S1 groups flagged for early stopping (rng 0).
Val: ALL fold-0 candidate pairs. Output in cache/gbdt_exp/."""
import argparse
import json

import numpy as np
import polars as pl

from ber import candidates as cand
from ber.data import load_gt_pairs
from ber.pairs import MODEL_FEATURES_BASE
from ber.paths import CACHE
from ber.splits import load_folds
from ber.stage2 import STAGE2_EXTRA

ap = argparse.ArgumentParser()
ap.add_argument("--train-s1", type=int, default=400_000)
ap.add_argument("--easy-neg-keep", type=float, default=0.25)
args = ap.parse_args()

OUT = CACHE / "gbdt_exp"
OUT.mkdir(exist_ok=True)
folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs().with_columns(y=pl.lit(1, pl.Int8))
lazy = pl.scan_parquet(CACHE / "pairs" / "train_v1_top12" / "*.parquet")
cols = lazy.collect_schema().names()
BASE = MODEL_FEATURES_BASE + list(dict.fromkeys(cand.retr_feature_cols(pl.DataFrame(schema=cols)) + ["pr", "pr_rank"]))
FEATS2 = BASE + STAGE2_EXTRA
s2x = pl.scan_parquet(CACHE / "s2x_train_v2_top12.parquet")


def rows(qs):
    qdf = pl.DataFrame({"q": np.asarray(qs, dtype=np.int32)}).lazy()
    d = (lazy.join(qdf, on="q", how="semi").select(["q", "t"] + BASE)
         .join(s2x.join(qdf, on="q", how="semi"), on=["q", "t"], how="left")
         .join(G.lazy(), on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
         .collect())
    return d.sort(["q", "t"])


def to_f32(d):
    X = np.empty((d.height, len(FEATS2)), dtype=np.float32)
    for j, c in enumerate(FEATS2):
        X[:, j] = d[c].cast(pl.Float32).to_numpy()
    return X


# ---- train
qtr = np.random.default_rng(99).choice(folds.filter(pl.col("fold") != 0)["q"].to_numpy(), args.train_s1, replace=False)
d = rows(qtr)
easy = ((d["y"] == 0) & (d["p1"] < 0.005)).to_numpy()
keep = ~easy | (np.random.default_rng(0).random(d.height) < args.easy_neg_keep)
d = d.filter(pl.Series(keep))
w = np.where(easy[keep], 1.0 / args.easy_neg_keep, 1.0).astype(np.float32)
uq = np.sort(d["q"].unique().to_numpy())
es_q = np.random.default_rng(0).choice(uq, len(uq) // 10, replace=False)
es = d["q"].is_in(es_q).to_numpy()
np.save(OUT / "Xtr.npy", to_f32(d))
np.save(OUT / "ytr.npy", d["y"].to_numpy().astype(np.float32))
np.save(OUT / "wtr.npy", w)
np.save(OUT / "estr.npy", es)
np.save(OUT / "qtr.npy", d["q"].to_numpy())
print(f"train rows {d.height:,}  pos {int(d['y'].sum()):,}  es rows {es.sum():,}", flush=True)
del d

# ---- val (fold 0), in chunks of S1 to bound memory
qva = np.sort(folds.filter(pl.col("fold") == 0)["q"].to_numpy())
n_va = lazy.join(pl.DataFrame({"q": qva.astype(np.int32)}).lazy(), on="q", how="semi").select(pl.len()).collect().item()
Xva = np.lib.format.open_memmap(OUT / "Xva.npy", mode="w+", dtype=np.float32, shape=(n_va, len(FEATS2)))
keys, off = [], 0
for ch in np.array_split(qva, 4):
    d = rows(ch)
    Xva[off:off + d.height] = to_f32(d)
    off += d.height
    keys.append(d.select("q", "t", "y"))
    del d
assert off == n_va
Xva.flush()
del Xva
pl.concat(keys).write_parquet(OUT / "va_keys.parquet")
(OUT / "meta.json").write_text(json.dumps({"feats2": FEATS2, "train_s1": args.train_s1, "easy_neg_keep": args.easy_neg_keep}))
print("val rows", sum(k.height for k in keys), flush=True)
