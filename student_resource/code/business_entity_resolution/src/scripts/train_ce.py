"""Two-fold cross-encoder on uncertain pairs (lo < p1 < hi).

CE_A trains on folds {1,2}, CE_B on folds {3,4}; each scores the other half (OOF for
fusion training), and both score fold 0 and test (averaged). Output: cache/extra_ce_<split>.parquet
with (q, t, ce_logit) for uncertain pairs; other pairs get null (the fusion model
treats them as "not needed")."""
import argparse
import time

import numpy as np
import polars as pl

from ber.data import load_gt_pairs
from ber.models.crossencoder import CrossEncoder, pair_texts
from ber.paths import CACHE
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v2_top12")
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--hi", type=float, default=0.98)
ap.add_argument("--n-train", type=int, default=160_000)
ap.add_argument("--stage", default="all", choices=["train", "oof", "val", "test", "all"])
ap.add_argument("--model-name", default=None, help="reuse CE models trained under this name (default: --name)")
args = ap.parse_args()

folds = load_folds().select(q="idx", fold="fold")
G = load_gt_pairs().with_columns(y=pl.lit(1, pl.Int8))
HALVES = {"A": (1, 2), "B": (3, 4)}


def uncertain(p1: pl.DataFrame) -> pl.DataFrame:
    return p1.filter((pl.col("p1") > args.lo) & (pl.col("p1") < args.hi))


P1 = uncertain(pl.read_parquet(CACHE / f"p1_train_{args.name}.parquet")).join(folds, on="q")
P1 = P1.join(G, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
print(f"uncertain train pairs: {P1.height}, pos rate {P1['y'].mean():.3f}", flush=True)


def model_dir(h):
    return CACHE / f"ce_{args.model_name or args.name}_{h}"


if args.stage in ("train", "all"):
    for h, fs in HALVES.items():
        if (model_dir(h) / "trained.ok").exists():  # checkpoints alone may be partial
            continue
        d = P1.filter(pl.col("fold").is_in(fs))
        d = d.sample(min(args.n_train, d.height), seed={"A": 1, "B": 2}[h])
        a, b = pair_texts(d, "train")
        t0 = time.time()
        ce = CrossEncoder()
        ce.train(a, b, d["y"].to_numpy(), save_to=str(model_dir(h)))
        (model_dir(h) / "trained.ok").write_text(str(d.height))
        print(f"CE {h} trained on {d.height} pairs in {time.time() - t0:.0f}s", flush=True)


def score(pairs, split, heads):
    a, b = pair_texts(pairs, split)
    logits = []
    for h in heads:
        ce = CrossEncoder(str(model_dir(h)))
        logits.append(ce.predict(a, b))
        del ce
    return pairs.select("q", "t").with_columns(ce_logit=pl.Series(np.mean(logits, axis=0).astype(np.float32)))


if args.stage in ("oof", "all"):
    out = CACHE / f"ce_oof_{args.name}.parquet"
    if not out.exists():
        parts = []
        for h, other in (("A", (3, 4)), ("B", (1, 2))):
            parts.append(score(P1.filter(pl.col("fold").is_in(other)), "train", [h]))
        pl.concat(parts).write_parquet(out)
        print("OOF done", flush=True)

if args.stage in ("val", "all"):
    out = CACHE / f"ce_val_{args.name}.parquet"
    if not out.exists():
        score(P1.filter(pl.col("fold") == 0), "train", ["A", "B"]).write_parquet(out)
        print("VAL done", flush=True)

if args.stage in ("test", "all"):
    out = CACHE / f"ce_test_{args.name}.parquet"
    if not out.exists():
        T = uncertain(pl.read_parquet(CACHE / f"test_scores_{args.name}.parquet").select("q", "t", "p1"))
        print(f"uncertain test pairs: {T.height}", flush=True)
        score(T, "test", ["A", "B"]).write_parquet(out)
        print("TEST done", flush=True)
