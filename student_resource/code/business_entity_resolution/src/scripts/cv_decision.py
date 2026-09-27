"""Decision-rule grid over cross-validated stage-2 predictions (cv_stage2.py output).

Grid per fold: cascade filter p1 >= tau (applied BEFORE the decision) x exclusivity scope
{none, fold (targets compete only among the evaluated fold's S1 -- what train_matcher's
fold-0 sweep does), global (all 5 folds' predictions compete -- what test inference does)}
x rule {expected_f(min_p, extra_mass), threshold}. Metric = macro F0.5 over every S1 of
the fold (GT pairs outside the candidate set count as misses).

Writes reports/cv_decision.csv (per fold) and reports/cv_decision_summary.csv.
"""
import argparse

import numpy as np
import polars as pl

from ber.data import load_gt_pairs
from ber.decide import exclusive, expected_f
from ber.paths import CACHE, REPORTS
from ber.splits import load_folds

ap = argparse.ArgumentParser()
ap.add_argument("--taus", default="0,0.01,0.02,0.03,0.05")
ap.add_argument("--check", action="store_true", help="verify vectorized expected_f against ber.decide")
args = ap.parse_args()
TAUS = [float(x) for x in args.taus.split(",")]
MINP = [0.0, 0.4, 0.5, 0.6, 0.7]
EXTRA = [0.0, 0.2, 0.4]
THRS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

folds = load_folds().select(q="idx", fold="fold")
NQ = int(folds["q"].max()) + 1
G = load_gt_pairs()
ngt = np.bincount(G["q"].to_numpy(), minlength=NQ).astype(np.float64)
fold_of = np.full(NQ, -1, np.int8)
fold_of[folds["q"].to_numpy()] = folds["fold"].to_numpy()

cv = {f: pl.read_parquet(CACHE / f"cv_pred_v2_fold{f}.parquet").select("q", "t", "y", "p1", "p") for f in range(5)}
prod0 = pl.read_parquet(CACHE / "val_pred_v2_top12.parquet").select("q", "t", "y", "p1", "p")


def f05(pred_q, pred_y, qs):
    """Macro F0.5 over universe qs for predicted pairs (q, y)."""
    tp = np.bincount(pred_q, weights=pred_y, minlength=NQ)[qs]
    npred = np.bincount(pred_q, minlength=NQ)[qs].astype(np.float64)
    ng = ngt[qs]
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(ng > 0, tp / ng, 0.0)
        F = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    F = np.where(ng == 0, (npred == 0).astype(np.float64), F)
    s = ng == 0
    return float(F.mean()), float(F[s].mean()), float(F[~s].mean()), int(npred.sum())


class Sorted:
    """Frame sorted by (q, p desc) with per-group quantities for vectorized expected-F."""

    def __init__(self, d: pl.DataFrame):
        d = d.sort(["q", "p"], descending=[False, True])
        self.q = d["q"].to_numpy()
        self.y = d["y"].to_numpy().astype(np.float64)
        self.t = d["t"].to_numpy()
        p = d["p"].to_numpy().astype(np.float64)
        self.p = p
        n = len(p)
        self.starts = np.flatnonzero(np.r_[True, self.q[1:] != self.q[:-1]]) if n else np.zeros(0, int)
        gid = np.cumsum(np.r_[True, self.q[1:] != self.q[:-1]]) - 1 if n else np.zeros(0, int)
        self.gid = gid
        self.k = np.arange(n) - self.starts[gid] + 1
        c = np.cumsum(p)
        off = np.r_[0.0, c][self.starts]
        self.cs = c - off[gid]
        self.gsum = np.add.reduceat(p, self.starts) if n else np.zeros(0)
        self.pempty = np.multiply.reduceat(1 - p, self.starts) if n else np.zeros(0)
        self.pmax = p[self.starts] if n else np.zeros(0)

    def expf(self, min_p, extra):
        if len(self.p) == 0:
            return np.zeros(0, bool)
        gid = self.gid
        ef = 1.25 * self.cs / (0.25 * (self.gsum[gid] + extra) + self.k)
        efmax = np.maximum.reduceat(ef, self.starts)
        kbest = np.minimum.reduceat(np.where(ef == efmax[gid], self.k, 1 << 30), self.starts)
        ok = (efmax > self.pempty) & (self.pmax >= min_p)
        return ok[gid] & (self.k <= kbest[gid])


def excl(d):
    return exclusive(d) if d.height else d


def run(frames: dict, label: str, scope_folds):
    """frames: fold -> predictions. Returns list of result rows."""
    out = []
    for tau in TAUS:
        filt = {f: d.filter(pl.col("p1") >= tau) for f, d in frames.items()}
        glob = excl(pl.concat(list(filt.values())))
        for f in scope_folds:
            qs = folds.filter(pl.col("fold") == f)["q"].to_numpy()
            d0 = filt[f]
            n_true = ngt[qs].sum()
            base = {"model": label, "fold": f, "tau": tau, "cand_per_s1": d0.height / len(qs),
                    "cand_recall": float(d0["y"].sum()) / n_true,
                    "cand_recall_rel": float(d0["y"].sum()) / float(frames[f]["y"].sum())}
            variants = {"none": d0, "fold": excl(d0), "global": glob.filter(pl.col("q").is_in(qs))}
            for scope, d in variants.items():
                S = Sorted(d)
                for mp in MINP:
                    for em in EXTRA:
                        keep = S.expf(mp, em)
                        m = f05(S.q[keep], S.y[keep], qs)
                        out.append({**base, "excl": scope, "rule": "expF", "min_p": mp, "extra_mass": em, "thr": None,
                                    "macro_f05": m[0], "f05_single": m[1], "f05_nonsingle": m[2], "n_pred": m[3]})
                for thr in THRS:
                    keep = S.p >= thr
                    m = f05(S.q[keep], S.y[keep], qs)
                    out.append({**base, "excl": scope, "rule": "thr", "min_p": None, "extra_mass": None, "thr": thr,
                                "macro_f05": m[0], "f05_single": m[1], "f05_nonsingle": m[2], "n_pred": m[3]})
            print(label, f, tau, "done", flush=True)
    return out


if args.check:
    d = exclusive(cv[0])
    S = Sorted(d)
    for mp, em in ((0.6, 0.2), (0.0, 0.0), (0.4, 0.4)):
        ref = expected_f(d, min_p=mp, extra_mass=em).select("q", "t")
        keep = S.expf(mp, em)
        mine = pl.DataFrame({"q": S.q[keep], "t": S.t[keep]})
        diff = ref.join(mine, on=["q", "t"], how="anti").height + mine.join(ref, on=["q", "t"], how="anti").height
        print(f"check expF({mp},{em}): ref {ref.height} mine {mine.height} symdiff {diff}", flush=True)

rows = run(cv, "cv300k", range(5))
rows += run({0: prod0, **{f: cv[f] for f in range(1, 5)}}, "prod1200k", [0])
R = pl.DataFrame(rows)
R.write_csv(REPORTS / "cv_decision.csv")

keys = ["tau", "excl", "rule", "min_p", "extra_mass", "thr"]
cvr = R.filter(pl.col("model") == "cv300k")
default = (cvr.filter((pl.col("tau") == 0) & (pl.col("excl") == "fold") & (pl.col("rule") == "expF")
                      & (pl.col("min_p") == 0.6) & (pl.col("extra_mass") == 0.2))
           .select("fold", f_def="macro_f05"))
summ = []
for name, sub in (("folds1-4", cvr.filter(pl.col("fold") >= 1)), ("folds0-4", cvr)):
    s = (sub.join(default, on="fold").with_columns(d=pl.col("macro_f05") - pl.col("f_def"))
         .group_by(keys, maintain_order=True)
         .agg(mean=pl.col("macro_f05").mean(), std=pl.col("macro_f05").std(), min=pl.col("macro_f05").min(),
              single=pl.col("f05_single").mean(), nonsingle=pl.col("f05_nonsingle").mean(),
              d_vs_default=pl.col("d").mean(), d_std=pl.col("d").std(), n_better=(pl.col("d") > 0).sum(),
              cand_per_s1=pl.col("cand_per_s1").mean(), cand_recall=pl.col("cand_recall").mean(),
              cand_recall_rel=pl.col("cand_recall_rel").mean())
         .with_columns(folds=pl.lit(name)).sort("mean", descending=True))
    summ.append(s)
S_ = pl.concat(summ)
S_.write_csv(REPORTS / "cv_decision_summary.csv")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(250)
print(S_.filter(pl.col("folds") == "folds1-4").head(25), flush=True)
print(S_.filter((pl.col("folds") == "folds1-4") & (pl.col("rule") == "expF") & (pl.col("min_p") == 0.6)
                & (pl.col("extra_mass") == 0.2)), flush=True)
