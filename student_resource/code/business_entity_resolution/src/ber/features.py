"""Pairwise features for (S1 query, S2/S3 target) candidates.

String features are computed with rapidfuzz in worker processes over chunks of pairs.
Token IDF tables are built once over the target corpus and shared copy-on-write with
forked workers. Context features (competition among candidates / among S1s for the
same target) are added in `add_context_features` after retrieval scores are known.
"""
import math
from multiprocessing import get_context

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import NORM_VERSION
from .paths import CACHE

Q_COLS = ["n_core", "n_sorted", "n_skel", "a_norm", "a_sorted", "a_nums", "a_postal", "a_region"]
T_COLS = Q_COLS + ["n_alias", "n_domain", "nonascii_name", "nonascii_addr"]


def load_pair_inputs(split: str):
    """Query / target frames restricted to the columns features need."""
    from .data import load_queries, load_targets
    q = load_queries(split, Q_COLS)
    t = load_targets(split, Q_COLS + ["n_alias", "n_domain", "business_name", "business_address"])
    t = t.with_columns(
        nonascii_name=(~pl.col("business_name").str.contains(r"^[\x00-\x7F]*$")).cast(pl.Int8),
        nonascii_addr=(~pl.col("business_address").fill_null("").str.contains(r"^[\x00-\x7F]*$")).cast(pl.Int8),
    ).drop("business_name", "business_address")
    return q, t

_IDF_N: dict = {}
_IDF_A: dict = {}
_IDF_DEFAULT = (0.0, 0.0)


def build_idf(split: str):
    """Token document frequencies over targets for name-core and address tokens."""
    from .data import load_targets
    out = {}
    for key, col in (("name", "n_core"), ("addr", "a_sorted")):
        path = CACHE / f"idf_{NORM_VERSION}_{split}_{key}.parquet"
        if not path.exists():
            t = load_targets(split, [col])
            n = t.height
            df = (t.select(tok=pl.col(col).str.split(" ").list.unique()).explode("tok")
                  .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
                  .group_by("tok").agg(df=pl.len())
                  .with_columns(idf=(pl.lit(n) / (pl.col("df") + 1)).log().cast(pl.Float32)))
            df.write_parquet(path)
        out[key] = pl.read_parquet(path)
    return out


def _load_idf_dicts(split):
    global _IDF_N, _IDF_A, _IDF_DEFAULT
    idf = build_idf(split)
    _IDF_N = dict(zip(idf["name"]["tok"].to_list(), idf["name"]["idf"].to_list()))
    _IDF_A = dict(zip(idf["addr"]["tok"].to_list(), idf["addr"]["idf"].to_list()))
    _IDF_DEFAULT = (max(_IDF_N.values()), max(_IDF_A.values()))


def _wovl(a: set, b: set, idf: dict, dflt: float):
    """IDF-weighted overlap: shared weight / min and / union of side weights."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    wa = {t: idf.get(t, dflt) for t in a}
    wb = {t: idf.get(t, dflt) for t in b}
    inter = sum(wa[t] for t in a & b)
    sa, sb = sum(wa.values()), sum(wb.values())
    uni = sa + sb - inter
    max_shared = max((wa[t] for t in a & b), default=0.0)
    return inter / max(min(sa, sb), 1e-6), inter / max(uni, 1e-6), max_shared


def _jac(a: set, b: set):
    if not a and not b:
        return -1.0
    return len(a & b) / max(len(a | b), 1)


FEATURE_NAMES = [
    "n_core_eq", "n_sorted_eq", "n_skel_eq",
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_nospace_lev",
    "n_skel_tset", "n_skel_ratio", "n_jac", "n_skel_jac",
    "n_widf_min", "n_widf_uni", "n_max_shared_idf", "n_q_idf_sum", "n_t_idf_sum",
    "n_alias_tset", "n_nospace_partial", "n_len_ratio", "n_tok_diff", "t_nonascii_name",
    "a_ratio", "a_tset", "a_partial", "a_jac", "a_widf_min", "a_widf_uni", "a_max_shared_idf",
    "num_jac", "num_shared", "num_first_eq", "num_conflict", "num_q_n", "num_t_n",
    "postal_eq", "postal_conflict", "region_eq", "region_conflict",
    "q_addr_missing", "t_addr_missing", "a_len_ratio", "t_nonascii_addr",
    "x_min", "x_max", "x_prod", "x_nm_skel_addr",
]


def _pair_feats(r):
    (qc, qs, qk, qa, qas, qnum, qpo, qreg,
     tc, ts, tk, ta, tas, tnum, tpo, treg, talias, tdom, t_na_name, t_na_addr) = r
    qct, tct = set(qc.split()), set(tc.split())
    qkt, tkt = set(qk.split()), set(tk.split())
    qnos, tnos = qc.replace(" ", ""), tc.replace(" ", "")
    wn = _wovl(qct, tct, _IDF_N, _IDF_DEFAULT[0])
    qsum = sum(_IDF_N.get(t, _IDF_DEFAULT[0]) for t in qct)
    tsum = sum(_IDF_N.get(t, _IDF_DEFAULT[0]) for t in tct)
    n_tset = fuzz.token_set_ratio(qc, tc) / 100
    n_tsort = fuzz.token_sort_ratio(qc, tc) / 100
    skel_tset = fuzz.token_set_ratio(qk, tk) / 100
    ta_missing = 1.0 if not ta else 0.0
    if ta and qa:
        a_ratio = fuzz.ratio(qa, ta) / 100
        a_tset = fuzz.token_set_ratio(qa, ta) / 100
        a_partial = fuzz.partial_ratio(qa, ta) / 100
        qat, tat = set(qas.split()), set(tas.split())
        a_jac = _jac(qat, tat)
        wa = _wovl(qat, tat, _IDF_A, _IDF_DEFAULT[1])
        alr = min(len(qa), len(ta)) / max(len(qa), len(ta))
    else:
        a_ratio = a_tset = a_partial = a_jac = -1.0
        wa = (-1.0, -1.0, -1.0)
        alr = -1.0
    qnl, tnl = qnum.split(), tnum.split()
    qns, tns = set(qnl), set(tnl)
    if qns and tns:
        num_jac = len(qns & tns) / len(qns | tns)
        num_conf = 1.0 if not (qns & tns) else 0.0
        num_first = 1.0 if qnl[0] == tnl[0] else 0.0
    else:
        num_jac, num_conf, num_first = -1.0, -1.0, -1.0
    qp, tp = set(qpo.split()), set(tpo.split())
    postal_eq = 1.0 if qp & tp else (-1.0 if not qp or not tp else 0.0)
    postal_conf = 1.0 if (qp and tp and not qp & tp) else 0.0
    qr, tr = set(qreg.split()), set(treg.split())
    region_eq = 1.0 if qr & tr else (-1.0 if not qr or not tr else 0.0)
    region_conf = 1.0 if (qr and tr and not qr & tr) else 0.0
    name_s = max(n_tset, skel_tset)
    addr_s = a_tset if a_tset >= 0 else 0.5
    return (
        float(qc == tc), float(qs == ts), float(qk == tk),
        fuzz.ratio(qc, tc) / 100, n_tsort, n_tset, fuzz.partial_ratio(qc, tc) / 100,
        JaroWinkler.normalized_similarity(qc, tc), Levenshtein.normalized_similarity(qnos, tnos),
        skel_tset, fuzz.ratio(qk, tk) / 100, _jac(qct, tct), _jac(qkt, tkt),
        wn[0], wn[1], wn[2], qsum, tsum,
        (fuzz.token_set_ratio(qc, talias) / 100) if talias else -1.0,
        fuzz.partial_ratio(qnos, tnos) / 100 if tdom else -1.0,
        min(len(qc), len(tc)) / max(len(qc), len(tc), 1), abs(len(qct) - len(tct)),
        float(t_na_name),
        a_ratio, a_tset, a_partial, a_jac, wa[0], wa[1], wa[2],
        num_jac, len(qns & tns), num_first, num_conf, len(qns), len(tns),
        postal_eq, postal_conf, region_eq, region_conf,
        1.0 if not qa else 0.0, ta_missing, alr, float(t_na_addr),
        min(name_s, addr_s), max(name_s, addr_s), name_s * addr_s, skel_tset * addr_s,
    )


def _chunk(rows):
    return np.asarray([_pair_feats(r) for r in rows], dtype=np.float32)


def pair_features(pairs: pl.DataFrame, qdf: pl.DataFrame, tdf: pl.DataFrame, split: str,
                  procs: int = 9, chunk: int = 20000) -> pl.DataFrame:
    """pairs: (q, t, ...). qdf has q + Q_COLS; tdf has t + T_COLS. Returns pairs with
    FEATURE_NAMES appended (row order preserved)."""
    if not _IDF_N:
        _load_idf_dicts(split)
    j = (pairs.select("q", "t").with_row_index("_i")
         .join(qdf.select(["q"] + Q_COLS).rename({c: f"q_{c}" for c in Q_COLS}), on="q", how="left")
         .join(tdf.select(["t"] + T_COLS).rename({c: f"t_{c}" for c in T_COLS}), on="t", how="left")
         .sort("_i"))
    cols = [f"q_{c}" for c in Q_COLS] + [f"t_{c}" for c in T_COLS]
    j = j.with_columns([pl.col(c).fill_null("") for c in cols if j[c].dtype == pl.Utf8])
    j = j.with_columns([pl.col(c).fill_null(0) for c in ("t_n_domain", "t_nonascii_name", "t_nonascii_addr")])
    rows = list(zip(*[j[c].to_list() for c in cols]))
    del j
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with get_context("fork").Pool(procs) as p:
        arr = np.concatenate(p.map(_chunk, chunks, chunksize=1)) if chunks else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    return pl.concat([pairs, pl.DataFrame(arr, schema=FEATURE_NAMES)], how="horizontal")


def add_frequency_features(pairs: pl.DataFrame, qdf: pl.DataFrame, tdf_counts: pl.DataFrame) -> pl.DataFrame:
    """Generic-name indicator: how many targets share the S1's sorted core name."""
    return (pairs.join(qdf.select("q", "n_sorted"), on="q", how="left")
            .join(tdf_counts, on="n_sorted", how="left")
            .with_columns(pl.col("name_freq_t").fill_null(0).log1p().cast(pl.Float32).alias("name_freq_t"))
            .drop("n_sorted"))


def add_context_features(df: pl.DataFrame, score: str, prefix: str) -> pl.DataFrame:
    """Competition features from a pair-level score column:
    - within the S1's candidate list: rank, gap to best, count within 0.05 of best
    - within the target's S1 competitors (exclusivity): rank, gap to best other S1."""
    s = pl.col(score)
    return df.with_columns(
        s.rank("ordinal", descending=True).over("q").cast(pl.Float32).alias(f"{prefix}_qrank"),
        (s.max().over("q") - s).alias(f"{prefix}_qgap"),
        s.rank("ordinal", descending=True).over("t").cast(pl.Float32).alias(f"{prefix}_trank"),
        (s - s.max().over("t")).alias(f"{prefix}_tgap_best"),
        pl.len().over("t").cast(pl.Float32).alias(f"{prefix}_t_ncomp"),
        pl.len().over("q").cast(pl.Float32).alias(f"{prefix}_q_ncand"),
    ).with_columns(
        # gap to the strongest *other* S1 for the same target (positive = this S1 wins)
        (s - pl.when(pl.col(f"{prefix}_trank") == 1).then(s.sort(descending=True).slice(1, 1).first().over("t"))
         .otherwise(s.max().over("t"))).fill_null(1.0).alias(f"{prefix}_tmargin"),
    )
