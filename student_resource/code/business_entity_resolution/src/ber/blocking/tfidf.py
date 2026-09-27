"""Sparse TF-IDF top-K retrieval (char or word n-grams), scoped by country.

The vectorizer is fit on the target corpus; queries and targets are L2-normalized so
the sparse dot product is cosine similarity. Top-K per query uses sparse_dot_topn
(multi-threaded C++). To stay within 16 GB, the target matrix is stored per country
already transposed, and only one country block is resident at a time."""
import gc
import hashlib
import json

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ..paths import CACHE


def cfg_hash(field, analyzer="char_wb", ngram=(3, 3), min_df=2, max_df=0.05, **_):
    from ..normalize import NORM_VERSION
    cfg = dict(field=field, analyzer=analyzer, ngram=list(ngram), min_df=min_df, max_df=max_df, norm=NORM_VERSION)
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10], cfg


def build_matrices(split, field, analyzer="char_wb", ngram=(3, 3), min_df=2, max_df=0.05, field_expr=None):
    """Fits TF-IDF on targets, writes Q (queries CSR) and per-country transposed target
    blocks. Returns the cache prefix."""
    from ..data import load_queries, load_targets
    h, cfg = cfg_hash(field, analyzer, ngram, min_df, max_df)
    pre = CACHE / f"tfidf_{split}_{h}"
    if (CACHE / f"tfidf_{split}_{h}.json").exists():
        return pre
    expr = field_expr if field_expr is not None else pl.col(field)
    need = sorted(expr.meta.root_names())
    T = load_targets(split, need + ["country"]).select("country", x=expr.fill_null(""))
    tc = T["country"].to_numpy()
    tt = T["x"].to_list()
    del T
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram), min_df=min_df, max_df=max_df,
                          sublinear_tf=True, dtype=np.float32, lowercase=False)
    Tm = vec.fit_transform(tt).astype(np.float32).tocsr()
    del tt
    gc.collect()
    qt = load_queries(split, need).select(x=expr.fill_null(""))["x"].to_list()
    sp.save_npz(f"{pre}_Q.npz", vec.transform(qt).astype(np.float32).tocsr(), compressed=False)
    del qt, vec
    countries = sorted(set(tc.tolist()))
    for c in countries:
        tsel = np.flatnonzero(tc == c).astype(np.int32)
        np.save(f"{pre}_tsel_{c}.npy", tsel)
        sp.save_npz(f"{pre}_TT_{c}.npz", Tm[tsel].T.tocsr(), compressed=False)
        gc.collect()
    del Tm
    (CACHE / f"tfidf_{split}_{h}.json").write_text(json.dumps({**cfg, "countries": countries}))
    return pre


def topk_block(pre, qmeta: pl.DataFrame, qs: np.ndarray, k: int = 20, threshold: float = 0.0,
               n_threads: int = 9, batch: int = 20000) -> pl.DataFrame:
    """Top-k targets for query rows `qs` within the query's country."""
    from pathlib import Path
    pre = Path(pre)
    Q = sp.load_npz(f"{pre}_Q.npz")
    qc = qmeta["country"].to_numpy()
    out = []
    for c in np.unique(qc[qs]):
        try:
            tsel = np.load(f"{pre}_tsel_{c}.npy")
            TT = sp.load_npz(f"{pre}_TT_{c}.npz")
        except FileNotFoundError:
            continue  # no targets for this country
        qsel = qs[qc[qs] == c]
        for i in range(0, len(qsel), batch):
            if i % (10 * batch) == 0:
                print(f"  fwd {pre.name} {c}: {i}/{len(qsel)}", flush=True)
            qb = qsel[i:i + batch]
            R = sp_matmul_topn(Q[qb], TT, top_n=k, threshold=threshold, n_threads=n_threads).tocoo()
            out.append(pl.DataFrame({"q": qb[R.row].astype(np.int32), "t": tsel[R.col],
                                     "score": R.data.astype(np.float32)}))
        del TT
        gc.collect()
    return pl.concat(out) if out else pl.DataFrame(schema={"q": pl.Int32, "t": pl.Int32, "score": pl.Float32})


def ensure_reverse(pre, qmeta: pl.DataFrame):
    from pathlib import Path
    pre = Path(pre)
    """Per-country transposed S1 matrices for reverse (target -> S1) retrieval."""
    qc = qmeta["country"].to_numpy()
    Q = None
    for c in np.unique(qc):
        if (CACHE / f"{pre.name}_QT_{c}.npz").exists():
            continue
        if Q is None:
            Q = sp.load_npz(f"{pre}_Q.npz")
        qsel = np.flatnonzero(qc == c).astype(np.int32)
        np.save(f"{pre}_qsel_{c}.npy", qsel)
        sp.save_npz(f"{pre}_QT_{c}.npz", Q[qsel].T.tocsr(), compressed=False)


def reverse_topk_block(pre, qmeta: pl.DataFrame, k: int = 3, threshold: float = 0.0,
                       n_threads: int = 9, batch: int = 50000, t_subset=None) -> pl.DataFrame:
    """For every target, its top-k S1 in the same country. Returns (q, t, score)."""
    from pathlib import Path
    pre = Path(pre)
    ensure_reverse(pre, qmeta)
    out = []
    for c in np.unique(qmeta["country"].to_numpy()):
        try:
            tsel = np.load(f"{pre}_tsel_{c}.npy")
            Tc = sp.load_npz(f"{pre}_TT_{c}.npz").T.tocsr()
        except FileNotFoundError:
            continue
        if t_subset is not None:
            m = np.isin(tsel, t_subset)
            tsel, Tc = tsel[m], Tc[m]
        qsel = np.load(f"{pre}_qsel_{c}.npy")
        QT = sp.load_npz(f"{pre}_QT_{c}.npz")
        for i in range(0, Tc.shape[0], batch):
            if i % (20 * batch) == 0:
                print(f"  rev {pre.name} {c}: {i}/{Tc.shape[0]}", flush=True)
            R = sp_matmul_topn(Tc[i:i + batch], QT, top_n=k, threshold=threshold, n_threads=n_threads).tocoo()
            out.append(pl.DataFrame({"q": qsel[R.col], "t": tsel[i + R.row].astype(np.int32),
                                     "score": R.data.astype(np.float32)}))
        del Tc, QT
        gc.collect()
    return pl.concat(out)
