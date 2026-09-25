"""Candidate generation (blocking).

For each country label separately (all true pairs share the label, and an
unseen label such as France simply forms its own block), every generator
builds a sparse binary feature matrix over S1 + pool records, weights features
by IDF computed on that country's records (unsupervised, no labels), drops
features that are too common to be useful for retrieval (df > max_df) or
unique (df < 2), L2-normalises rows and takes the top-K pool records per S1
by cosine similarity, via chunked sparse matrix multiplication.

The union over generators is the candidate set; per generator we keep the
cosine and the rank, which later become features.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer

import config


# ------------------------------------------------------------------ analyzers
def an_name_tok(row):
    """Name word tokens from the core name and any DBA alternatives."""
    toks = set(row[0].split())
    if row[1]:
        toks.update(row[1].replace("|", " ").split())
    return list(toks)


def an_name_c4(row):
    """Character 4-grams of the space-free core name, with boundary markers."""
    s = "^" + row[0] + "$"
    if len(s) < 4:
        return [s]
    return list({s[i:i + 4] for i in range(len(s) - 3)})


def an_addr_tok(row):
    """Address word unigrams + adjacent bigrams (bigrams make rare combos like '95 church')."""
    t = row[0].split()
    return list(set(t) | {t[i] + "_" + t[i + 1] for i in range(len(t) - 1)})


def _uni_bi(tokens, prefix):
    """Unigrams + adjacent bigrams of a token list, tagged with a field prefix."""
    return [prefix + x for x in tokens] + [prefix + tokens[i] + "_" + tokens[i + 1]
                                           for i in range(len(tokens) - 1)]


def an_combo(row):
    """Name (core) + address unigrams/bigrams in one space, so pairs agreeing on both rank first."""
    return list(set(_uni_bi(row[0].split(), "n:") + _uni_bi(row[1].split(), "a:")))


def an_name_bi(row):
    """Name unigrams + bigrams (core name and DBA alternatives)."""
    out = _uni_bi(row[0].split(), "")
    for alt in row[1].split("|") if row[1] else []:
        out += _uni_bi(alt.split(), "")
    return list(set(out))


SPACES = {
    "combo": (an_combo, ["name_core", "addr_norm"]),
    "name_bi": (an_name_bi, ["name_core", "dba"]),
    "name_tok": (an_name_tok, ["name_core", "dba"]),
    "name_c4": (an_name_c4, ["name_compact"]),
    "addr_tok": (an_addr_tok, ["addr_norm"]),
}


def build_matrix(docs, analyzer, max_df, n_s1):
    """Binary doc-feature matrix -> IDF-weighted, df-pruned, L2-normalised CSR (float32).

    Returns (S1 matrix, pool matrix).
    """
    cv = CountVectorizer(analyzer=analyzer, binary=True, dtype=np.float32)
    X = cv.fit_transform(docs).tocsc()
    df = np.diff(X.indptr)
    keep = (df >= 2) & (df <= max_df)
    X = X[:, np.flatnonzero(keep)]
    idf = np.log(X.shape[0] / np.diff(X.indptr).astype(np.float32)).astype(np.float32)
    X = X.tocsr()
    X.data = idf[X.indices]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = sp.diags((1.0 / norms).astype(np.float32)) @ X
    X = X.tocsr()
    return X[:n_s1], X[n_s1:]


def _topk_chunk(S, PT, lo, hi, k):
    """Top-k columns per row of S[lo:hi] @ PT. Returns (rows, cols, scores, ranks).

    Uses a per-row argpartition, which is far cheaper than sorting all non-zeros
    when rows have tens of thousands of partial matches.
    """
    R = (S[lo:hi] @ PT).tocsr()
    ip, ind, dat = R.indptr, R.indices, R.data
    rows, cols, vals, ranks = [], [], [], []
    for r in range(hi - lo):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        d = dat[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k - 1)[:k]
        else:
            sel = np.arange(b - a)
        sel = sel[np.argsort(-d[sel], kind="stable")]
        rows.append(np.full(len(sel), r + lo, np.int64))
        cols.append(ind[a:b][sel].astype(np.int64))
        vals.append(d[sel])
        ranks.append(np.arange(len(sel), dtype=np.int64))
    if not rows:
        e = np.empty(0, np.int64)
        return e, e, np.empty(0, np.float32), e
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals),
            np.concatenate(ranks))


def topk_cosine(S, P, k, chunk=config.BLOCK_CHUNK, n_jobs=1):
    """Top-k pool rows per S1 row by cosine over sparse L2-normalised features.

    Chunks run in separate processes (scipy sparse matmul holds the GIL); joblib
    memory-maps the large pool matrix instead of copying it to each worker.
    """
    PT = P.T.tocsr()
    spans = [(lo, min(lo + chunk, S.shape[0])) for lo in range(0, S.shape[0], chunk)]
    if n_jobs == 1 or len(spans) == 1:
        res = [_topk_chunk(S, PT, lo, hi, k) for lo, hi in spans]
    else:
        res = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
            delayed(_topk_chunk)(S[lo:hi], PT, 0, hi - lo, k) for lo, hi in spans)
        res = [(r[0] + lo, r[1], r[2], r[3]) for r, (lo, hi) in zip(res, spans)]
    if not res:
        e = np.empty(0, np.int64)
        return e, e, np.empty(0, np.float32), e
    return tuple(np.concatenate([r[i] for r in res]) for i in range(4))


def generate_candidates(s1, pool, generators=None, n_jobs=config.N_JOBS, verbose=True):
    """Union of top-K candidates from every generator, blocked by country.

    Parameters
    ----------
    s1, pool : normalised frames (see normalize.normalize_frame), any index.
    generators : dict name -> dict(space, k, max_df); default config.BLOCKING.

    Returns a DataFrame with integer positions ``i`` (row of s1) and ``j`` (row of
    pool) plus, per generator g, columns ``<g>_sim`` (cosine, 0 if not found) and
    ``<g>_rank`` (rank, 99 if not found).
    """
    generators = generators or config.BLOCKING
    s1 = s1.reset_index(drop=True)
    pool = pool.reset_index(drop=True)
    parts = []
    for country in sorted(set(s1["country"])):
        si = np.flatnonzero(s1["country"].values == country)
        pj = np.flatnonzero(pool["country"].values == country)
        if len(si) == 0 or len(pj) == 0:
            continue
        per_gen = []
        for g, gc in generators.items():
            t0 = time.time()
            analyzer, cols = SPACES[gc["space"]]
            docs = list(zip(*[pd.concat([s1[c].iloc[si], pool[c].iloc[pj]]).tolist() for c in cols]))
            S, P = build_matrix(docs, analyzer, gc["max_df"], len(si))
            r, c, v, rk = topk_cosine(S, P, gc["k"], n_jobs=n_jobs)
            per_gen.append(pd.DataFrame({"i": si[r], "j": pj[c], f"{g}_sim": v,
                                         f"{g}_rank": rk.astype(np.int16)}))
            if verbose:
                print(f"  [{country}] {g}: {len(si)} S1 x {len(pj)} pool, "
                      f"{len(r)} pairs, {time.time() - t0:.1f}s", flush=True)
        m = per_gen[0]
        for d in per_gen[1:]:
            m = m.merge(d, on=["i", "j"], how="outer")
        parts.append(m)
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["i", "j"])
    for g in generators:
        cand[f"{g}_sim"] = cand[f"{g}_sim"].fillna(0).astype(np.float32)
        cand[f"{g}_rank"] = cand[f"{g}_rank"].fillna(99).astype(np.int16)
    cand["i"] = cand["i"].astype(np.int64)
    cand["j"] = cand["j"].astype(np.int64)
    return cand
