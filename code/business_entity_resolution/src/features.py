"""Pair features for (S1, candidate) pairs.

All string similarities use rapidfuzz's parallel pairwise ``cpdist``. No
feature uses the raw country string, so an unseen country (France) gets the
same feature definitions as US/India.
"""
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz, process


W = -1  # rapidfuzz workers: all cores


def _pair(scorer, a, b, dtype=np.float32):
    """Elementwise similarity of two equal-length string lists."""
    return process.cpdist(a, b, scorer=scorer, workers=W, dtype=dtype).astype(np.float32)


def _num_set(s):
    """Set of number tokens in a space-joined house-number string."""
    return set(s.split()) if s else set()


def string_features(a, b):
    """Plain name/address similarity features for aligned S1 rows ``a`` and pool rows ``b``."""
    f = {}
    for col in ("name_norm", "name_core"):
        x, y = a[col].tolist(), b[col].tolist()
        f[f"{col}_ratio"] = _pair(fuzz.ratio, x, y)
        f[f"{col}_partial"] = _pair(fuzz.partial_ratio, x, y)
        f[f"{col}_tsort"] = _pair(fuzz.token_sort_ratio, x, y)
        f[f"{col}_tset"] = _pair(fuzz.token_set_ratio, x, y)
        f[f"{col}_jw"] = _pair(distance.JaroWinkler.normalized_similarity, x, y)
    x, y = a["name_compact"].tolist(), b["name_compact"].tolist()
    f["compact_ratio"] = _pair(fuzz.ratio, x, y)
    f["compact_partial"] = _pair(fuzz.partial_ratio, x, y)
    f["name_exact"] = (a["name_core"].values == b["name_core"].values).astype(np.float32)
    f["acronym_hit"] = ((a["acronym"].values == b["name_compact"].values) |
                        (b["acronym"].values == a["name_compact"].values)).astype(np.float32)
    f["first_tok_eq"] = np.array([p.split(" ", 1)[0] == q.split(" ", 1)[0]
                                  for p, q in zip(a["name_core"].values, b["name_core"].values)],
                                 dtype=np.float32)
    la = a["name_core"].str.len().values.astype(np.float32)
    lb = b["name_core"].str.len().values.astype(np.float32)
    f["name_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    # DBA alternatives: best token_set over alternative names on either side
    dba_b = b["dba"].values
    alt = np.zeros(len(a), np.float32)
    idx = np.flatnonzero(dba_b != "")
    for k in idx:
        alt[k] = max(fuzz.token_set_ratio(a["name_core"].values[k], p) for p in dba_b[k].split(" | "))
    f["dba_best"] = np.maximum(alt, f["name_core_tset"])

    x, y = a["addr_norm"].tolist(), b["addr_norm"].tolist()
    f["addr_ratio"] = _pair(fuzz.ratio, x, y)
    f["addr_partial"] = _pair(fuzz.partial_ratio, x, y)
    f["addr_tsort"] = _pair(fuzz.token_sort_ratio, x, y)
    f["addr_tset"] = _pair(fuzz.token_set_ratio, x, y)
    f["addr_empty_b"] = (b["addr_norm"].values == "").astype(np.float32)
    f["addr_len_b"] = b["addr_norm"].str.len().values.astype(np.float32)
    f["addr_len_a"] = a["addr_norm"].str.len().values.astype(np.float32)

    pa, pb = a["postcode"].values, b["postcode"].values
    f["pc_state"] = np.where((pa == "") | (pb == ""), 0, np.where(pa == pb, 1, -1)).astype(np.float32)
    ha, hb = a["house_numbers"].values, b["house_numbers"].values
    first_eq, overlap, conflict = [], [], []
    for p, q in zip(ha, hb):
        sp_, sq = _num_set(p), _num_set(q)
        if not sp_ or not sq:
            first_eq.append(0); overlap.append(0); conflict.append(0)
            continue
        fe = p.split(" ", 1)[0] == q.split(" ", 1)[0]
        ov = len(sp_ & sq)
        first_eq.append(1 if fe else -1)
        overlap.append(ov / min(len(sp_), len(sq)))
        conflict.append(1 if ov == 0 else 0)
    f["house_first"] = np.array(first_eq, np.float32)
    f["house_overlap"] = np.array(overlap, np.float32)
    f["house_conflict"] = np.array(conflict, np.float32)
    f["landmark_any"] = (a["landmark"].values | b["landmark"].values).astype(np.float32)
    return pd.DataFrame(f)


def context_features(c, score_col, cand_side=True):
    """Within-S1 (and optionally within-candidate) competition features from one score column.

    c needs columns i, j and ``score_col``. Candidate-side features compare this S1
    with every other S1 that retrieved the same pool record, so they must be
    computed on the full candidate set, never on a subsample of S1.
    """
    out = {}
    s = c[score_col].values
    g = c.groupby("i")[score_col]
    best = g.transform("max").values
    out[f"{score_col}_gap"] = (best - s).astype(np.float32)
    out[f"{score_col}_rank_s1"] = g.rank(ascending=False, method="min").values.astype(np.float32)
    close = pd.Series((best - s) <= 0.05, index=c.index)
    out[f"{score_col}_n_close"] = close.groupby(c["i"]).transform("sum").values.astype(np.float32)
    if cand_side:
        gj = c.groupby("j")[score_col]
        bestj = gj.transform("max").values
        out[f"{score_col}_rank_cand"] = gj.rank(ascending=False, method="min").values.astype(np.float32)
        out[f"{score_col}_gap_cand"] = (bestj - s).astype(np.float32)
        out[f"{score_col}_n_s1_for_cand"] = gj.transform("size").values.astype(np.float32)
    return pd.DataFrame(out, index=c.index)


def global_context(cand):
    """Blocking-score context features over the FULL candidate set (all S1)."""
    parts = [context_features(cand, "combo_sim"), context_features(cand, "name_c4_sim")]
    out = pd.concat(parts, axis=1)
    out["n_cand_s1"] = cand.groupby("i")["j"].transform("size").values.astype(np.float32)
    return out.add_prefix("ctx_")


def chunk_features(c, s1, pool):
    """String features + blocking columns + S1-side string contexts for a chunk of pairs.

    ``c`` must contain whole S1 groups (all candidates of each S1 present).
    """
    a = s1.iloc[c["i"].values].reset_index(drop=True)
    b = pool.iloc[c["j"].values].reset_index(drop=True)
    f = string_features(a, b)
    f["src3"] = (b["src"].values == 3).astype(np.float32)
    f.index = c.index
    for col in c.columns:
        if col.endswith("_sim") or col.endswith("_rank") or col.startswith("ctx_"):
            f[col] = c[col].values.astype(np.float32)
    tmp = pd.DataFrame({"i": c["i"].values, "j": c["j"].values,
                        "name_sim": f["name_core_tset"].values,
                        "addr_sim": f["addr_tset"].values}, index=c.index)
    for col in ("name_sim", "addr_sim"):
        f = pd.concat([f, context_features(tmp, col, cand_side=False)], axis=1)
    return f


# --------------------------------------------------------------------------
# Name "extra token" encoding and house-number cluster features.
#
# Hard negatives in this data are sibling branches of the S1 business: the same
# name plus an extra word (east, holdings, riverside, ...) and a changed house
# number. True copies get a different kind of extra word (center, inc, services)
# or none. We learn, from labelled TRAIN pairs only, how often each extra token
# appears on negative vs positive pairs (out-of-fold for the training rows).
# Tokens never seen in training (e.g. French branch words) get a neutral 0.
# --------------------------------------------------------------------------
def extra_tokens(n1, n2):
    """Tokens of candidate name (n2) not in the S1 name (n1), and S1 tokens missing from n2."""
    t1, t2 = set(n1.split()), set(n2.split())
    return t2 - t1, t1 - t2


def token_counts(extras, y):
    """Count, per token, how many negative / positive pairs carry it as an extra token."""
    neg, pos = {}, {}
    for ex, lab in zip(extras, y):
        d = pos if lab else neg
        for w in ex:
            d[w] = d.get(w, 0) + 1
    return neg, pos


def encode_extras(extras, neg, pos, prior=2.0, min_count=5):
    """Per pair: max / sum smoothed log-odds(negative) over extra tokens, and #unknown tokens."""
    mx = np.zeros(len(extras), np.float32)
    sm = np.zeros(len(extras), np.float32)
    unk = np.zeros(len(extras), np.float32)
    for k, ex in enumerate(extras):
        best, tot = 0.0, 0.0
        for w in ex:
            n, p = neg.get(w, 0), pos.get(w, 0)
            if n + p < min_count:
                unk[k] += 1
                continue
            v = np.log((n + prior) / (p + prior))
            tot += v
            if v > best:
                best = v
        mx[k], sm[k] = best, tot
    return mx, sm, unk


def pair_extras(c, s1, pool):
    """List of extra-token sets (candidate vs S1 core names) for every pair in ``c``."""
    n1 = s1["name_core"].values[c["i"].values]
    n2 = pool["name_core"].values[c["j"].values]
    return [set(b.split()) - set(a.split()) for a, b in zip(n1, n2)]


def cluster_features(c, s1, pool):
    """House-number agreement features, including agreement with the S1's other candidates.

    A distractor branch usually comes as several noisy copies that share a house
    number different from the S1's; true copies mostly share the S1's number.
    """
    h1 = s1["house_numbers"].values[c["i"].values]
    h2 = pool["house_numbers"].values[c["j"].values]
    f1 = np.array([x.split(" ", 1)[0] for x in h1])
    f2 = np.array([x.split(" ", 1)[0] for x in h2])

    def num(x):
        d = "".join(ch for ch in x if ch.isdigit())[:9]
        return float(d) if d else np.nan
    v1 = np.array([num(x) for x in f1])
    v2 = np.array([num(x) for x in f2])
    out = {}
    diff = np.abs(v1 - v2)
    out["house_absdiff_log"] = np.where(np.isnan(diff), -1, np.log1p(diff)).astype(np.float32)
    out["house_reldiff"] = np.where(np.isnan(diff), -1,
                                    diff / np.maximum(np.maximum(v1, v2), 1)).astype(np.float32)
    # S1 first number appears anywhere among candidate numbers
    out["house_s1_in_cand"] = np.array([(a != "" and a in b.split()) for a, b in zip(f1, h2)],
                                       np.float32)
    tmp = pd.DataFrame({"i": c["i"].values, "h2": f2, "eq": (f1 == f2) & (f1 != "")})
    grp = tmp.groupby(["i", "h2"])["i"].transform("size").values
    out["house_cluster_size"] = np.where(f2 == "", 0, grp - 1).astype(np.float32)
    out["n_cand_house_eq_s1"] = tmp.groupby("i")["eq"].transform("sum").values.astype(np.float32)
    return pd.DataFrame(out, index=c.index)


def add_dup_counts(s1, pool):
    """Name-ambiguity counts (unsupervised): how many S1 records share each core name.

    s1["dup_s1"]   : number of S1 records (same country) with this S1's core name
    pool["dup_s1"] : number of S1 records (same country) with this candidate's core name
    A name-only candidate (empty address) is only safe when its name is unique among S1.
    """
    key_s1 = s1["country"] + "|" + s1["name_core"]
    vc = key_s1.value_counts()
    s1 = s1.assign(dup_s1=key_s1.map(vc).values.astype(np.float32))
    key_p = pool["country"] + "|" + pool["name_core"]
    pool = pool.assign(dup_s1=key_p.map(vc).fillna(0).values.astype(np.float32))
    return s1, pool


def dup_features(c, s1, pool):
    """Per-pair name-ambiguity features from add_dup_counts columns."""
    return pd.DataFrame({"s1_name_dup": s1["dup_s1"].values[c["i"].values],
                         "cand_name_dup": pool["dup_s1"].values[c["j"].values]}, index=c.index)
