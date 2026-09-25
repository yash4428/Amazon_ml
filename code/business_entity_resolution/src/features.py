"""Pair features for (S1, candidate) pairs.

All string similarities use rapidfuzz's parallel pairwise ``cpdist``. No
feature uses the raw country string, so an unseen country (France) gets the
same feature definitions as US/India.
"""
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz, process

import config

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


def context_features(c, score_col):
    """Within-S1 and within-candidate competition features from a pair-level score column."""
    out = {}
    s = c[score_col].values
    g = c.groupby("i")[score_col]
    best = g.transform("max").values
    out[f"{score_col}_gap"] = best - s
    out[f"{score_col}_rank_s1"] = g.rank(ascending=False, method="min").values.astype(np.float32)
    out[f"{score_col}_n_close"] = (c.assign(_c=(best - s) <= 0.05)
                                   .groupby("i")["_c"].transform("sum").values.astype(np.float32))
    gj = c.groupby("j")[score_col]
    bestj = gj.transform("max").values
    out[f"{score_col}_rank_cand"] = gj.rank(ascending=False, method="min").values.astype(np.float32)
    out[f"{score_col}_gap_cand"] = bestj - s
    out[f"{score_col}_n_s1_for_cand"] = gj.transform("size").values.astype(np.float32)
    return pd.DataFrame(out, index=c.index)


def build_features(cand, s1, pool, chunk=5_000_000):
    """Full feature frame for a candidate frame (columns i, j + blocking sims/ranks)."""
    parts = []
    for lo in range(0, len(cand), chunk):
        c = cand.iloc[lo:lo + chunk]
        a = s1.iloc[c["i"].values].reset_index(drop=True)
        b = pool.iloc[c["j"].values].reset_index(drop=True)
        f = string_features(a, b)
        f["src3"] = (b["src"].values == 3).astype(np.float32)
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)
    feats.index = cand.index
    for col in cand.columns:
        if col.endswith("_sim") or col.endswith("_rank"):
            feats[col] = cand[col].values.astype(np.float32)
    feats["n_cand_s1"] = cand.groupby("i")["j"].transform("size").values.astype(np.float32)
    tmp = cand[["i", "j"]].copy()
    tmp["name_sim"] = feats["name_core_tset"].values
    tmp["addr_sim"] = feats["addr_tset"].values
    tmp["combo"] = cand["combo_sim"].values
    for col in ("name_sim", "addr_sim", "combo"):
        feats = pd.concat([feats, context_features(tmp, col).set_index(feats.index)], axis=1)
    return feats
