"""Stage-1 candidate filter.

Blocking is run wide (high recall, ~70 candidates per S1). A small LightGBM on
cheap features (blocking cosines/ranks, three rapidfuzz scores, house-number
agreement, per-S1 gaps) scores every pair; only pairs with p >= min_p (at most
max_keep per S1) survive. The survivors are exactly the pairs the full matching
model scores, and they are what candidate_pairs.tsv contains.

On a 30k-S1 train sample: wide union 71.7 cand/S1 recall 0.9888 -> stage-1
p>=0.001 keeps 10.0 cand/S1 with recall 0.9842 (vs 30 cand/S1 recall 0.9814
for the previous narrow blocking).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

import config


def stage1_features(c, s1, pool):
    """Cheap per-pair features for the stage-1 filter (c must hold whole S1 groups)."""
    ii, jj = c["i"].values, c["j"].values
    f = pd.DataFrame(index=c.index)
    for g in [col[:-4] for col in c.columns if col.endswith("_sim")]:
        f[f"{g}_sim"] = c[f"{g}_sim"].values
        f[f"{g}_rank"] = c[f"{g}_rank"].values.astype(np.float32)
    f["name_tset"] = process.cpdist(s1["name_core"].values[ii].tolist(), pool["name_core"].values[jj].tolist(),
                                    scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    f["addr_tset"] = process.cpdist(s1["addr_norm"].values[ii].tolist(), pool["addr_norm"].values[jj].tolist(),
                                    scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    f["compact_ratio"] = process.cpdist(s1["name_compact"].values[ii].tolist(),
                                        pool["name_compact"].values[jj].tolist(),
                                        scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    h1 = np.array([x.split(" ", 1)[0] for x in s1["house_numbers"].values[ii]])
    h2 = np.array([x.split(" ", 1)[0] for x in pool["house_numbers"].values[jj]])
    f["house"] = np.where((h1 == "") | (h2 == ""), 0, np.where(h1 == h2, 1, -1)).astype(np.float32)
    f["addr_empty"] = (pool["addr_norm"].values[jj] == "").astype(np.float32)
    grp = pd.Series(ii, index=c.index)
    for g in [x for x in f.columns if x.endswith("_sim")] + ["name_tset", "addr_tset"]:
        f[g + "_gap"] = f[g].groupby(grp).transform("max").values - f[g].values
    return f


def train_stage1(F, y):
    """Fit the stage-1 LightGBM."""
    p = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=50,
             verbose=-1, seed=config.SEED, num_threads=config.N_JOBS, deterministic=True,
             force_row_wise=True)
    return lgb.train(p, lgb.Dataset(F, y), config.STAGE1["rounds"])


def prune(c, s1, pool, model, chunk_rows=15_000_000):
    """Score candidate pairs with the stage-1 model and keep p >= min_p, top max_keep per S1."""
    c = c.sort_values("i", kind="stable").reset_index(drop=True)
    p = np.zeros(len(c), np.float32)
    iv = c["i"].values
    starts = np.flatnonzero(np.r_[True, iv[1:] != iv[:-1]])
    bounds = [0]
    for s in starts:
        if s - bounds[-1] >= chunk_rows:
            bounds.append(s)
    bounds.append(len(c))
    for a, b in zip(bounds[:-1], bounds[1:]):
        F = stage1_features(c.iloc[a:b], s1, pool)
        p[a:b] = model.predict(F[model.feature_name()])
    c["p1"] = p
    r = c.groupby("i")["p1"].rank(ascending=False, method="first").values
    keep = (p >= config.STAGE1["min_p"]) & (r <= config.STAGE1["max_keep"])
    return c[keep].drop(columns=["p1"]).reset_index(drop=True)
