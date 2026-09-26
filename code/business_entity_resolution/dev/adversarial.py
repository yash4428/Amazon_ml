"""Adversarial validation: which model features look different on test (per country) than in training?

Builds exp17 features (same functions as run_pipeline) for a sample of TRAIN S1 (crowded pool, like the model's
training data) and samples of TEST S1 per country, then for each test country trains a LightGBM to separate
train pairs from that country's test pairs. AUC ≈ 0.5 = no shift; the top features by gain are what shifted.
Also prints per-feature mean/quantiles for the most shifted features.
Usage: python adversarial.py --n 20000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from features import (add_dup_counts, add_raw_names, add_synthetic_branches, chunk_features,  # noqa: E402
                      cluster_features, dup_features, encode_extras, global_context, hmis_features,
                      house_mismatch_token_rates, pair_extras, token_counts)
from io_utils import read_truth  # noqa: E402
from normalize import apply_translit_dict, build_translit_dict, load_normalized  # noqa: E402
from run_pipeline import blocking_cached, label_pairs, stage1_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
a = ap.parse_args()
rng = np.random.RandomState(0)


def build(s1, pool, tag, n_per_country, tok=None, truth=None):
    """Feature frame for a sample of S1 per country (exp17 feature set)."""
    cand = blocking_cached(s1, pool, tag, post=None, post_key=stage1_key())
    hm = house_mismatch_token_rates(cand, s1, pool)
    cand = pd.concat([cand, global_context(cand, s1, pool)], axis=1)
    out = []
    for c in sorted(set(s1["country"])):
        idx = np.flatnonzero(s1["country"].values == c)
        pick = rng.choice(idx, min(n_per_country, len(idx)), replace=False)
        sub = cand[np.isin(cand["i"].values, pick)].copy()
        X = pd.concat([chunk_features(sub, s1, pool), cluster_features(sub, s1, pool),
                       dup_features(sub, s1, pool)], axis=1)
        ex = pair_extras(sub, s1, pool)
        X["hmis_max"], X["hmis_mean"] = hmis_features(ex, hm)
        if tok is None:
            label_pairs(sub, s1, pool, truth)
            tok = token_counts(ex, sub["y"].values)
        mx, sm, unk = encode_extras(ex, *tok)
        X["xtok_max"], X["xtok_sum"], X["xtok_unknown"] = mx, sm, unk
        X["xtok_n"] = np.array([len(e) for e in ex], np.float32)
        X = X.drop(columns=[k for k in config.DROP_FEATURES if k in X.columns])
        X["_cty"] = c
        out.append(X)
        print(f"[{tag}] {c}: {len(X)} pairs from {len(pick)} S1", flush=True)
    return pd.concat(out, ignore_index=True), tok


tr_dir = os.path.join(config.ROOT, "dataset", "train")
s1, pool = load_normalized(tr_dir, "train")
truth = read_truth(tr_dir)
tl = build_translit_dict(s1, pool, truth)
s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
pool = add_synthetic_branches(pool, truth, 0.5)
s1, pool = add_dup_counts(s1, pool)
s1, pool = add_raw_names(s1), add_raw_names(pool)
XT, tok = build(s1, pool, "train", a.n, truth=truth)
del s1, pool
s1, pool = load_normalized(os.path.join(config.ROOT, "dataset", "test"), "test")
s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
s1, pool = add_dup_counts(s1, pool)
s1, pool = add_raw_names(s1), add_raw_names(pool)
XE, _ = build(s1, pool, "test", a.n, tok=tok)
cols = [c for c in XT.columns if c != "_cty"]
XT.to_parquet(os.path.join(config.ROOT, "runs", "adv_train.parquet")); XE.to_parquet(os.path.join(config.ROOT, "runs", "adv_test.parquet"))
p = dict(objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1, num_threads=8, seed=0)
for c in sorted(set(XE["_cty"])):
    A = XT.loc[XT["_cty"] == c, cols] if c in set(XT["_cty"]) else XT[cols]   # same country when available
    B = XE.loc[XE["_cty"] == c, cols]
    X = pd.concat([A, B], ignore_index=True); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    fold = np.random.RandomState(1).randint(0, 2, len(X)); pred = np.zeros(len(X)); imp = np.zeros(len(cols))
    for k in (0, 1):
        m = lgb.train(p, lgb.Dataset(X[fold != k], y[fold != k]), 200)
        pred[fold == k] = m.predict(X[fold == k]); imp += m.feature_importance("gain")
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, pred)
    top = pd.Series(imp, index=cols).sort_values(ascending=False).head(12)
    print(f"\n=== train vs TEST {c}: adversarial AUC {auc:.3f}")
    for f_, g in top.items():
        print(f"   {f_:34s} gain {g:12.0f}  train mean {A[f_].mean():9.3f} q50 {A[f_].median():8.3f} | test mean {B[f_].mean():9.3f} q50 {B[f_].median():8.3f}")
