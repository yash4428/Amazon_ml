"""Unseen-country simulator: which features break when the test country is new?

Builds exp05 features once for a sample of TRAIN S1 (full-train blocking from the
cache), then for several feature subsets trains LightGBM with
  - country folds (train US -> predict India, train India -> predict US), and
  - random group folds (for reference),
tunes the decision rule on each OOF and prints macro F0.5 per country.
Usage: python country_cv.py --model-s1 150000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from decide import apply_rule, macro_f05_fast, tune_rule  # noqa: E402
from features import (add_dup_counts, chunk_features, cluster_features, dup_features,  # noqa: E402
                      encode_extras, global_context, hmis_features,
                      house_mismatch_token_rates, pair_extras, token_counts)
from io_utils import read_truth  # noqa: E402
from model import group_folds  # noqa: E402
from normalize import apply_translit_dict, build_translit_dict, load_normalized  # noqa: E402
from run_pipeline import blocking_cached, label_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-s1", type=int, default=150000)
a = ap.parse_args()

d = os.path.join(config.ROOT, "dataset", "train")
s1, pool = load_normalized(d, "train")
truth = read_truth(d)
tl = build_translit_dict(s1, pool, truth)
s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
s1, pool = add_dup_counts(s1, pool)
cand = blocking_cached(s1, pool, "train")
hm = house_mismatch_token_rates(cand, s1, pool)
cand = pd.concat([cand, global_context(cand)], axis=1)
rng = np.random.RandomState(config.SEED)
pos = np.sort(rng.choice(len(s1), a.model_s1, replace=False))
sub = cand[np.isin(cand["i"].values, pos)].copy()
del cand
n_true = label_pairs(sub, s1, pool, truth)
y = sub["y"].values.astype(np.int8)
X = pd.concat([chunk_features(sub, s1, pool), cluster_features(sub, s1, pool),
               dup_features(sub, s1, pool)], axis=1)
ex = pair_extras(sub, s1, pool)
X["hmis_max"], X["hmis_mean"] = hmis_features(ex, hm)
cn = s1["country"].values[sub["i"].values]
folds = {"country": pd.factorize(cn, sort=True)[0].astype(np.int8), "group": group_folds(sub["i"].values, 2)}
# xtok encoding must follow the fold scheme (encodings from the other fold only)
for fname, fold in folds.items():
    xf = np.zeros((len(sub), 3), np.float32)
    for k in np.unique(fold):
        m = fold == k
        neg, pos_ = token_counts([e for e, t in zip(ex, ~m) if t], y[~m])
        mx, sm, unk = encode_extras([e for e, t in zip(ex, m) if t], neg, pos_)
        xf[m] = np.stack([mx, sm, unk], 1)
    X[f"xtok_max@{fname}"], X[f"xtok_sum@{fname}"], X[f"xtok_unknown@{fname}"] = xf[:, 0], xf[:, 1], xf[:, 2]
X["xtok_n"] = np.array([len(e) for e in ex], np.float32)
print("features", X.shape, flush=True)

local = np.searchsorted(pos, sub["i"].values)
nt = pd.Series(n_true.values[pos], index=np.arange(len(pos)))
cty_s1 = s1["country"].values[pos]
base = [c for c in X.columns if "@" not in c]
ctx_cand = [c for c in base if c.endswith(("_rank_cand", "_gap_cand", "_n_s1_for_cand"))]
cc = base + ["xtok_max@country", "xtok_sum@country", "xtok_unknown@country"]
fold = folds["country"]
oof = np.zeros(len(y), np.float32)
params = dict(config.LGB_PARAMS)
for k in np.unique(fold):
    tr, va = fold != k, fold == k
    mdl = lgb.train(params, lgb.Dataset(X.loc[tr, cc], y[tr]), 650)
    oof[va] = mdl.predict(X.loc[va, cc])
c = pd.DataFrame({"i": local, "j": sub["j"].values, "y": sub["y"].values, "score": oof})
c.to_parquet(os.path.join(config.ROOT, "runs", "country_cv_oof.parquet"), index=False)
nt.to_frame("n_true").to_parquet(os.path.join(config.ROOT, "runs", "country_cv_nt.parquet"))
np.save(os.path.join(config.ROOT, "runs", "country_cv_cty.npy"), cty_s1)
def per(prm):
    pr = apply_rule(c, *prm)
    return {cty: round(macro_f05_fast(pr[np.isin(pr["i"].values, np.flatnonzero(cty_s1 == cty))], nt,
                                      np.flatnonzero(cty_s1 == cty)), 4) for cty in sorted(set(cty_s1))}
for prm in [(0.78, 0.72, 0.0), (0.74, 0.02, 0.75), (0.86, 0.02, 0.85), (0.84, 0.8, 0.0), (0.9, 0.9, 0.0), (0.82, 0.02, 0.85)]:
    print("unseen-country OOF with params", prm, per(prm), flush=True)
prm, f = tune_rule(c, nt, np.arange(len(pos)), verbose=False)
print("tuned on unseen-country OOF", prm, per(prm), flush=True)
