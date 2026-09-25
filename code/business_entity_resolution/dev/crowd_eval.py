"""A/B test: does training under test-like distractor density help on test-like data?

Builds exp06-style features for the same sampled TRAIN S1 twice:
  N = normal train pool, C = pool + synthetic fake branches (add_synthetic_branches).
Model A trains on N rows, model B on C rows (same S1 folds). Each model's decision
thresholds are tuned on its OWN training-condition OOF (as in the real pipeline) and
then applied to both validation conditions. The comparison that matters for test is
A-on-C vs B-on-C; B-on-N checks that B does not hurt normal conditions.
Usage: python crowd_eval.py --model-s1 150000 --crowd 0.5
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
from features import (add_dup_counts, add_synthetic_branches, chunk_features,  # noqa: E402
                      cluster_features, dup_features, encode_extras, global_context,
                      hmis_features, house_mismatch_token_rates, pair_extras, token_counts)
from io_utils import read_truth  # noqa: E402
from normalize import apply_translit_dict, build_translit_dict, load_normalized  # noqa: E402
from run_pipeline import blocking_cached, label_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-s1", type=int, default=150000)
ap.add_argument("--crowd", type=float, default=0.5)
ap.add_argument("--folds", type=int, default=3)
a = ap.parse_args()

d = os.path.join(config.ROOT, "dataset", "train")
s1_0, pool_0 = load_normalized(d, "train")
truth = read_truth(d)
tl = build_translit_dict(s1_0, pool_0, truth)
s1_0, pool_0 = apply_translit_dict(s1_0, tl), apply_translit_dict(pool_0, tl)
rng = np.random.RandomState(config.SEED)
pos = np.sort(rng.choice(len(s1_0), a.model_s1, replace=False))
fold_of = np.full(len(s1_0), -1, np.int8)
fold_of[pos] = np.random.RandomState(1).randint(0, a.folds, len(pos))
nt = pd.Series([len(truth.get(s, ())) for s in s1_0["entity_id"].values[pos]], index=np.arange(len(pos)))


def build(pool_in, tag):
    """Features / labels / folds for the sampled S1 against one pool version."""
    s1, pool = add_dup_counts(s1_0, pool_in)
    cand = blocking_cached(s1, pool, "train")
    hm = house_mismatch_token_rates(cand, s1, pool)
    cand = pd.concat([cand, global_context(cand)], axis=1)
    sub = cand[np.isin(cand["i"].values, pos)].copy()
    del cand
    label_pairs(sub, s1, pool, truth)
    y = sub["y"].values.astype(np.int8)
    fold = fold_of[sub["i"].values]
    X = pd.concat([chunk_features(sub, s1, pool), cluster_features(sub, s1, pool),
                   dup_features(sub, s1, pool)], axis=1)
    ex = pair_extras(sub, s1, pool)
    X["hmis_max"], X["hmis_mean"] = hmis_features(ex, hm)
    xf = np.zeros((len(sub), 3), np.float32)
    for k in range(a.folds):
        m = fold == k
        neg, p_ = token_counts([e for e, t in zip(ex, ~m) if t], y[~m])
        mx, sm, unk = encode_extras([e for e, t in zip(ex, m) if t], neg, p_)
        xf[m] = np.stack([mx, sm, unk], 1)
    X["xtok_max"], X["xtok_sum"], X["xtok_unknown"] = xf[:, 0], xf[:, 1], xf[:, 2]
    X["xtok_n"] = np.array([len(e) for e in ex], np.float32)
    X = X.drop(columns=[c for c in config.DROP_FEATURES if c in X.columns])
    local = np.searchsorted(pos, sub["i"].values)
    n_syn = int(pd.Series(pool["entity_id"].values[sub["j"].values]).str.startswith("SYN").sum())
    print(f"[{tag}] pairs {len(sub)}, positives {int(y.sum())}, synthetic-branch rows {n_syn}", flush=True)
    return X, y, fold, local, sub["j"].values


XN, yN, fN, lN, jN = build(pool_0, "normal")
XC, yC, fC, lC, jC = build(add_synthetic_branches(pool_0, truth, a.crowd), "crowded")
cols = list(XN.columns)
params = dict(config.LGB_PARAMS)
oof = {k: np.zeros(n, np.float32) for k, n in [("A_N", len(yN)), ("A_C", len(yC)), ("B_N", len(yN)), ("B_C", len(yC))]}
for k in range(a.folds):
    mA = lgb.train(params, lgb.Dataset(XN.loc[fN != k, cols], yN[fN != k]), 650)
    mB = lgb.train(params, lgb.Dataset(XC.loc[fC != k, cols], yC[fC != k]), 650)
    oof["A_N"][fN == k] = mA.predict(XN.loc[fN == k, cols])
    oof["A_C"][fC == k] = mA.predict(XC.loc[fC == k, cols])
    oof["B_N"][fN == k] = mB.predict(XN.loc[fN == k, cols])
    oof["B_C"][fC == k] = mB.predict(XC.loc[fC == k, cols])
    print(f"fold {k} done", flush=True)


def frame(key):
    """OOF pair frame for one (model, condition) combination."""
    loc, j, y = (lN, jN, yN) if key.endswith("N") else (lC, jC, yC)
    return pd.DataFrame({"i": loc, "j": j, "y": y.astype(bool), "score": oof[key]})


cty = s1_0["country"].values[pos]
prmA, fA = tune_rule(frame("A_N"), nt, np.arange(len(pos)), verbose=False)
prmB, fB = tune_rule(frame("B_C"), nt, np.arange(len(pos)), verbose=False)
for key, prm in [("A_N", prmA), ("A_C", prmA), ("B_N", prmB), ("B_C", prmB)]:
    pr = apply_rule(frame(key), *prm)
    tot = macro_f05_fast(pr, nt, np.arange(len(pos)))
    per = {c: round(macro_f05_fast(pr[np.isin(pr["i"].values, np.flatnonzero(cty == c))], nt,
                                   np.flatnonzero(cty == c)), 4) for c in sorted(set(cty))}
    print(f"model {key[0]} on {'normal ' if key.endswith('N') else 'CROWDED'} val: macro {tot:.4f} {per} params {prm}",
          flush=True)
