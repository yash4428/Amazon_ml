"""LightGBM pair classifier with grouped CV (by S1) and out-of-fold predictions."""
import time

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

import config


def group_folds(groups, n_folds=config.N_FOLDS):
    """Fold id per row so that all rows of one S1 share a fold (GroupKFold)."""
    fold = np.zeros(len(groups), np.int8)
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(groups, groups, groups)):
        fold[va] = k
    return fold


def train_oof(X, y, fold, params=None, verbose=True):
    """Train one model per fold id in ``fold`` (rows of a fold are held out together).

    Returns (oof probabilities, list of boosters, gain importance averaged over folds).
    """
    params = dict(config.LGB_PARAMS, **(params or {}))
    oof = np.zeros(len(y), np.float32)
    models = []
    imp = np.zeros(X.shape[1])
    ks = np.unique(fold)
    for k in ks:
        tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr, free_raw_data=True)
        m = lgb.train(params, dtr, num_boost_round=config.LGB_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        models.append(m)
        imp += m.feature_importance("gain")
        if verbose:
            print(f"  fold {k}: best_iter {m.best_iteration}, "
                  f"valid logloss {m.best_score['valid_0']['binary_logloss']:.4f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return oof, models, imp / len(ks)


def train_full(X, y, n_rounds, params=None):
    """Single model on all rows with a fixed number of rounds (used for test inference)."""
    params = dict(config.LGB_PARAMS, **(params or {}))
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=int(n_rounds))


def predict(models, X):
    """Average probability of the given models (each at its best iteration if it has one)."""
    return np.mean([m.predict(X, num_iteration=(m.best_iteration or None)) for m in models],
                   axis=0).astype(np.float32)
