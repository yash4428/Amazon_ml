"""LightGBM pair classifier with grouped CV (by S1) and out-of-fold predictions."""
import time

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

import config


def train_oof(X, y, groups, params=None, n_folds=config.N_FOLDS, verbose=True):
    """Train ``n_folds`` models with GroupKFold on S1 ids.

    Returns (oof probabilities, list of boosters, gain importance averaged over folds).
    """
    params = dict(config.LGB_PARAMS, **(params or {}))
    oof = np.zeros(len(y), np.float32)
    models = []
    imp = np.zeros(X.shape[1])
    gkf = GroupKFold(n_splits=n_folds)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
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
    return oof, models, imp / n_folds


def predict(models, X):
    """Average probability of the fold models."""
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)
