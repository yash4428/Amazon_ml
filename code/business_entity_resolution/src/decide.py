"""Decision step: turn scored candidate pairs into per-S1 match lists.

Order (each piece measurable separately):
  1. one-to-one: each pool record goes to at most one S1 (the highest score);
  2. top-1 threshold t1: the best remaining candidate is kept if score >= t1
     (otherwise the S1 is predicted empty -> singleton gate);
  3. extra candidates kept if score >= t2 and score >= r * best score.
Thresholds are tuned to maximise macro F0.5 on labelled training pairs only.
"""
import numpy as np
import pandas as pd


def one_to_one(c, score="score"):
    """Keep, for every pool record j, only its highest-scoring S1 (ties -> first)."""
    c = c.sort_values(score, ascending=False, kind="stable")
    return c.drop_duplicates("j", keep="first")


def apply_rule(c, t1, t2, r, score="score", o2o=True):
    """Return the subset of candidate rows predicted as matches."""
    if o2o:
        c = one_to_one(c, score)
    s = c[score].values
    best = c.groupby("i")[score].transform("max").values
    is_top = s >= best  # ties at the top all count as top
    keep = np.where(is_top, s >= t1, (s >= t2) & (s >= r * best) & (best >= t1))
    return c[keep]


def macro_f05_fast(pred, n_true, s1_index):
    """Macro F0.5 over the S1 ids in ``s1_index`` (vectorised, same definition as evaluate.f05).

    pred    : DataFrame of predicted pairs with columns i and y (bool: is a true match)
    n_true  : Series i -> number of true matches (0 for singletons), indexed by all S1 i
    """
    npred = pred.groupby("i").size().reindex(s1_index, fill_value=0).values
    tp = pred.groupby("i")["y"].sum().reindex(s1_index, fill_value=0).values
    nt = n_true.reindex(s1_index, fill_value=0).values
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / np.maximum(npred, 1), 0)
        rc = np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(tp > 0, 1.25 * p * rc / (0.25 * p + rc), 0.0)
    f = np.where((nt == 0) & (npred == 0), 1.0, f)
    return float(f.mean())


def tune_rule(c, n_true, s1_index, score="score", o2o=True, step=0.02, passes=2, verbose=True):
    """Coordinate-descent search of (t1, t2, r) maximising macro F0.5 on labelled candidates.

    ``c`` must have columns i, j, y and ``score``. One-to-one is applied once up front;
    each evaluation is a few numpy bincounts, so hundreds of settings are cheap.
    ``s1_index`` must be 0..n-1 positions (all S1 rows, singletons included).
    """
    base = one_to_one(c, score) if o2o else c
    n = len(s1_index)
    i = base["i"].values
    s = base[score].values.astype(np.float64)
    y = base["y"].values.astype(np.float64)
    best_s = np.full(n, -1.0)
    np.maximum.at(best_s, i, s)
    b = best_s[i]
    is_top = s >= b
    nt = n_true.reindex(s1_index, fill_value=0).values.astype(np.float64)

    def ev(t1, t2, r):
        keep = np.where(is_top, s >= t1, (s >= t2) & (s >= r * b) & (b >= t1))
        npred = np.bincount(i[keep], minlength=n).astype(np.float64)
        tp = np.bincount(i[keep], weights=y[keep], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = tp / np.maximum(npred, 1)
            rc = tp / np.maximum(nt, 1)
            f = np.where(tp > 0, 1.25 * p * rc / (0.25 * p + rc), 0.0)
        f = np.where((nt == 0) & (npred == 0), 1.0, f)
        return float(f.mean())

    grid = np.round(np.arange(0.02, 0.99, step), 3)
    r_grid = np.round(np.arange(0.0, 0.96, 0.05), 2)
    t1, t2, r = 0.5, 0.5, 0.0
    best = ev(t1, t2, r)
    for _ in range(passes):
        for name, g in (("t1", grid), ("t2", grid), ("r", r_grid)):
            for v in g:
                cand = dict(t1=t1, t2=t2, r=r)
                cand[name] = float(v)
                f = ev(cand["t1"], cand["t2"], cand["r"])
                if f > best:
                    best, t1, t2, r = f, cand["t1"], cand["t2"], cand["r"]
    if verbose:
        print(f"  tuned rule t1={t1} t2={t2} r={r} -> train macroF0.5 {best:.4f}", flush=True)
    return (t1, t2, r), best
