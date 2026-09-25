"""Expected-F0.5 subset selection vs tuned thresholds, on a run's OOF predictions."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from decide import apply_rule, one_to_one, macro_f05_fast
R = sys.argv[1]
c = pd.read_parquet(f"{R}/report/oof_pairs.parquet"); s = pd.read_parquet(f"{R}/report/oof_s1.parquet")
nt = pd.Series(s.n_true.values, index=np.arange(len(s))); n = len(s)
prm = json.load(open(f"{R}/report/oof.json"))["params"]
base = macro_f05_fast(apply_rule(c, *prm), nt, np.arange(n)); print("tuned rule", prm, round(base, 5))
# calibration check / isotonic on half, apply to other half (by S1 parity)
half = (c.i.values % 2 == 0)
iso = IsotonicRegression(out_of_bounds="clip").fit(c.score[half], c.y[half])
iso2 = IsotonicRegression(out_of_bounds="clip").fit(c.score[~half], c.y[~half])
c["p"] = np.where(half, iso2.predict(c.score), iso.predict(c.score))
for lo, hi in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
    m = (c.score >= lo) & (c.score < hi); print(f"  raw score [{lo},{hi}): n={m.sum():7d} true-rate {c.y[m].mean():.3f}")
def expf(c, miss=0.03, gamma=1.0):
    o = one_to_one(c, "p").sort_values(["i", "p"], ascending=[True, False])
    p = o.p.values ** gamma; i = o.i.values
    tot = pd.Series(p).groupby(i).transform("sum").values
    cum = pd.Series(p).groupby(i).cumsum().values
    k = pd.Series(np.ones(len(p))).groupby(i).cumsum().values
    exp_true = tot / (1 - miss)                      # expected #true incl. blocking-missed
    ef = 1.25 * cum / (0.25 * exp_true + k)          # E[F] approx for keeping top-k
    # k=0: prob(no true) ≈ prod(1-p) * P(no missed)
    lp0 = pd.Series(np.log1p(-np.clip(p, 0, 0.999999))).groupby(i).transform("sum").values
    e0 = np.exp(lp0) * np.exp(-miss * exp_true)
    best = pd.DataFrame({"i": i, "ef": ef, "k": k}).groupby("i").ef.transform("max").values
    keep = (ef >= best) & (best > e0)
    # keep all up to best k
    bk = pd.DataFrame({"i": i, "k": k, "hit": keep}).assign(kk=lambda d: np.where(d.hit, d.k, 0)).groupby("i").kk.transform("max").values
    return o[(k <= bk)]
for miss in (0.0, 0.03, 0.06):
    for g in (1.0,):
        pr = expf(c, miss, g); print(f"expected-F miss={miss} gamma={g}: {macro_f05_fast(pr, nt, np.arange(n)):.5f}")
