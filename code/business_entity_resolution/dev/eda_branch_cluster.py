"""EDA: does a SHARED shifted house number (another candidate of the same S1 has the same number) separate
fake branches from noisy true copies? Train OOF (labels) vs test (rates)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized, reparse_numbers
from decide import apply_rule
from features import pair_extras, hmis_features, house_mismatch_token_rates
RUN = "runs/exp21_test"; prm = tuple(json.load(open(f"{RUN}/report/oof.json"))["params"])

def annotate(pr, allc, s1, pool, hm):
    """For accepted pairs: shifted-number flags and whether the shifted number is shared by another candidate."""
    h = lambda arr: np.array([x.split(" ", 1)[0] for x in arr])
    f1 = h(s1.house_numbers.values[pr.i.values]); f2 = h(pool.house_numbers.values[pr.j.values])
    both = (f1 != "") & (f2 != ""); diff = both & (f1 != f2)
    # numbers of ALL candidates of each S1 (not only accepted)
    ac = allc[["i", "j"]].copy(); ac["h"] = h(pool.house_numbers.values[ac.j.values])
    cnt = ac[ac.h != ""].groupby(["i", "h"]).size()
    key = pd.MultiIndex.from_arrays([pr.i.values, f2])
    shared = cnt.reindex(key).fillna(0).values - 1          # other candidates with the same number
    ex = pair_extras(pr, s1, pool); mx, _ = hmis_features(ex, hm)
    return pd.DataFrame({"diff": diff, "shared": shared >= 1, "bword": mx > 0.8})

for split in ("train", "test"):
    s1, pool = load_normalized(f"dataset/{split}", split)
    s1, pool = reparse_numbers(s1, n_jobs=10), reparse_numbers(pool, n_jobs=10)
    if split == "train":
        o = pd.read_parquet(f"{RUN}/report/oof_pairs.parquet"); o = o[~o.x.str.startswith("SY")]
        sid = {e: k for k, e in enumerate(s1.entity_id)}; pid = {e: k for k, e in enumerate(pool.entity_id)}
        o["i"] = o.s1.map(sid).values; o["j"] = o.x.map(pid).values
        allc = o; pr = apply_rule(o, *prm).reset_index(drop=True)
        n1 = pd.Series(s1.country.values[list(set(o.i))]).value_counts()
    else:
        allc = pd.read_parquet(f"{RUN}/test_scores.parquet"); pr = apply_rule(allc, *prm).reset_index(drop=True)
        n1 = s1.country.value_counts()
    hm = house_mismatch_token_rates(allc.assign(combo_rank=0), s1, pool)
    A = annotate(pr, allc, s1, pool, hm); cty = s1.country.values[pr.i.values]
    for c in sorted(set(cty)):
        m = (cty == c) & A["diff"].values
        rows = []
        for lab, mm in (("shared", A.shared.values), ("unique", ~A.shared.values)):
            k = m & mm
            prec = f" prec {pr.y.values[k].mean():.3f}" if split == "train" else ""
            rows.append(f"{lab}: {100 * k.sum() / n1[c]:5.2f}/100 S1{prec}")
        print(f"{split:5s} {c:7s} accepted pairs with a DIFFERENT house number -> " + " | ".join(rows))
