"""EDA: kinds of ACCEPTED pairs per 100 S1 by country (test) vs train precision of the same kind (OOF)."""
import os, sys, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized
from decide import apply_rule
from features import cluster_features
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/exp17_test"
NUM = re.compile(r"\d+")
def kinds(c, s1, pool):
    cf = cluster_features(c, s1, pool)
    n1 = s1.name_core.values[c.i.values]; n2 = pool.name_core.values[c.j.values]
    ts = process.cpdist(n1.tolist(), n2.tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    emp = pool.addr_norm.values[c.j.values] == ""
    diff = cf.house_absdiff_log.values > 0; same = cf.house_absdiff_log.values == 0
    street = cf.addr_nonum_tset.values >= 90
    return pd.DataFrame({
        "house_diff_same_street": diff & street,
        "branch_sig": cf.branch_sig.values == 1,
        "digit_drop": cf.house_digit_substr.values == 1,
        "name<70_same_house": (ts < 70) & same,
        "name<70_house_diff": (ts < 70) & diff,
        "empty_addr": emp,
    }, index=c.index)
prm = tuple(json.load(open(f"{RUN}/report/oof.json"))["params"])
s1, pool = load_normalized("dataset/test", "test")
t = pd.read_parquet(f"{RUN}/test_scores.parquet"); pr = apply_rule(t, *prm).reset_index(drop=True)
K = kinds(pr, s1, pool); cty = s1.country.values[pr.i.values]; n1 = s1.country.value_counts()
s1t, poolt = load_normalized("dataset/train", "train")
o = pd.read_parquet(f"{RUN}/report/oof_pairs.parquet"); o = o[~o.x.str.startswith("SYN")]
sid = {e: k for k, e in enumerate(s1t.entity_id)}; pid = {e: k for k, e in enumerate(poolt.entity_id)}
po = apply_rule(o, *prm).reset_index(drop=True); po["i"] = po.s1.map(sid).values; po["j"] = po.x.map(pid).values
KO = kinds(po, s1t, poolt); ctyo = s1t.country.values[po.i.values]; n1o = s1t[s1t.entity_id.isin(set(o.s1))].country.value_counts()
print(f"accepted pairs per 100 S1 (test) | train: per 100 S1 and PRECISION of that kind  [{RUN}]")
for k in K.columns:
    row = " | ".join(f"{c} {100 * K[k].values[cty == c].sum() / n1[c]:6.2f}" for c in ("France", "India", "US"))
    rowo = " | ".join(f"{c} {100 * KO[k].values[ctyo == c].sum() / n1o[c]:5.2f} prec {po.y.values[(ctyo == c) & KO[k].values].mean():.3f}" for c in ("India", "US"))
    print(f"  {k:24s} TEST {row}   || TRAIN {rowo}")
