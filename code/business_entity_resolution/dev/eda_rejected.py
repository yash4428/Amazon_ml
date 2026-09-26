"""EDA: 'copy-like but rejected' candidates per S1 — test by country vs train OOF (with labels)."""
import os, sys, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from io_utils import read_source
from normalize import load_normalized
from decide import apply_rule
NUM = re.compile(r"\d+")
def kinds(c, s1, pool):
    n1 = s1.name_core.values[c.i.values]; n2 = pool.name_core.values[c.j.values]
    ts = process.cpdist(n1.tolist(), n2.tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    h1 = [ (NUM.findall(a) or [""])[0].lstrip("0") for a in s1.business_address.values[c.i.values]]
    h2 = [ (NUM.findall(a) or [""])[0].lstrip("0") for a in pool.business_address.values[c.j.values]]
    emp = pool.addr_norm.values[c.j.values] == ""
    same = np.array([a != "" and a == b for a, b in zip(h1, h2)])
    nonum = np.array([b == "" for b in h2]) & ~emp
    return np.select([(ts >= 90) & same, (ts >= 90) & emp, (ts >= 90) & nonum], ["name90+same_house", "name90+empty_addr", "name90+no_number"], "other")
# test
s1, pool = load_normalized("dataset/test", "test")
t = pd.read_parquet("runs/exp17_test/test_scores.parquet"); prm = tuple(json.load(open("runs/exp17_test/report/oof.json"))["params"])
pr = apply_rule(t, *prm); got = set(zip(pr.i, pr.j))
t["pred"] = [(a, b) in got for a, b in zip(t.i, t.j)]
t = t[t.score >= 0.02]  # ignore hopeless pairs (speed)
t["kind"] = kinds(t, s1, pool); t["cty"] = s1.country.values[t.i.values]
n1 = s1.country.value_counts()
print("TEST: per 100 S1 — copy-like candidates accepted / rejected (score >= 0.02)")
for k in ["name90+same_house", "name90+empty_addr", "name90+no_number"]:
    row = []
    for c in ("France", "India", "US"):
        m = (t.kind == k) & (t.cty == c)
        row.append(f"{c}: acc {100 * t.pred[m].sum() / n1[c]:6.1f} rej {100 * (~t.pred[m]).sum() / n1[c]:5.1f}")
    print(f"  {k:20s} " + " | ".join(row))
# train OOF with labels
s1t, poolt = load_normalized("dataset/train", "train")
o = pd.read_parquet("runs/exp17_test/report/oof_pairs.parquet"); o = o[~o.x.str.startswith("SYN")]
sid = {e: k for k, e in enumerate(s1t.entity_id)}; pid = {e: k for k, e in enumerate(poolt.entity_id)}
pr2 = apply_rule(o, *prm); got2 = set(zip(pr2.s1, pr2.x))
o["pred"] = [(a, b) in got2 for a, b in zip(o.s1, o.x)]
o = o[o.score >= 0.02].copy(); o["i"] = o.s1.map(sid).values; o["j"] = o.x.map(pid).values
o["kind"] = kinds(o, s1t, poolt); o["cty"] = s1t.country.values[o.i.values]
nt = s1t[s1t.entity_id.isin(set(o.s1))].country.value_counts()
print("\nTRAIN OOF (exp17): per 100 sampled S1 — rejected copy-like candidates and how often they are TRUE")
for k in ["name90+same_house", "name90+empty_addr", "name90+no_number"]:
    row = []
    for c in ("India", "US"):
        m = (o.kind == k) & (o.cty == c) & ~o.pred
        row.append(f"{c}: rej {100 * m.sum() / nt[c]:5.1f} P(true|rej) {o.y[m].mean():.3f}")
    print(f"  {k:20s} " + " | ".join(row))
