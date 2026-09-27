"""Accepted pairs by (name relation x house-number relation): train (precision) vs test per country, per 100 S1."""
import os, sys, json, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from decide import apply_rule
from normalize import load_normalized
from eda_name_relations import rel  # noqa (module has no top-level side effects beyond defs? guarded below)

def house(s1, pool, i, j):
    h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[i]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[j]])
    e = pool.addr_norm.values[j] == ""
    return np.where(e, "empty", np.where((h1 == "") | (h2 == ""), "nonum", np.where(h1 == h2, "eq", "diff")))

run = sys.argv[1] if len(sys.argv) > 1 else "runs/exp23_test"
o = pd.read_parquet(os.path.join(_ROOT, run, "report", "oof_pairs.parquet"))
prm = json.load(open(os.path.join(_ROOT, run, "report", "oof.json")))["params"]
acc = apply_rule(o, *prm)
acc = acc[acc.x.str[:3] != "SYN"].copy()
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "train"), "train", n_jobs=8)
freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
i = pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(acc.s1).values; j = pd.Series(np.arange(len(pool)), index=pool.entity_id).reindex(acc.x).values
acc["rel"] = [rel(x, y, freq) for x, y in zip(s1.name_norm.values[i], pool.name_norm.values[j])]
acc["house"] = house(s1, pool, i, j)
acc["c"] = s1.country.values[i]
ns_tr = pd.Series(s1.country.values[pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(o.s1.unique()).values]).value_counts()
tr = pd.DataFrame(index=acc.groupby(["rel", "house"]).size().index)
for c in ["US", "India"]:
    g = acc[acc.c == c].groupby(["rel", "house"])
    tr[f"tr_{c}"] = g.size() * 100 / ns_tr[c]
    tr[f"tr_{c}_prec"] = g.y.mean()
del s1, pool
t = pd.read_parquet(os.path.join(_ROOT, "runs", "eda", "test_accepted_rel.parquet"))
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
t["house"] = house(s1, pool, t.i.values, t.j.values)
n = s1.country.value_counts()
for c in ["US", "India", "France"]:
    tr[c] = t[t.c == c].groupby(["rel", "house"]).size() * 100 / n[c]
for c in ["US", "India"]:
    tr[c + "_excess"] = tr[c] - tr[f"tr_{c}"]
pd.set_option("display.width", 250)
print(tr.round(3).sort_values("US_excess", ascending=False).head(14).to_string())
print(tr.round(3).sort_values("India_excess", ascending=False).head(8).to_string())
