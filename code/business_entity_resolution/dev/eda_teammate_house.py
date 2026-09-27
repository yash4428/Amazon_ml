"""Teammate file vs ours on house-number-DIFFERENT pairs: acceptance by digit relation, per country; France: which of
our rules removed the pairs she keeps."""
import os, sys, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
from postprocess import word_swap, street_number
m = pd.read_parquet(os.path.join(_ROOT, "runs", "eda", "teammate_diff.parquet"))
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
def rel(a, b):
    if not a.isdigit() or not b.isdigit():
        return "non-numeric"
    if a.lstrip("0") == b.lstrip("0"):
        return "zero-pad"
    if a in b or b in a:
        return "digit drop/add"
    x, y = int(a), int(b)
    if len(a) == len(b):
        d = abs(x - y)
        return "same-len shift<=30" if d <= 30 else ("same-len 1 digit" if sum(p != q for p, q in zip(a, b)) == 1 else "same-len other")
    return "len differs other"
d = m[m.house == "diff"].copy()
h1 = [x.split(" ", 1)[0] for x in s1.house_numbers.values[d.i.values]]; h2 = [x.split(" ", 1)[0] for x in pool.house_numbers.values[d.j.values]]
d["rel"] = [rel(a, b) for a, b in zip(h1, h2)]
d["who"] = np.where(d.t & d.o, "both", np.where(d.t, "teammate only", "ours only"))
for c in ["US", "India", "France"]:
    print(f"\n{c}: house-DIFFERENT accepted pairs by digit relation (counts)")
    print(d[d.c == c].pivot_table(index="rel", columns="who", values="i", aggfunc="size", fill_value=0).to_string())
f = m[(m.c == "France") & m.t & ~m.o & (m.score >= 0.72)].copy()
n1 = s1.name_norm.values[f.i.values]; n2 = pool.name_norm.values[f.j.values]
f["swap"] = [word_swap(a, b) is not None for a, b in zip(n1, n2)]
print(f"\nFrance teammate-only pairs with our exp23 score >= 0.72: {len(f)}; house {f.house.value_counts().to_dict()}; our swap-rule flag {f.swap.mean():.3f}")
ex = f[~f.swap & (f.house == "eq")].sample(10, random_state=0)
for r in ex.itertuples():
    print(f"   {r.score:.2f} {s1.business_name.values[r.i][:34]:34s} | {s1.business_address.values[r.i][:28]:28s} <- {pool.business_name.values[r.j][:34]:34s} | {pool.business_address.values[r.j][:28]}")
