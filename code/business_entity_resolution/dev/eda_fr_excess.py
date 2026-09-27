"""France-specific excess categories in a submission file: (name relation x house relation) accepted per 100 S1, vs the
same category in US / India of the same file. Large France-only excess + low train precision = FP cluster."""
import os, sys, csv, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from normalize import load_normalized
from eda_name_relations import rel
sub = sys.argv[1]
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
M = rd(os.path.join(_ROOT, sub, "matching_results.tsv"))
t = pd.DataFrame([(ip[s], jp[x]) for s, ms in zip(M.iloc[:, 0], M.iloc[:, 1]) for x in ms.split(",") if x], columns=["i", "j"])
t["c"] = s1.country.values[t.i.values]
freq = collections.Counter(w for n in s1.name_norm.values for w in set(n.split()))
t["rel"] = [rel(a, b, freq) for a, b in zip(s1.name_norm.values[t.i.values], pool.name_norm.values[t.j.values])]
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[t.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[t.j.values]])
t["house"] = np.where(pool.addr_norm.values[t.j.values] == "", "empty", np.where((h1 == "") | (h2 == ""), "nonum", np.where(h1 == h2, "eq", "diff")))
t["src"] = pool.src.values[t.j.values]
n = s1.country.value_counts()
g = t.groupby(["rel", "house", "c"]).size().unstack("c").fillna(0)
for c in n.index: g[c] = 100 * g[c] / n[c]
g["FR_excess_vs_max_other"] = g["France"] - g[["US", "India"]].max(axis=1)
pd.set_option("display.width", 200)
print(g.round(3).sort_values("FR_excess_vs_max_other", ascending=False).head(12).to_string())
t.to_parquet(os.path.join(_ROOT, "runs", "eda", "fr_excess_pairs.parquet"))
