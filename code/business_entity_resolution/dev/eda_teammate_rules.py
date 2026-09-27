"""Hypotheses about the teammate's decisions (0.984833 file) on OUR candidate pairs (exp23 scores):
H1 shifted house number (same length, |diff| <= 30): accepted by her depending on whether the S1 has an exact-number copy.
H2 France: acceptance vs our score for same-number pairs (threshold) and for no-number / empty-address pairs."""
import os, sys, csv
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
from decide import apply_rule
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
M = rd(os.path.join(_ROOT, "submissions/teammate_0.984833/matching_results.tsv"))
T = set((ip[s], jp[x]) for s, ms in zip(M.iloc[:, 0], M.iloc[:, 1]) for x in ms.split(",") if x)
t = pd.read_parquet(os.path.join(_ROOT, "runs/exp23_test/test_scores.parquet"))
t = t[t.score >= 0.02].reset_index(drop=True)
t["her"] = [(a, b) in T for a, b in zip(t.i.values, t.j.values)]
t["c"] = s1.country.values[t.i.values]
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[t.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[t.j.values]])
empty = pool.addr_norm.values[t.j.values] == ""
t["house"] = np.where(empty, "empty", np.where((h1 == "") | (h2 == ""), "nonum", np.where(h1 == h2, "eq", "diff")))
def shift(a, b):
    return a.isdigit() and b.isdigit() and len(a) == len(b) and 0 < abs(int(a) - int(b)) <= 30
t["shifted"] = [shift(a, b) for a, b in zip(h1, h2)]
# does the S1 have an exact-number candidate that SHE accepts?
eq_her = t[(t.house == "eq") & t.her].groupby("i").size()
t["s1_has_eq_her"] = t.i.map(eq_her).fillna(0).values > 0
eq_any = t[(t.house == "eq") & (t.score >= 0.5)].groupby("i").size()
t["s1_has_eq_cand"] = t.i.map(eq_any).fillna(0).values > 0
t["band"] = pd.cut(t.score, [0.02, 0.3, 0.5, 0.72, 0.9, 0.99, 1.01], right=False)
pd.set_option("display.width", 200)
for c in ["US", "India", "France"]:
    d = t[(t.c == c) & t.shifted]
    print(f"\n{c} SHIFTED pairs: her acceptance by (S1 has exact-number candidate p>=0.5) x our score band")
    print(d.pivot_table(index="band", columns="s1_has_eq_cand", values="her", aggfunc=["mean", "size"], observed=True).round(3).to_string())
for c in ["France", "US", "India"]:
    d = t[(t.c == c) & ~t.shifted]
    print(f"\n{c} NON-shift pairs: her acceptance by house x band")
    print(d.pivot_table(index="band", columns="house", values="her", aggfunc="mean", observed=True).round(3).to_string())
t.to_parquet(os.path.join(_ROOT, "runs", "eda", "teammate_on_our_cands.parquet"))
