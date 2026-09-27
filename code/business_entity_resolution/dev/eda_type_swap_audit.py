"""Audit the type-swap rule: top (removed -> added) type words in France test accepted pairs; train precision of
accepted type-swaps per country (labels)."""
import os, sys, json, csv, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from decide import apply_rule
from normalize import load_normalized
from postprocess import _TYPE_IGNORE

def types_for(s1, pool, c, minf=20, maxr=1.3):
    ms, mp = s1.country.values == c, pool.country.values == c
    fs = collections.Counter(t for n in s1.name_norm.values[ms] for t in set(n.split()))
    fp = collections.Counter(t for n in pool.name_norm.values[mp] for t in set(n.split()))
    return {w for w, f in fs.items() if f >= minf and (fp[w] / mp.sum()) / ((f + 1) / ms.sum()) < maxr}

def swaps(n1, n2, types):
    a, b = set(n1.split()) - _TYPE_IGNORE, set(n2.split()) - _TYPE_IGNORE
    rem, add = a - b, b - a
    for r in list(rem):
        for d in list(add):
            if Levenshtein.normalized_similarity(r, d) >= 0.5 or r in d or d in r:
                rem.discard(r); add.discard(d); break
    R, A = rem & types, add & types
    return (min(R) + "->" + min(A)) if R and A else ""

# train
run = "runs/exp23_test"
o = pd.read_parquet(os.path.join(_ROOT, run, "report", "oof_pairs.parquet"))
prm = json.load(open(os.path.join(_ROOT, run, "report", "oof.json")))["params"]
acc = apply_rule(o, *prm); acc = acc[acc.x.str[:3] != "SYN"].copy()
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "train"), "train", n_jobs=8)
i = pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(acc.s1).values; j = pd.Series(np.arange(len(pool)), index=pool.entity_id).reindex(acc.x).values
acc["c"] = s1.country.values[i]
nS = pd.Series(s1.country.values[pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(o.s1.unique()).values]).value_counts()
for c in ["US", "India"]:
    T = types_for(s1, pool, c)
    m = acc.c.values == c
    sw = np.array([swaps(x, y, T) for x, y in zip(s1.name_norm.values[i[m]], pool.name_norm.values[j[m]])])
    y = acc.y.values[m]
    print(f"TRAIN {c}: accepted type-swaps per 100 S1 {100 * (sw != '').sum() / nS[c]:.2f}, precision {y[sw != ''].mean():.3f}; top: {collections.Counter(sw[sw != '']).most_common(8)}")
del s1, pool, o, acc
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
T = types_for(s1, pool, "France")
print("France type words include france/compagnie/center/fils/freres:", [w in T for w in ["france", "compagnie", "center", "fils", "freres", "services", "groupe"]])
m = pd.read_csv(os.path.join(_ROOT, "submissions/day3_exp23_rules_swapFR", "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
fr = set(s1.entity_id[s1.country == "France"])
rows = [(ip[s], jp[x]) for s, ms in zip(m.source1_entity_id, m.matched_entity_ids) if s in fr for x in ms.split(",") if x]
t = pd.DataFrame(rows, columns=["i", "j"])
t["sw"] = [swaps(x, y, T) for x, y in zip(s1.name_norm.values[t.i.values], pool.name_norm.values[t.j.values])]
f = t[t.sw != ""]
print(f"France type-swaps (after swap rule): {len(f)} = {100 * len(f) / len(fr):.2f} per 100 S1")
print("top:", collections.Counter(f.sw).most_common(30))
for r in f.sample(12, random_state=1).itertuples():
    print(f"   {s1.business_name.values[r.i][:38]:38s} | {s1.business_address.values[r.i][:28]:28s} <- {pool.business_name.values[r.j][:38]:38s} | {pool.business_address.values[r.j][:28]}")
