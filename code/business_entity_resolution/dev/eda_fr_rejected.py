"""Rejected same-number candidates: France (test) vs US/India (train OOF with labels, and test counts)."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)

import sys, csv, json, numpy as np, pandas as pd

import config
from decide import apply_rule
from normalize import load_normalized
from rapidfuzz import fuzz, process
R = config.ROOT + "/"; S = _OUT
def slice_feats(df, s1, pool):
    a, b = df.i.values, df.j.values
    df["name"] = process.cpdist(s1.name_core.values[a].tolist(), pool.name_core.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=4)
    df["addr"] = process.cpdist(s1.addr_norm.values[a].tolist(), pool.addr_norm.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=4)
    h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[a]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[b]])
    df["house"] = np.where((h1 == "") | (h2 == ""), "miss", np.where(h1 == h2, "eq", "diff"))
    df["pempty"] = pool.addr_norm.values[b] == ""
    df["sb"] = pd.cut(df.score, [0.05, 0.3, 0.5, 0.74], right=False)
    return df
# ---- train OOF (labels)
o = pd.read_parquet(R + "runs/exp17_test/report/oof_pairs.parquet")
prm = json.load(open(R + "runs/exp17_test/report/oof.json"))["params"]
acc = apply_rule(o, *prm).index
o["acc"] = False; o.loc[acc, "acc"] = True
o = o[(~o.acc) & (o.score >= 0.05) & (o.score < 0.74) & (o.x.str[:3] != "SYN")].copy()
s1t, poolt = load_normalized(R + "dataset/train", "train", n_jobs=4)
si = pd.Series(np.arange(len(s1t)), index=s1t.entity_id); pj = pd.Series(np.arange(len(poolt)), index=poolt.entity_id)
o["i"] = si.reindex(o.s1).values; o["j"] = pj.reindex(o.x).values
o = slice_feats(o, s1t, poolt)
q = (o.house == "eq") & (o.addr >= 90) & (o.name >= 80) & ~o.pempty
print("TRAIN OOF rejected, same number, addr>=90, name>=80: per 100 S1 by score bin, true rate")
print(o[q].groupby("sb", observed=True).agg(per100=("y", lambda s: 100 * len(s) / 800000), p_true=("y", "mean")).round(3).to_string())
del s1t, poolt
# ---- test
s1, pool = load_normalized(R + "dataset/test", "test", n_jobs=4)
t = pd.read_parquet(R + "runs/exp17_test/test_scores.parquet")
t = t[(t.score >= 0.05) & (t.score < 0.74)].copy()
m = pd.read_csv(R + "submissions/day2_C_frfix/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
ipos = pd.Series(np.arange(len(s1)), index=s1.entity_id); jpos = pd.Series(np.arange(len(pool)), index=pool.entity_id)
accp = set((ipos[s], jpos[x]) for s, ms in zip(m.source1_entity_id, m.matched_entity_ids) for x in ms.split(",") if x)
t = t[[(a, b) not in accp for a, b in zip(t.i.values, t.j.values)]]
t["c"] = s1.country.values[t.i.values]
t = slice_feats(t, s1, pool)
n = s1.country.value_counts()
q = (t.house == "eq") & (t.addr >= 90) & (t.name >= 80) & ~t.pempty
print("\nTEST rejected, same number, addr>=90, name>=80: per 100 S1 by score bin")
print((t[q].groupby(["c", "sb"], observed=True).size() / n.reindex(t[q].groupby(["c", "sb"], observed=True).size().index.get_level_values(0)).values * 100).round(2).to_string())
t[q & (t.c == "France")].to_parquet(S + "fr_rej.parquet")
ex = t[q & (t.c == "France")].sample(25, random_state=0)
for r in ex.itertuples():
    print(f"  {r.score:.2f} | {s1.business_name.values[r.i][:38]:38s} | {s1.business_address.values[r.i][:40]:40s} <- {pool.business_name.values[r.j][:38]:38s} | {pool.business_address.values[r.j][:40]}")
