"""Reverse-engineer the teammate's blocking: characterise her matched pairs that are NOT in our candidate set
(name/address similarity, script, house relation, shared tokens) vs those that are."""
import os, sys, csv, re, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
M = rd(os.path.join(_ROOT, "submissions/teammate_0.984833/matching_results.tsv"))
H = pd.DataFrame([(ip[s], jp[x]) for s, ms in zip(M.iloc[:, 0], M.iloc[:, 1]) for x in ms.split(",") if x], columns=["i", "j"])
C = pd.read_parquet(os.path.join(_ROOT, "runs/exp23_test/test_scores.parquet"), columns=["i", "j"])
C["inc"] = True
H = H.merge(C, on=["i", "j"], how="left"); H["inc"] = H.inc.fillna(False).astype(bool)
H["c"] = s1.country.values[H.i.values]
a, b = H.i.values, H.j.values
H["name_tset"] = process.cpdist(s1.name_norm.values[a].tolist(), pool.name_norm.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=8)
H["addr_tset"] = process.cpdist(s1.addr_norm.values[a].tolist(), pool.addr_norm.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=8)
H["native"] = [bool(re.search(r"[ऀ-෿]", x)) for x in pool.business_name.values[b]]
H["pempty"] = pool.addr_norm.values[b] == ""
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[a]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[b]])
H["house"] = np.where(H.pempty, "empty", np.where((h1 == "") | (h2 == ""), "nonum", np.where(h1 == h2, "eq", "diff")))
pc1 = s1.postcode.values[a] if "postcode" in s1 else np.array([""] * len(a)); pc2 = pool.postcode.values[b] if "postcode" in pool else np.array([""] * len(b))
H["pc_eq"] = (pc1 != "") & (pc1 == pc2)
H.to_parquet(os.path.join(_ROOT, "runs", "eda", "re_her_pairs.parquet"))
for c in ["India", "France", "US"]:
    for tag, m in [("NOT in our candidates", ~H.inc), ("in our candidates", H.inc)]:
        d = H[(H.c == c) & m]
        print(f"\n{c} her pairs {tag}: {len(d)}")
        print(f"   name token-set median {d.name_tset.median():.0f} (<50: {(d.name_tset < 50).mean():.2f}) | addr token-set median {d.addr_tset.median():.0f} (<50: {(d.addr_tset < 50).mean():.2f})")
        print(f"   native-script name {d.native.mean():.3f} | empty addr {d.pempty.mean():.3f} | house {d.house.value_counts(normalize=True).round(2).to_dict()} | postcode equal {d.pc_eq.mean():.3f}")
d = H[(H.c == "India") & ~H.inc].sample(15, random_state=0)
for r in d.itertuples():
    print(f"   {s1.business_name.values[r.i][:30]:30s} | {s1.business_address.values[r.i][:32]:32s} <- {pool.business_name.values[r.j][:30]:30s} | {pool.business_address.values[r.j][:32]}")
