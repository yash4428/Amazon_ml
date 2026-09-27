"""Compare the teammate's 0.984833 file with ours: for pairs only in hers / only in ours, were they in our candidate
set, what was our blended score, which of our rules removed them, and what do they look like."""
import os, sys, csv, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
def pairs(path):
    m = rd(path)
    r = [(ip[s], jp[x]) for s, ms in zip(m.iloc[:, 0], m.iloc[:, 1]) for x in ms.split(",") if x]
    return pd.DataFrame(r, columns=["i", "j"])
T = pairs(os.path.join(_ROOT, "submissions/teammate_0.984833/matching_results.tsv")); T["t"] = True
O = pairs(os.path.join(_ROOT, "submissions/day3_exp23_rules_swapFR/matching_results.tsv")); O["o"] = True
sc = pd.concat([pd.read_parquet(os.path.join(_ROOT, r, "test_scores.parquet")).sort_values(["i", "j"]).reset_index(drop=True)
                for r in ["runs/exp23_test"]])
m = T.merge(O, on=["i", "j"], how="outer").fillna(False)
m = m.merge(sc, on=["i", "j"], how="left")
m["c"] = s1.country.values[m.i.values]
m["in_cand"] = m.score.notna()
# owner conflict: is the record assigned to a DIFFERENT S1 in the other file?
tj = T.set_index("j").i; oj = O.set_index("j").i
m["t_owner_other"] = [(j in tj.index) and tj.get(j) != i for i, j in zip(m.i, m.j)]
m["o_owner_other"] = [(j in oj.index) and oj.get(j) != i for i, j in zip(m.i, m.j)]
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[m.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[m.j.values]])
m["house"] = np.where(pool.addr_norm.values[m.j.values] == "", "empty", np.where((h1 == "") | (h2 == ""), "nonum", np.where(h1 == h2, "eq", "diff")))
m.to_parquet(os.path.join(_ROOT, "runs", "eda", "teammate_diff.parquet"))
for c in ["France", "India", "US"]:
    for side, mask in [("ONLY TEAMMATE", m.t & ~m.o), ("ONLY OURS", m.o & ~m.t)]:
        d = m[mask & (m.c == c)]
        print(f"\n== {c} {side}: {len(d)} pairs | in our candidates {d.in_cand.mean():.3f} | house {d.house.value_counts(normalize=True).round(3).to_dict()}")
        print("   our exp23 score quantiles:", np.nanquantile(d.score, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3) if d.in_cand.any() else "-",
              "| record given to another S1 in the OTHER file:", round(float((d.o_owner_other if side.startswith("ONLY T") else d.t_owner_other).mean()), 3))
