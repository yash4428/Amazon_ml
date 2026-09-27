"""Remove US/India pairs with a SHIFTED house number (same length, |diff| 1-30) that our 3-seed blend scores < 0.9,
only when the S1 keeps another match (never empties a row). Base-rate argument: true shifted copies are 3.66 (US) /
4.16 (India) per 100 S1 in train; the base file accepts 4.53 / 4.85, so the excess are fake branches."""
import os, sys, csv, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("out"); ap.add_argument("--max-score", type=float, default=0.9)
a = ap.parse_args()
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
M = pd.read_csv(a.base, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
P = pd.DataFrame([(s, x, ip[s], jp[x]) for s, ms in zip(M.iloc[:, 0], M.iloc[:, 1]) for x in ms.split(",") if x], columns=["s", "x", "i", "j"])
b = pd.read_parquet(os.path.join(_ROOT, "runs", "eda", "blend_scores_cat.parquet"))[["i", "j", "score"]]
P = P.merge(b, on=["i", "j"], how="left"); P["c"] = s1.country.values[P.i.values]
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[P.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[P.j.values]])
def shift(u, v): return u.isdigit() and v.isdigit() and len(u) == len(v) and 0 < abs(int(u) - int(v)) <= 30
sh = np.array([shift(u, v) for u, v in zip(h1, h2)])
cand = sh & P.c.isin(["US", "India"]).values & (P.score.fillna(0.0).values < a.max_score)
keep_other = pd.Series(~cand).groupby(P.i.values).transform("any").values      # row keeps a non-removed match
drop = cand & keep_other
print("shifted low-score pairs:", int(cand.sum()), "| dropped (row keeps another match):", int(drop.sum()), P[drop].c.value_counts().to_dict())
K = P[~drop].groupby("s", sort=False).x.apply(lambda v: ",".join(v))
os.makedirs(a.out, exist_ok=True)
o = M.iloc[:, [0]].copy(); o.columns = ["source1_entity_id"]; o["matched_entity_ids"] = o.source1_entity_id.map(K).fillna("")
o.to_csv(os.path.join(a.out, "matching_results.tsv"), sep="\t", index=False, quoting=csv.QUOTE_NONE)
o.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(os.path.join(a.out, "candidate_pairs.tsv"), sep="\t", index=False, quoting=csv.QUOTE_NONE)
