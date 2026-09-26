"""EDA: can an S1's OTHER confidently-matched copies break ties for empty-address records? (train OOF exp17)"""
import os, sys, json, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from io_utils import read_source
R = "runs/exp17_test/report"; c = pd.read_parquet(f"{R}/oof_pairs.parquet"); c = c[~c.x.str.startswith("SYN")]
s1 = read_source("dataset/train", "train", 1).set_index("entity_id")
pool = pd.concat([read_source("dataset/train", "train", 2), read_source("dataset/train", "train", 3)]).set_index("entity_id")
def raw(t):
    t = unicodedata.normalize("NFKD", t); t = "".join(ch for ch in t if not unicodedata.combining(ch)); return re.sub(r"\s+", " ", t).strip()
emp = set(pool.index[pool.business_address == ""])
e = c[c.x.isin(emp)].copy()
# tied records: >=2 claimants with near-identical score, exactly one true
e["best"] = e.groupby("x").score.transform("max")
tie = e[(e.best - e.score) < 0.15]
g = tie.groupby("x").agg(n=("s1", "size"), ntrue=("y", "sum"))
ok = g[(g.n >= 2) & (g.ntrue == 1)].index
tie = tie[tie.x.isin(ok)].copy()
print("tied empty-address records with exactly one true claimant:", len(ok))
conf = c[(c.score > 0.95) & ~c.x.isin(emp)]       # confidently matched OTHER copies of each S1
sib = conf.groupby("s1").x.apply(list).to_dict()
feat = {"raw_case_exact": [], "sib_raw_ratio": [], "sib_case_ratio": [], "src_share": []}
xname = {x: pool.at[x, "business_name"] for x in ok}; xsrc = {x: pool.at[x, "src"] for x in ok}
for s, x in zip(tie.s1, tie.x):
    sibs = sib.get(s, [])
    nm = xname[x]
    sn = [pool.at[y, "business_name"] for y in sibs]
    feat["sib_raw_ratio"].append(max([fuzz.ratio(raw(nm).lower(), raw(o).lower()) for o in sn], default=-1))
    feat["sib_case_ratio"].append(max([fuzz.ratio(raw(nm), raw(o)) for o in sn], default=-1))        # case-sensitive
    feat["raw_case_exact"].append(max([1 if raw(nm) == raw(o) else 0 for o in sn], default=0))
    feat["src_share"].append(np.mean([pool.at[y, "src"] == xsrc[x] for y in sibs]) if sibs else -1)
for k, v in feat.items():
    tie[k] = v
tie["s1_ratio"] = [fuzz.ratio(raw(s1.at[s, "business_name"]), raw(xname[x])) for s, x in zip(tie.s1, tie.x)]
for col in ["s1_ratio", "sib_raw_ratio", "sib_case_ratio", "raw_case_exact"]:
    mx = tie.groupby("x")[col].transform("max"); cnt = tie.groupby("x")[col].transform(lambda s: (s == s.max()).sum())
    win = (tie[col] == mx) & (cnt == 1)
    w = tie[win]
    print(f"  {col:15s}: unique winner in {w.x.nunique() / len(ok):.3f} of ties; winner correct {w.y.mean():.3f}")
