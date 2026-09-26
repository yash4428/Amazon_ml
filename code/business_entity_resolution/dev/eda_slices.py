"""26 Sep night EDA (see PROGRESS.md §6c). Run from student_resource/."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)
import sys, json, numpy as np, pandas as pd

import config
from decide import apply_rule
from normalize import load_normalized
from rapidfuzz import fuzz, process
R = config.ROOT + "/"; S = _OUT
o = pd.read_parquet(R + "runs/exp17_test/report/oof_pairs.parquet")
prm = json.load(open(R + "runs/exp17_test/report/oof.json"))["params"]
pred = apply_rule(o, *prm); o["acc"] = False; o.loc[pred.index, "acc"] = True
rng = np.random.RandomState(0); keep = set(rng.choice(o.s1.unique(), 150000, replace=False))
o = o[o.s1.isin(keep) & (o.x.str[:3] != "SYN")].copy()
s1, pool = load_normalized(R + "dataset/train", "train", n_jobs=4)
si = pd.Series(np.arange(len(s1)), index=s1.entity_id); pj = pd.Series(np.arange(len(pool)), index=pool.entity_id)
a = si.reindex(o.s1).values; b = pj.reindex(o.x).values
o["name"] = process.cpdist(s1.name_core.values[a].tolist(), pool.name_core.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=4)
o["addr"] = process.cpdist(s1.addr_norm.values[a].tolist(), pool.addr_norm.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=4)
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[a]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[b]])
o["house"] = np.where((h1 == "") | (h2 == ""), "miss", np.where(h1 == h2, "eq", "diff"))
o["pempty"] = pool.addr_norm.values[b] == ""
o["cty"] = s1.country.values[a]
o = o[~o.pempty]
o["nb"] = pd.cut(o.name, [-1, 40, 60, 75, 90, 100]); o["ab"] = pd.cut(o.addr, [-1, 40, 60, 75, 90, 100])
g = o.groupby(["nb", "ab", "house"], observed=True).agg(n=("y", "size"), true=("y", "sum"), p_true=("y", "mean"), acc=("acc", "mean"),
      FN=("y", lambda s: int((s & ~o.loc[s.index, "acc"]).sum())), FP=("y", lambda s: int((~s & o.loc[s.index, "acc"]).sum())))
g = g[g.n >= 200]
print("slices sorted by FN (non-empty addresses, 150k S1 sample):")
print(g.sort_values("FN", ascending=False).head(25).round(3).to_string())
print("\nslices sorted by FP:")
print(g.sort_values("FP", ascending=False).head(15).round(3).to_string())
o.to_parquet(S + "slices.parquet")
