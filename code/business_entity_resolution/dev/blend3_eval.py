"""Evaluate an equal blend of N runs vs their pairs/singles on the S1 common to all OOF samples."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from decide import tune_rule
runs = sys.argv[1:]
O = [pd.read_parquet(f"{r}/report/oof_pairs.parquet") for r in runs]
S = [set(pd.read_parquet(f"{r}/report/oof_s1.parquet").s1) for r in runs]
common = sorted(set.intersection(*S)); loc = {x: k for k, x in enumerate(common)}
nt_src = pd.read_parquet(f"{runs[0]}/report/oof_s1.parquet").set_index("s1").n_true
nt = pd.Series(nt_src.reindex(common).values, index=np.arange(len(common)))
m = None
for k, o in enumerate(O):
    o = o[o.s1.isin(loc)][["s1", "x", "y", "score"]].rename(columns={"score": f"sc{k}", "y": f"y{k}"})
    m = o if m is None else m.merge(o, on=["s1", "x"], how="outer")
m["y"] = m[[c for c in m.columns if c.startswith("y")]].fillna(False).any(axis=1)
for c in [c for c in m.columns if c.startswith("sc") and c[2:].isdigit()]:
    m[c] = m[c].fillna(0.0)
m["i"] = m.s1.map(loc).values; m["j"] = pd.factorize(m.x)[0]
print("common S1", len(common), "pairs", len(m))
import itertools
idx = list(range(len(runs)))
for r in range(1, len(runs) + 1):
    for comb in itertools.combinations(idx, r):
        c = m.assign(score=m[[f"sc{k}" for k in comb]].mean(axis=1))
        prm, f = tune_rule(c, nt, np.arange(len(common)), verbose=False)
        print(f"{'+'.join(os.path.basename(runs[k]) for k in comb):45s} {f:.5f} {prm}")
