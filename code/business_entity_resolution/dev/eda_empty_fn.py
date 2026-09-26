"""26 Sep night EDA (see PROGRESS.md §6c). Run from student_resource/."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)
import sys, numpy as np, pandas as pd

import config
from normalize import load_normalized
from io_utils import read_truth
S = _OUT
R = config.ROOT + "/"
d = R + "dataset/train"
s1, pool = load_normalized(d, "train", n_jobs=config.N_JOBS); truth = read_truth(d)
o = pd.read_parquet(R + "runs/exp17_test/report/oof_pairs.parquet")
o = o[o.x.str[:3] != "SYN"]
sn = dict(zip(s1.entity_id, s1.name_core)); sc = dict(zip(s1.entity_id, s1.country))
pn = dict(zip(pool.entity_id, pool.name_core)); pa = dict(zip(pool.entity_id, pool.addr_norm))
k = s1.groupby(["name_core", "country"]).size().to_dict()
o["pempty"] = [pa.get(x, "") == "" for x in o.x.values]
e = o[o.pempty].copy()
e["exact"] = [sn[s] == pn[x] for s, x in zip(e.s1, e.x)]
e["k"] = [k.get((sn[s], sc[s]), 0) for s in e.s1]
e["best"] = o.groupby("s1").score.max().reindex(e.s1).values
print("empty-address candidate pairs in OOF:", len(e), " true share", e.y.mean())
g = e[e.exact].groupby(e.k.clip(1, 4))
print("\nEXACT name_core, by #S1 sharing the name:")
print(g.agg(n=("y", "size"), p_true=("y", "mean"), mean_score=("score", "mean"),
            acc_rate=("score", lambda s: (s >= 0.74).mean())).to_string())
k1 = e[e.exact & (e.k == 1)]
print("\nk=1 exact, score bins: true rate per bin")
print(k1.groupby(pd.cut(k1.score, [0, .1, .3, .5, .74, .9, 1.01])).agg(n=("y", "size"), p_true=("y", "mean")).to_string())
print("\nk=1 exact TRUE but score<0.74: examples")
m = k1[k1.y & (k1.score < 0.74)]
print(len(m))
# how many other pool candidates of the same S1 have the same exact name (competing duplicates)?
cnt = e[e.exact].groupby(["s1"]).size()
m = m.assign(n_same_name_empty=cnt.reindex(m.s1).values)
print(m.n_same_name_empty.value_counts().head())
for t in m.sample(15, random_state=1).itertuples():
    print(f"  {t.s1} {sn[t.s1][:35]:35s} <- {t.x} {pn[t.x][:35]:35s} score {t.score:.3f} best {t.best:.3f}")
