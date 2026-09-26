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
d = config.ROOT + "/dataset/train"
s1, pool = load_normalized(d, "train", n_jobs=config.N_JOBS); truth = read_truth(d)
owner = {x: s for s, m in truth.items() for x in m}
pool["owner"] = [owner.get(x, "") for x in pool.entity_id]
e = pool[pool.addr_norm == ""].copy()
print("empty-address pool records:", len(e), " owned share:", (e.owner != "").mean())
key = "name_core"
s1g = s1.groupby([key, "country"]).entity_id.apply(list).to_dict()
e["cands"] = [s1g.get((n, c), []) for n, c in zip(e[key], e.country)]
e["k"] = e.cands.str.len()
e["hit"] = [o in c for o, c in zip(e.owner, e.cands)]
print("\nexact name_core match to k S1 (same country):")
print(e.groupby(e.k.clip(0, 5)).agg(n=("entity_id", "size"), owned=("owner", lambda s: (s != "").mean()), owner_among_k=("hit", "mean")).to_string())
# for k==1: owner is the one S1 vs distractor vs other
k1 = e[e.k == 1]
print("\nk=1: owner==that S1:", k1.hit.mean(), " distractor (no owner):", (k1.owner == "").mean(), " owned by another S1:", ((k1.owner != "") & ~k1.hit).mean())
# distractor pool records with the same name_core and a non-empty address (fake branches) for those S1
pn = pool[pool.addr_norm != ""].groupby([key, "country"]).owner.apply(lambda s: int((s == "").sum())).to_dict()
k1 = k1.assign(n_distr_same_name=[pn.get((n, c), 0) for n, c in zip(k1[key], k1.country)])
print(k1.groupby(k1.n_distr_same_name.clip(0, 3)).agg(n=("entity_id", "size"), p_true=("hit", "mean")).to_string())
