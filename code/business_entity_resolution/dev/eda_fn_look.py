"""26 Sep night EDA (see PROGRESS.md §6c). Run from student_resource/."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)
import sys, numpy as np, pandas as pd

from io_utils import read_sources, read_truth
R = _ROOT + "/"; S = _OUT
d = pd.read_parquet(S + "d.parquet").set_index("s1"); fn = pd.read_parquet(S + "fn.parquet")
s1, pool = read_sources(R + "dataset/train", "train")
sn = dict(zip(s1.entity_id, s1.business_name)); sa = dict(zip(s1.entity_id, s1.business_address))
pn = dict(zip(pool.entity_id, pool.business_name)); pa = dict(zip(pool.entity_id, pool.business_address))
def f05(nt, np_, tp):
    if nt == 0: return 1.0 if np_ == 0 else 0.0
    if tp == 0: return 0.0
    p, r = tp / np_, tp / nt; return 1.25 * p * r / (0.25 * p + r)
base = d.f.sum(); N = len(d)
for name, m in [("all blocking misses", fn.why == "blocking"), ("blocking non-empty addr", (fn.why == "blocking") & ~fn.emptyaddr),
                ("blocking empty addr", (fn.why == "blocking") & fn.emptyaddr), ("model empty addr", (fn.why != "blocking") & fn.emptyaddr),
                ("model non-empty addr", (fn.why != "blocking") & ~fn.emptyaddr), ("ALL FN", fn.why != "")]:
    add = fn[m].groupby("s1").size()
    dd = d.loc[add.index]
    newf = [f05(nt, np_ + a, tp + a) for nt, np_, tp, a in zip(dd.nt, dd["np"], dd.tp, add.values)]
    print(f"{name:26s} pairs {int(m.sum()):6d}  gain if all recovered: +{(sum(newf) - dd.f.sum()) / N:.5f}")
# name-sharing count for empty address FN
from collections import Counter
import re, unicodedata
def core(x):
    x = unicodedata.normalize("NFKD", x.lower()); x = "".join(c for c in x if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", x).split())
cnt = Counter(core(n) for n in s1.business_name)
rng = np.random.RandomState(0)
for name, m in [("MODEL-MISS EMPTY ADDR", (fn.why != "blocking") & fn.emptyaddr), ("MODEL-MISS NON-EMPTY", (fn.why != "blocking") & ~fn.emptyaddr),
                ("BLOCKING NON-EMPTY", (fn.why == "blocking") & ~fn.emptyaddr), ("BLOCKING EMPTY", (fn.why == "blocking") & fn.emptyaddr)]:
    sub = fn[m]
    k = np.array([cnt[core(sn[s])] for s in sub.s1])
    same = np.array([core(pn[x]) == core(sn[s]) for s, x in zip(sub.s1, sub.x)])
    print(f"\n=== {name}: n={len(sub)}  S1 name shared by k S1 (k=1: {np.mean(k==1):.2f}, 2: {np.mean(k==2):.2f}, 3+: {np.mean(k>=3):.2f})  pool name == S1 name (norm): {same.mean():.2f}")
    for t in sub.iloc[rng.choice(len(sub), 12, replace=False)].itertuples():
        print(f"  S1 {sn[t.s1][:40]:40s} | {sa[t.s1][:45]:45s}  <-  {pn[t.x][:40]:40s} | {pa[t.x][:45]}")
