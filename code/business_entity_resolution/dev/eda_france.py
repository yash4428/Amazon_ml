"""EDA: name sharing among S1 and ambiguity of empty-address records, per country (train vs test)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
for d, p in (("dataset/train", "train"), ("dataset/test", "test")):
    s1, pool = load_normalized(d, p)
    k1 = s1.country + "|" + s1.name_core; vc = k1.value_counts()
    s1d = k1.map(vc)
    emp = pool[pool.addr_norm == ""]
    kd = (emp.country + "|" + emp.name_core).map(vc).fillna(0)
    for c in sorted(s1.country.unique()):
        m = s1.country.values == c; me = emp.country.values == c
        dd = kd[me]
        print(f"{p:5s} {c:7s} S1 sharing core name with >=1 other S1: {np.mean(s1d[m] > 1):.3f} | "
              f"mean #S1 per name {s1d[m].mean():.2f} | empty-addr records: name unique among S1 {np.mean(dd == 1):.3f}, "
              f"shared by 2-3 {np.mean((dd >= 2) & (dd <= 3)):.3f}, by 4+ {np.mean(dd >= 4):.3f}, no exact S1 {np.mean(dd == 0):.3f}")
    # same exact address shared by several S1?
    ka = s1.country + "|" + s1.addr_norm; va = ka.value_counts(); s1a = ka.map(va)
    for c in sorted(s1.country.unique()):
        m = s1.country.values == c
        print(f"      {c:7s} S1 sharing exact normalised address with another S1: {np.mean(s1a[m] > 1):.3f}")
