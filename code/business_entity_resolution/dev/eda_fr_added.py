"""26 Sep night EDA (see PROGRESS.md §6c). Run from student_resource/."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)
import sys, csv, collections, numpy as np, pandas as pd

import config
from normalize import load_normalized
R = config.ROOT + "/"
s1, pool = load_normalized(R + "dataset/test", "test", n_jobs=config.N_JOBS)
m = pd.read_csv(R + "submissions/day2_C_frfix/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
sn = dict(zip(s1.entity_id, s1.name_norm)); sc = dict(zip(s1.entity_id, s1.country)); sa = dict(zip(s1.entity_id, s1.addr_norm))
pn = dict(zip(pool.entity_id, pool.name_norm)); pa = dict(zip(pool.entity_id, pool.addr_norm))
added = collections.defaultdict(collections.Counter); npairs = collections.Counter(); ns1 = collections.Counter(s1.country)
for s, ms in zip(m.source1_entity_id, m.matched_entity_ids):
    c = sc[s]
    for x in filter(None, ms.split(",")):
        npairs[c] += 1
        a = set(pn[x].split()) - set(sn[s].split())
        for t in a:
            added[c][t] += 1
for c in added:
    print(f"\n== {c}: {npairs[c]} accepted pairs; top added tokens per 100 S1")
    print(", ".join(f"{t} {100*n/ns1[c]:.2f}" for t, n in added[c].most_common(45)))
