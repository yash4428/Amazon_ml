"""France, label-free: added-name-token rates among exp17-accepted pairs by house-number relation.
same number = (almost all) true copies; shifted number = fake branches; none = cannot tell."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)

import sys, json, collections, numpy as np, pandas as pd

import config
from decide import apply_rule
from normalize import load_normalized
from postprocess import street_number
R = config.ROOT + "/"; S = _OUT
s1, pool = load_normalized(R + "dataset/test", "test", n_jobs=config.N_JOBS)
t = pd.read_parquet(R + "runs/exp17_test/test_scores.parquet")
prm = json.load(open(R + "runs/exp17_test/report/oof.json"))["params"]
pr = apply_rule(t, *prm).reset_index(drop=True)
pr = pr[s1.country.values[pr.i.values] == "France"].reset_index(drop=True)
a1 = s1.business_address.values[pr.i.values]; a2 = pool.business_address.values[pr.j.values]
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[pr.i.values]])
h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[pr.j.values]])
sn1 = np.array([street_number(x) for x in a1]); sn2 = np.array([street_number(x) for x in a2])
n1 = np.where(sn1 != "", sn1, h1); n2 = np.where(sn2 != "", sn2, h2)
rel = np.where(pool.addr_norm.values[pr.j.values] == "", "empty_addr",
      np.where((n1 == "") | (n2 == ""), "no_number", np.where(n1 == n2, "same", "differ")))
pr["rel"] = rel
nm1 = s1.name_norm.values[pr.i.values]; nm2 = pool.name_norm.values[pr.j.values]
pr["added"] = [tuple(sorted(set(b.split()) - set(a.split()))) for a, b in zip(nm1, nm2)]
pr.to_parquet(S + "fr_pairs.parquet")
print(pr.rel.value_counts())
cnt = {r: collections.Counter(tk for ad in pr.added[pr.rel == r] for tk in ad) for r in ["same", "differ", "no_number", "empty_addr"]}
N = pr.rel.value_counts()
tok = collections.Counter(); [tok.update(c) for c in cnt.values()]
rows = []
for tk, n in tok.most_common(80):
    rows.append((tk, n, *[100 * cnt[r][tk] / N[r] for r in ["same", "differ", "no_number", "empty_addr"]]))
df = pd.DataFrame(rows, columns=["token", "n", "same%", "differ%", "no_number%", "empty%"])
df["differ/same"] = df["differ%"] / df["same%"].clip(lower=0.01)
print(df.round(2).to_string())
