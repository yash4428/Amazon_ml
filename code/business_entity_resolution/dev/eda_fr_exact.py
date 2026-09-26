"""France: rejected exact-address candidates (same house number, address tset>=95) with a partly different name
(name 50-90), acronym, or domain. Is this S1 the strongest claimant of the record? Examples."""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized, reparse_numbers
from decide import apply_rule
RUN = "runs/exp17_test"; prm = tuple(json.load(open(f"{RUN}/report/oof.json"))["params"])
s1, pool = load_normalized("dataset/test", "test"); s1, pool = reparse_numbers(s1, n_jobs=10), reparse_numbers(pool, n_jobs=10)
t = pd.read_parquet(f"{RUN}/test_scores.parquet").reset_index(drop=True)
pr = apply_rule(t, *prm); acc = set(zip(pr.i, pr.j)); accj = set(pr.j)
fr = s1.country.values[t.i.values] == "France"; c = t[fr].reset_index(drop=True)
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[c.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[c.j.values]])
at = process.cpdist(s1.addr_norm.values[c.i.values].tolist(), pool.addr_norm.values[c.j.values].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
nt = process.cpdist(s1.name_core.values[c.i.values].tolist(), pool.name_core.values[c.j.values].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
raw2 = pool.business_name.values[c.j.values]
acr = np.array([bool(re.match(r"^[A-Z]{2,5}$", n.strip())) for n in raw2]); dom = np.array([bool(re.search(r"\.com|^@|^#|com$", n, re.I)) for n in raw2])
exact = (h1 != "") & (h1 == h2) & (at >= 95)
c["kind"] = np.select([acr, dom, (nt >= 50) & (nt < 90)], ["acronym", "domain", "name50-90"], "other")
c["acc"] = [(a, b) in acc for a, b in zip(c.i, c.j)]
best = t.groupby("j").score.max()                                    # strongest claim on the record by ANY S1
c["top_claim"] = c.score.values >= best.reindex(c.j.values).values - 1e-9
c["taken_elsewhere"] = c.j.isin(accj).values & ~c.acc.values          # record accepted by another S1
s1_has = pr.groupby("i").size()
c["s1_empty"] = ~c.i.isin(s1_has.index)
n1 = (s1.country.values == "France").sum()
for k in ("name50-90", "acronym", "domain"):
    m = exact & (c.kind.values == k) & ~c.acc.values
    print(f"France rejected exact-address {k:9s}: {100 * m.sum() / n1:5.2f}/100 S1 | this S1 is the top claimant {c.top_claim.values[m].mean():.3f} "
          f"| record accepted by ANOTHER S1 {c.taken_elsewhere.values[m].mean():.3f} | S1 currently EMPTY {c.s1_empty.values[m].mean():.3f} "
          f"| score median {np.median(c.score.values[m]):.2f}")
m = exact & (c.kind.values == "name50-90") & ~c.acc.values & c.top_claim.values & ~c.taken_elsewhere.values
print(f"\nexamples (top claimant, not taken elsewhere): {m.sum()} pairs")
for r in c[m].sample(14, random_state=2).itertuples():
    print(f"   {r.score:.2f}  S1: {s1.business_name.values[r.i]} | {s1.business_address.values[r.i]}\n          C : {pool.business_name.values[r.j]} | {pool.business_address.values[r.j]}")
