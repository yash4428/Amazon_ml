"""Checks before spending the d2-5 slot: (1) French accepted acronym pairs in C — same number? alternative owners
with the same initials at the same address? (2) France house-rule dropped pairs by house-number relation."""
import os, sys, json, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized, reparse_numbers
from features import cluster_features
from decide import apply_rule
from io_utils import read_tsv
s1, pool = load_normalized("dataset/test", "test"); s1, pool = reparse_numbers(s1, n_jobs=10), reparse_numbers(pool, n_jobs=10)
t = pd.read_parquet("runs/exp17_test/test_scores.parquet"); prm = tuple(json.load(open("runs/exp17_test/report/oof.json"))["params"])
pr = apply_rule(t, *prm).reset_index(drop=True); cty = s1.country.values[pr.i.values]
cf = cluster_features(pr, s1, pool)
fr = cty == "France"; n_fr = (s1.country.values == "France").sum()
ACR = re.compile(r"^[A-Z]{2,5}$")
def initials(name):
    t_ = unicodedata.normalize("NFKD", name); t_ = "".join(ch for ch in t_ if not unicodedata.combining(ch))
    toks = re.findall(r"[A-Za-z]+", t_)
    legal = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "cie", "llc", "inc", "ltd", "pvt", "private", "limited", "corp", "co"}
    small = {"de", "du", "des", "la", "le", "les", "et", "of", "the", "and", "d", "l"}
    a = "".join(x[0].upper() for x in toks if x.lower() not in small and x.lower() not in legal)
    b = "".join(x[0].upper() for x in toks if x.lower() not in small)
    return a, b
cand_name = pool.business_name.values[pr.j.values]
isacr = np.array([bool(ACR.match(n.strip())) for n in cand_name])
kept = ~(fr & (cf.house_absdiff_log.values > 0))          # C keeps these in France (house rule)
m = fr & isacr & kept
print(f"(1) France accepted acronym pairs remaining in C: {m.sum()} = {100 * m.sum() / n_fr:.2f} per 100 French S1")
same = cf.house_absdiff_log.values == 0; miss = cf.house_absdiff_log.values < 0
print(f"    house number: same {np.mean(same[m]):.3f}  missing on one side {np.mean(miss[m]):.3f}")
# alternative owners: other S1 at the same normalised address whose initials equal the acronym
addr_key = s1.country + "|" + s1.addr_norm
by_addr = pd.Series(np.arange(len(s1))).groupby(addr_key.values).apply(list).to_dict()
alt = []; match_own = []
for i, j in zip(pr.i.values[m], pr.j.values[m]):
    ac = pool.business_name.values[j].strip()
    own = ac in initials(s1.business_name.values[i]); match_own.append(own)
    others = [k for k in by_addr.get(addr_key.values[i], []) if k != i and ac in initials(s1.business_name.values[k])]
    alt.append(len(others) > 0)
print(f"    acronym equals THIS S1's initials: {np.mean(match_own):.3f}; another S1 at the same address with the same initials: {np.mean(alt):.3f}")
# how many acronym pool records per French S1 vs train-like expectation
emp = pool.addr_norm.values[pr.j.values] == ""
print(f"    of which candidate address empty: {np.mean(emp[m]):.3f}")
# (2) France house-rule dropped pairs by relation
d = fr & (cf.house_absdiff_log.values > 0)
rel = np.select([cf.branch_sig.values[d] == 1, cf.house_digit_substr.values[d] == 1, cf.house_small_offset.values[d] == 1,
                 cf.house_absdiff_log.values[d] > np.log1p(30)], ["shift<=30 same street", "digit drop/add", "shift<=30 other street", "far (>30)"], "other")
print(f"\n(2) France house-rule dropped pairs: {d.sum()}")
print(pd.Series(rel).value_counts().to_string())
ex = pr[d].assign(rel=rel)
for r_ in ("digit drop/add", "far (>30)"):
    print(f"   -- {r_} examples")
    for q in ex[ex.rel == r_].sample(min(6, (ex.rel == r_).sum()), random_state=1).itertuples():
        print(f"      {q.score:.2f} {s1.business_name.values[q.i]} | {s1.business_address.values[q.i]}  <-  {pool.business_name.values[q.j]} | {pool.business_address.values[q.j]}")
