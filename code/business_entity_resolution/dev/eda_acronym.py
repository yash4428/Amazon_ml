"""EDA: acronym records (2-5 capitals) — are they initials of the S1, and does the model accept them? (test, day2_best)"""
import os, sys, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from io_utils import read_source
from decide import apply_rule
t = pd.read_parquet("runs/blend_13_14_15/test_scores.parquet")
pr = apply_rule(t, 0.72, 0.70, 0.0); got = set(zip(pr.i, pr.j))
s1 = read_source("dataset/test", "test", 1); pool = pd.concat([read_source("dataset/test", "test", 2), read_source("dataset/test", "test", 3)], ignore_index=True)
ACR = re.compile(r"^[A-Z]{2,5}$")
isacr = np.array([bool(ACR.match(n.strip())) for n in pool.business_name.values])
c = t[isacr[t.j.values]].copy()
def initials(name, drop_legal):
    t_ = unicodedata.normalize("NFKD", name); t_ = "".join(ch for ch in t_ if not unicodedata.combining(ch))
    toks = re.findall(r"[A-Za-z]+", t_)
    legal = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "llc", "inc", "ltd", "pvt", "private", "limited", "corp", "co", "lp", "pc"}
    small = {"de", "du", "des", "la", "le", "les", "et", "of", "the", "and", "d", "l"}
    toks = [x for x in toks if x.lower() not in small and (not drop_legal or x.lower() not in legal)]
    return "".join(x[0].upper() for x in toks)
names = s1.business_name.values[c.i.values]; acr = pool.business_name.values[c.j.values]
c["init_nolegal"] = [initials(n, True) == a.strip() for n, a in zip(names, acr)]
c["init_legal"] = [initials(n, False) == a.strip() for n, a in zip(names, acr)]
c["init_any"] = c.init_nolegal | c.init_legal
c["pred"] = [(a, b) in got for a, b in zip(c.i, c.j)]
c["cty"] = s1.country.values[c.i.values]
c["same_addr_house"] = [ (lambda x, y: bool(x) and x == y)(re.findall(r"\d+", s1.business_address.values[i])[:1], re.findall(r"\d+", pool.business_address.values[j])[:1]) for i, j in zip(c.i, c.j)]
g = c.groupby(["cty", "init_any", "same_addr_house"]).agg(n=("pred", "size"), accepted=("pred", "mean"), mean_score=("score", "mean"))
print(g.round(3).to_string())
m = c[(c.cty == "France") & c.init_any & c.same_addr_house & ~c.pred]
print(f"\nFrance acronym = S1 initials at same house number but REJECTED: {len(m)}  (France S1: 259452)")
for r in m.sample(min(10, len(m)), random_state=1).itertuples():
    print(f"   {r.score:.3f}  {s1.business_name.values[r.i]} | {s1.business_address.values[r.i]}   <-  {pool.business_name.values[r.j]} | {pool.business_address.values[r.j]}")
