"""Word-substitution pairs: the candidate name REPLACES a word of the S1 name with a dissimilar word
("4l Ecole SARL" -> "4l Amicale SARL"). Train (labels): precision of accepted substitution pairs; test: rates per
country. Hypothesis: in France these are co-located different businesses accepted because city + legal form +
exact address match."""
import os, sys, json, csv
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from decide import apply_rule
from normalize import load_normalized
from io_utils import read_truth

NOISE = set("inc incorporated llc llp ltd limited corp corporation co company plc lp pvt private opc huf sarl sas sasu sa eurl "
            "sci snc sca cie ste societe the and et of de du des la le les a d l center centre services service dba aka "
            "formerly fka nee known as doing business trading ta shri sri smt mr ms dr m s fils compagnie associes france "
            "partners group groupe holding holdings international".split())

def swap_type(n1, n2):
    """Return (removed word, added word) if n2 substitutes a dissimilar non-noise word of n1, else None."""
    a, b = set(n1.split()) - NOISE, set(n2.split()) - NOISE
    rem, add = a - b, b - a
    if len(rem) != 1 or len(add) != 1:
        return None
    r, d = next(iter(rem)), next(iter(add))
    if len(r) < 3 or len(d) < 3 or r.isdigit() or d.isdigit():
        return None
    if Levenshtein.normalized_similarity(r, d) >= 0.5 or r in d or d in r:   # typo / truncation, not a swap
        return None
    return r, d

def tag(df, s1, pool):
    n1 = s1.name_norm.values[df.i.values]; n2 = pool.name_norm.values[df.j.values]
    sw = [swap_type(a, b) for a, b in zip(n1, n2)]
    df["swap"] = [x is not None for x in sw]; df["sw"] = [f"{x[0]}->{x[1]}" if x else "" for x in sw]
    a1 = s1.addr_norm.values[df.i.values]; a2 = pool.addr_norm.values[df.j.values]
    df["same_addr"] = [x == y and x != "" for x, y in zip(a1, a2)]
    return df

out = {}
# ---- train OOF (exp23 run, crowd 0.9 pool => drop SYN)
run = sys.argv[1] if len(sys.argv) > 1 else "runs/exp23_test"
o = pd.read_parquet(os.path.join(_ROOT, run, "report", "oof_pairs.parquet"))
prm = json.load(open(os.path.join(_ROOT, run, "report", "oof.json")))["params"]
acc = apply_rule(o, *prm); o["acc"] = False; o.loc[acc.index, "acc"] = True
o = o[o.x.str[:3] != "SYN"].copy()
s1t, pt = load_normalized(os.path.join(_ROOT, "dataset", "train"), "train", n_jobs=8)
o["i"] = pd.Series(np.arange(len(s1t)), index=s1t.entity_id).reindex(o.s1).values
o["j"] = pd.Series(np.arange(len(pt)), index=pt.entity_id).reindex(o.x).values
o = tag(o, s1t, pt)
n_oof = o.s1.nunique()
for nm, m in [("accepted swap", o.acc & o.swap), ("accepted swap same_addr", o.acc & o.swap & o.same_addr),
              ("all cand swap same_addr", o.swap & o.same_addr)]:
    print(f"TRAIN {nm:28s}: per 100 S1 {100 * m.sum() / n_oof:6.2f}   true rate {o.y[m].mean():.3f}")
del s1t, pt, o
# ---- test accepted pairs (submission file)
sub = sys.argv[2] if len(sys.argv) > 2 else "submissions/day3_exp23_rules"
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
m = pd.read_csv(os.path.join(_ROOT, sub, "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
rows = [(ip[s], jp[x]) for s, ms in zip(m.source1_entity_id, m.matched_entity_ids) for x in ms.split(",") if x]
t = tag(pd.DataFrame(rows, columns=["i", "j"]), s1, pool)
t["c"] = s1.country.values[t.i.values]
n = s1.country.value_counts()
for c in ["US", "India", "France"]:
    tc = t[t.c == c]
    print(f"TEST {c:6s} accepted swap per 100 S1 {100 * tc.swap.sum() / n[c]:6.2f} | swap & same address {100 * (tc.swap & tc.same_addr).sum() / n[c]:6.2f}")
fr = t[(t.c == "France") & t.swap]
print("France top swaps:", fr.sw.value_counts().head(25).to_dict())
fr.to_parquet(os.path.join(_ROOT, "runs", "eda", "fr_swaps.parquet"))
for r in fr.sample(15, random_state=0).itertuples():
    print(f"  {s1.business_name.values[r.i][:36]:36s} | {s1.business_address.values[r.i][:34]:34s} <- {pool.business_name.values[r.j][:36]:36s} | {pool.business_address.values[r.j][:34]}")
