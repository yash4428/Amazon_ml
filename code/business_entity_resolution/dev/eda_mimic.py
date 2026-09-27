"""Build decision variants on OUR blended scores with per-(country-agnostic) house-category thresholds, and score every
file against the teammate's 0.984833 file used as a reference (macro F0.5 per country)."""
import os, sys, csv, json, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
from decide import one_to_one
from postprocess import word_swap
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
def load_pairs(path):
    m = rd(path); return set((ip[s], jp[x]) for s, ms in zip(m.iloc[:, 0], m.iloc[:, 1]) for x in ms.split(",") if x)
REF = load_pairs(os.path.join(_ROOT, "submissions/teammate_0.984833/matching_results.tsv"))
ref_n = collections.Counter(i for i, _ in REF)
cty = s1.country.values
def score_vs_ref(P):
    """macro F0.5 per country of pair set P against REF (empty/empty = 1)."""
    npred = collections.Counter(i for i, _ in P); tp = collections.Counter(i for i, j in P if (i, j) in REF)
    out = {}
    for c in ["US", "India", "France"]:
        idx = np.flatnonzero(cty == c); f = np.empty(len(idx))
        for k, i in enumerate(idx):
            nt, npd, t = ref_n.get(i, 0), npred.get(i, 0), tp.get(i, 0)
            if nt == 0: f[k] = 1.0 if npd == 0 else 0.0
            elif t == 0: f[k] = 0.0
            else:
                p, r = t / npd, t / nt; f[k] = 1.25 * p * r / (0.25 * p + r)
        out[c] = f.mean()
    out["ALL"] = sum(out[c] * (cty == c).mean() for c in ["US", "India", "France"])
    return out
for name in ["day3_exp23_rules_swapFR", "day3_blend23x3_swap", "day3_blend23x3_swap_typeswap"]:
    print(f"{name:34s}", {k: round(v, 4) for k, v in score_vs_ref(load_pairs(os.path.join(_ROOT, "submissions", name, "matching_results.tsv"))).items()})
# ---- mimic variants on blended scores
t = None
for r in ["runs/exp23_test", "runs/exp23s4_test", "runs/exp23s5_test"]:
    u = pd.read_parquet(os.path.join(_ROOT, r, "test_scores.parquet")).sort_values(["i", "j"]).reset_index(drop=True)
    t = u if t is None else t.assign(score=t.score.values + u.score.values)
t["score"] = t.score / 3
t = t[t.score >= 0.1].reset_index(drop=True)
t["c"] = cty[t.i.values]
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[t.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[t.j.values]])
empty = pool.addr_norm.values[t.j.values] == ""
def shift(a, b): return a.isdigit() and b.isdigit() and len(a) == len(b) and 0 < abs(int(a) - int(b)) <= 30
sh = np.array([shift(a, b) for a, b in zip(h1, h2)])
t["cat"] = np.where(empty, "empty", np.where((h1 == "") | (h2 == ""), "nonum", np.where(h1 == h2, "eq", np.where(sh, "shift", "diff"))))
sw = np.array([word_swap(a, b) is not None for a, b in zip(s1.name_norm.values[t.i.values], pool.name_norm.values[t.j.values])])
t["swap"] = sw & (t.c.values == "France")
t.to_parquet(os.path.join(_ROOT, "runs", "eda", "blend_scores_cat.parquet"))
def variant(th, swap=True):
    """th: dict (country, cat) -> threshold; accept after one-to-one; France swap pairs removed if swap."""
    thr = np.array([th.get((c, k), th.get(("*", k), 0.74)) for c, k in zip(t.c.values, t.cat.values)])
    c = t[(t.score.values >= thr) & ~(t.swap.values if swap else False)]
    c = one_to_one(c)
    return set(zip(c.i.values, c.j.values))
base = {("*", "eq"): 0.74, ("*", "nonum"): 0.74, ("*", "empty"): 0.74, ("*", "diff"): 0.74, ("*", "shift"): 0.74,
        ("France", "diff"): 1.1, ("France", "shift"): 1.1, ("US", "shift"): 1.1}
V = {"base (~ours)": base,
     "mimic": {**base, ("France", "eq"): 0.4, ("France", "nonum"): 0.72, ("France", "empty"): 0.7,
               ("US", "shift"): 0.99, ("India", "shift"): 0.95, ("US", "diff"): 0.8, ("India", "diff"): 0.8,
               ("US", "eq"): 0.65, ("India", "eq"): 0.65, ("US", "nonum"): 0.7, ("India", "nonum"): 0.7}}
for fr_eq in [0.3, 0.4, 0.5, 0.6]:
    V[f"mimic FReq{fr_eq}"] = {**V["mimic"], ("France", "eq"): fr_eq}
for k, th in V.items():
    print(f"{k:34s}", {kk: round(v, 4) for kk, v in score_vs_ref(variant(th)).items()})
