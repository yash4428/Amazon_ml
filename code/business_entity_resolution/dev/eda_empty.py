"""EDA: empty-address candidates — what separates true copies from wrong ones (train OOF, exp13)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized, build_translit_dict, apply_translit_dict
from io_utils import read_truth
from decide import apply_rule
s1, pool = load_normalized("dataset/train", "train"); truth = read_truth("dataset/train")
tl = build_translit_dict(s1, pool, truth); s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
R = "runs/exp13_test/report"; c = pd.read_parquet(f"{R}/oof_pairs.parquet"); prm = json.load(open(f"{R}/oof.json"))["params"]
c = c[~c.x.str.startswith("SYN")]
pr = apply_rule(c, *prm); got = set(zip(pr.s1, pr.x))
pid = {e: k for k, e in enumerate(pool.entity_id)}; sid = {e: k for k, e in enumerate(s1.entity_id)}
c["jj"] = c.x.map(pid).values; c["ii"] = c.s1.map(sid).values
e = c[pool.addr_norm.values[c.jj.values] == ""].copy()
e["pred"] = [(a, b) in got for a, b in zip(e.s1, e.x)]
n1 = s1.name_core.values[e.ii.values]; n2 = pool.name_core.values[e.jj.values]
e["exact_core"] = n1 == n2
e["tset"] = process.cpdist(n1.tolist(), n2.tolist(), scorer=fuzz.token_set_ratio, workers=-1)
e["ratio"] = process.cpdist(n1.tolist(), n2.tolist(), scorer=fuzz.ratio, workers=-1)
e["n_extra"] = [len(set(b.split()) - set(a.split())) for a, b in zip(n1, n2)]
e["n_miss"] = [len(set(a.split()) - set(b.split())) for a, b in zip(n1, n2)]
# how many S1 (same country) have EXACTLY this candidate's core name / this S1's core name
key = s1.country + "|" + s1.name_core; vc = key.value_counts()
e["s1_dup"] = key.values[e.ii.values]; e["s1_dup"] = e.s1_dup.map(vc).values
e["cand_dup"] = (pool.country.values[e.jj.values] + "|" + n2); e["cand_dup"] = e.cand_dup.map(vc).fillna(0).values
# does ANOTHER candidate-claiming S1 have a better name match? (competition among S1 that retrieved this record)
best_other = e.groupby("x").tset.transform("max")
e["n_claim"] = e.groupby("x").s1.transform("size")
print(f"empty-address candidate pairs in OOF: {len(e)}  positives {e.y.mean():.3f}  recall-on-them {e.pred[e.y].mean():.3f}  precision {e.y[e.pred].mean():.3f}")
def show(mask, lab):
    s = e[mask]; print(f"  {lab:48s} n={len(s):7d}  P(true)={s.y.mean():.3f}  predicted={s.pred.mean():.3f}  recall={s.pred[s.y].mean() if s.y.any() else float('nan'):.3f}")
show(e.exact_core, "exact core name")
show(~e.exact_core & (e.tset >= 90), "tset>=90 (not exact)")
show((e.tset >= 70) & (e.tset < 90), "tset 70-90")
show(e.tset < 70, "tset <70")
print("-- exact core name, by #S1 sharing that name (cand_dup):")
for k in (1, 2, 3):
    show(e.exact_core & (e.cand_dup == k), f"exact core & cand_dup={k}")
show(e.exact_core & (e.cand_dup >= 4), "exact core & cand_dup>=4")
print("-- tset>=90, by extra tokens:")
show((e.tset >= 90) & (e.n_extra == 0), "tset>=90 & no extra token")
show((e.tset >= 90) & (e.n_extra >= 1), "tset>=90 & >=1 extra token")
print("-- competition: this S1 is the unique best name match among S1 that retrieved the record")
e["is_best"] = e.tset >= best_other; e["uniq_best"] = e.is_best & (e.groupby("x").tset.transform(lambda s: (s == s.max()).sum()) == 1)
show(e.uniq_best & (e.tset >= 90), "tset>=90 & unique best claimant")
show(e.is_best & ~e.uniq_best & (e.tset >= 90), "tset>=90 & tied best claimant")
show(~e.is_best & (e.tset >= 90), "tset>=90 & another S1 matches better")
print("-- score distribution of TRUE empty-address pairs the model rejected:")
print(e[e.y & ~e.pred].score.describe(percentiles=[.25, .5, .75, .9]).round(3).to_dict())

print("\n==== TIE-BREAKING among S1 that share the candidate's name (tied best claimant, tset>=90) ====")
import unicodedata, re
def raw(t):
    t = unicodedata.normalize("NFKD", t); t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", t).strip()
tie = e[e.is_best & ~e.uniq_best & (e.tset >= 90)].copy()
# keep only records where exactly one claimant is true (answerable ties)
tie = tie[tie.groupby("x").y.transform("sum") == 1]
r1 = [raw(t) for t in s1.business_name.values[tie.ii.values]]; r2 = [raw(t) for t in pool.business_name.values[tie.jj.values]]
tie["raw_ratio"] = process.cpdist(r1, r2, scorer=fuzz.ratio, workers=-1)
tie["raw_tsort"] = process.cpdist(r1, r2, scorer=fuzz.token_sort_ratio, workers=-1)
tie["raw_exact"] = np.array(r1) == np.array(r2)
for col in ("raw_ratio", "raw_tsort"):
    mx = tie.groupby("x")[col].transform("max"); cnt = tie.groupby("x")[col].transform(lambda s: (s == s.max()).sum())
    win = (tie[col] == mx) & (cnt == 1)
    solved = tie[win].groupby("x").y.first()
    print(f"  {col:10s}: ties with a unique winner {tie[win].x.nunique()/tie.x.nunique():.3f}; winner is the true S1 {solved.mean():.3f}")
print(f"  raw exact name equality: P(true)={tie.y[tie.raw_exact].mean():.3f} (n={tie.raw_exact.sum()}), unique raw-exact among claimants: "
      f"{(tie[tie.raw_exact].groupby('x').size()==1).mean():.3f}")
# S1-side: do the S1s sharing a core name differ in raw name / city?
g = tie.groupby("x")
print(f"  answerable tied records: {tie.x.nunique()}  (these are {tie.x.nunique() / e[e.y].x.nunique():.3f} of all true empty-address records)")
