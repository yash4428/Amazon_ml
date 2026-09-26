"""26 Sep night EDA (see PROGRESS.md §6c). Run from student_resource/."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)
import sys, json, numpy as np, pandas as pd

from decide import apply_rule
from io_utils import read_sources, read_truth
R = _ROOT + "/"
run = sys.argv[1] if len(sys.argv) > 1 else "runs/exp17_test"
o = pd.read_parquet(R + run + "/report/oof_pairs.parquet")
os1 = pd.read_parquet(R + run + "/report/oof_s1.parquet")
prm = json.load(open(R + run + "/report/oof.json"))["params"]
s1, pool = read_sources(R + "dataset/train", "train")
truth = read_truth(R + "dataset/train")
owner = {x: s for s, m in truth.items() for x in m}
cty = dict(zip(s1["entity_id"], s1["country"]))
paddr = dict(zip(pool["entity_id"], pool["business_address"]))
pname = dict(zip(pool["entity_id"], pool["business_name"]))
pred = apply_rule(o, *prm)
P = pred.groupby("s1")["x"].apply(set).to_dict()
C = o.groupby("s1")["x"].apply(set).to_dict()
rows = []
for s in os1["s1"].values:
    T = truth[s]; Pr = P.get(s, set()); tp = len(T & Pr)
    if not T: f = 1.0 if not Pr else 0.0
    elif tp == 0: f = 0.0
    else:
        p, r = tp / len(Pr), tp / len(T); f = 1.25 * p * r / (0.25 * p + r)
    rows.append((s, cty[s], len(T), len(Pr), tp, f))
d = pd.DataFrame(rows, columns=["s1", "cty", "nt", "np", "tp", "f"])
d["loss"] = 1 - d["f"]
print(f"{run}: macro F0.5 {d.f.mean():.5f}  total loss {d.loss.sum()/len(d):.5f}  n={len(d)}")
def cat(r):
    if r.nt == 0: return "singleton->FP" if r.np else "ok"
    if r.np == 0: return "nonsingle->empty"
    fp, fn = r.np - r.tp, r.nt - r.tp
    if r.tp == 0: return "all-wrong"
    return "FP+FN" if fp and fn else ("FP only" if fp else ("FN only" if fn else "ok"))
d["cat"] = d.apply(cat, axis=1)
g = d.groupby("cat").agg(n=("s1", "size"), loss=("loss", "sum"))
g["loss_share_of_score"] = g["loss"] / len(d)
print(g.sort_values("loss", ascending=False).to_string())
print(d.groupby("cty").f.mean())
# FN pairs: why missed
fnrows = []
for s in d.loc[d.nt > 0, "s1"].values:
    T = truth[s]; Pr = P.get(s, set()); Cc = C.get(s, set())
    for x in T - Pr:
        a = paddr.get(x, ""); why = "blocking" if x not in Cc else "model/threshold"
        fnrows.append((s, x, why, a.strip() == "" or a.strip().upper() in ("NULL", "##", "N/A", "NA")))
fn = pd.DataFrame(fnrows, columns=["s1", "x", "why", "emptyaddr"])
print("\nFN pairs:", len(fn)); print(fn.groupby(["why", "emptyaddr"]).size())
# stolen by o2o?
won = pred.set_index("x")["s1"].to_dict()
fn["stolen"] = [won.get(x) not in (None, s) for s, x in zip(fn.s1, fn.x)]
print(fn.groupby(["why", "emptyaddr", "stolen"]).size())
# FP pairs: what are they
fprows = []
for s in d.loc[d.np > 0, "s1"].values:
    T = truth[s]
    for x in P.get(s, set()) - T:
        ow = owner.get(x)
        kind = "synthetic" if x not in paddr else ("distractor" if ow is None else "other_S1_copy")
        fprows.append((s, x, kind, len(T) == 0, paddr.get(x, "").strip() == ""))
fp = pd.DataFrame(fprows, columns=["s1", "x", "kind", "on_singleton", "emptyaddr"])
print("\nFP pairs:", len(fp)); print(fp.groupby(["kind", "on_singleton", "emptyaddr"]).size())
d.to_parquet(_OUT + "d.parquet")
fn.to_parquet(_OUT + "fn.parquet")
fp.to_parquet(_OUT + "fp.parquet")
