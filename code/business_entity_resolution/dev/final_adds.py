"""Add back to a (teammate-based) file OUR model's confident pairs it misses: our 3-seed blend score >= 0.99, house
number equal / missing / empty address (never a different number), record not used by the base file for any S1."""
import os, sys, csv, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("ours"); ap.add_argument("--out", default="")
ap.add_argument("--min-score", type=float, default=0.99)
ap.add_argument("--scores", default=os.path.join(_ROOT, "runs", "eda", "blend_scores_cat.parquet"), help="make_blend_scores.py output"); ap.add_argument("--no-empty-rows", action="store_true")
a = ap.parse_args()
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
M = rd(a.base)
H = pd.DataFrame([(ip[s], jp[x]) for s, ms in zip(M.iloc[:, 0], M.iloc[:, 1]) for x in ms.split(",") if x], columns=["i", "j"])
her_rows, her_recs = set(H.i), set(H.j)
her_pairs = set(zip(H.i.values, H.j.values))
O = rd(a.ours)
Op = pd.DataFrame([(ip[s], jp[x]) for s, ms in zip(O.iloc[:, 0], O.iloc[:, 1]) for x in ms.split(",") if x], columns=["i", "j"])
b = pd.read_parquet(a.scores)[["i", "j", "score", "cat"]]
Op = Op.merge(b, on=["i", "j"], how="left"); Op["c"] = s1.country.values[Op.i.values]
newp = np.array([(i, j) not in her_pairs for i, j in zip(Op.i.values, Op.j.values)])
m = (Op.score.values >= a.min_score) & Op.cat.isin(["eq", "nonum", "empty"]).values & ~Op.j.isin(her_recs).values & newp
m &= ~((Op.c.values == "US") & (Op.cat.values == "nonum"))   # exp17 parser reads 5-digit US house numbers as postcodes
add = Op[m].copy(); add["row_empty_her"] = ~add.i.isin(her_rows).values
if a.no_empty_rows:
    add = add[~add.row_empty_her]
print("pairs to add (country x base-row-empty):", add.groupby(["c", "row_empty_her"]).size().to_dict())
print("by category:", add.groupby(["c", "cat"]).size().to_dict())
e = add[add.row_empty_her]
for r in e.sample(min(12, len(e)), random_state=2).itertuples():
    print(f"  {r.c:6s} {r.score:.3f} {s1.business_name.values[r.i][:32]:32s} | {s1.business_address.values[r.i][:24]:24s} <- {pool.business_name.values[r.j][:34]:34s} | {pool.business_address.values[r.j][:24]}")
if a.out:
    ids1, idsp = s1.entity_id.values, pool.entity_id.values
    extra = add.groupby("i").j.apply(lambda v: ",".join(idsp[v])).to_dict()
    o = M.iloc[:, :2].copy(); o.columns = ["source1_entity_id", "matched_entity_ids"]
    o["matched_entity_ids"] = [",".join(filter(None, [m_, extra.get(ip[s], "")])) for s, m_ in zip(o.source1_entity_id, o.matched_entity_ids)]
    os.makedirs(a.out, exist_ok=True)
    o.to_csv(os.path.join(a.out, "matching_results.tsv"), sep="\t", index=False, quoting=csv.QUOTE_NONE)
    o.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(os.path.join(a.out, "candidate_pairs.tsv"), sep="\t", index=False, quoting=csv.QUOTE_NONE)
