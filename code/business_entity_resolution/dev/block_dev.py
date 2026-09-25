"""Blocking development harness on the TRAIN part of a split (labels allowed there).

Samples N S1 records, runs generators against the full train-part pool and
reports pair recall / avg candidates / oracle F0.5, per generator and cumulative.
Usage: python block_dev.py --split val --n 30000 --k 5,10,20,50
"""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import config
from normalize import load_normalized
from io_utils import read_truth
from blocking import generate_candidates

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--k", default="5,10,20,50"); ap.add_argument("--gens", default="")
ap.add_argument("--maxdf", default=""); ap.add_argument("--mix", default=""); ap.add_argument("--country", default="")
a = ap.parse_args()
d = os.path.join(config.ROOT, "splits", a.split, "train")
s1, pool = load_normalized(d, "train", n_jobs=config.N_JOBS)
truth = read_truth(d)
if a.country:
    s1 = s1[s1.country == a.country]
s1 = s1.sample(min(a.n, len(s1)), random_state=config.SEED).reset_index(drop=True)
kmax = max(int(x) for x in a.k.split(","))
gens = {g: dict(v, k=kmax) for g, v in config.BLOCKING.items() if not a.gens or g in a.gens.split(",")}
if a.maxdf:
    for g in gens: gens[g]["max_df"] = int(a.maxdf)
t = time.time()
cand = generate_candidates(s1, pool, gens)
print(f"blocking {time.time()-t:.0f}s, {len(cand)} pairs")
pid = pool.entity_id.values
cand["s1"] = s1.entity_id.values[cand.i]; cand["x"] = pid[cand.j]
tr = {s: truth.get(s, set()) for s in s1.entity_id}
cand["y"] = [x in tr[s] for s, x in zip(cand.s1, cand.x)]
ntrue = sum(len(v) for v in tr.values())

def report(mask, label):
    c = cand[mask]
    found = c[c.y].groupby("s1").size()
    f = []
    for s, t_ in tr.items():
        if not t_: f.append(1.0); continue
        k = found.get(s, 0)
        if k == 0: f.append(0.0); continue
        r = k / len(t_); f.append(1.25 * r / (0.25 + r))
    print(f"{label:40s} recall {c.y.sum()/ntrue:.4f}  cand/S1 {len(c)/len(tr):6.1f}  oracle {np.mean(f):.4f}")

gl = list(gens)
for k in [int(x) for x in a.k.split(",")]:
    print(f"--- K={k}")
    for g in gl:
        report(cand[f"{g}_rank"] < k, g)
    report(np.logical_or.reduce([cand[f"{g}_rank"] < k for g in gl]), "UNION")

for mix in [m for m in a.mix.split(";") if m]:
    spec = dict(x.split(":") for x in mix.split("+"))
    report(np.logical_or.reduce([cand[f"{g}_rank"] < int(k) for g, k in spec.items()]), mix)
