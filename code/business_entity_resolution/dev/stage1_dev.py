"""Feasibility of a cheap stage-1 pruning model on blocking-only features (train part, labels allowed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, lightgbm as lgb, config
from normalize import load_normalized, build_translit_dict, apply_translit_dict
from io_utils import read_truth
from blocking import generate_candidates
d = os.path.join(config.ROOT, "splits", "val", "train")
s1, pool = load_normalized(d, "train"); truth = read_truth(d)
tl = build_translit_dict(s1, pool, truth); s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
s1 = s1.sample(30000, random_state=1).reset_index(drop=True)
G = {"combo_c": dict(space="combo_c", k=40, max_df=20000), "combo": dict(space="combo", k=40, max_df=20000),
     "name_c4": dict(space="name_c4", k=20, max_df=5000), "addr_tok": dict(space="addr_tok", k=20, max_df=5000)}
c = generate_candidates(s1, pool, G, verbose=False)
sid, pid = s1.entity_id.values, pool.entity_id.values
c["y"] = [x in truth.get(s, ()) for s, x in zip(sid[c.i.values], pid[c.j.values])]
nt = np.array([len(truth.get(s, ())) for s in sid])
F = []
for g in G:
    F += [f"{g}_sim", f"{g}_rank"]
    c[f"{g}_gap"] = c.groupby("i")[f"{g}_sim"].transform("max") - c[f"{g}_sim"]; F.append(f"{g}_gap")
c["n_found"] = sum((c[f"{g}_rank"] < 99).astype(int) for g in G); F.append("n_found")
fold = (c.i.values % 2)
p = np.zeros(len(c))
for k in (0, 1):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=8),
                  lgb.Dataset(c.loc[fold != k, F], c.y[fold != k]), 300)
    p[fold == k] = m.predict(c.loc[fold == k, F])
c["p"] = p
c["r"] = c.groupby("i")["p"].rank(ascending=False, method="first")
tot = nt.sum(); print(f"union: {len(c)/len(s1):.1f} cand/S1, recall {c.y.sum()/tot:.4f}")
def oracle(mask):
    k = np.bincount(c.i[mask], weights=c.y[mask], minlength=len(s1)); r = k / np.maximum(nt, 1)
    f = np.where(nt == 0, 1.0, np.where(k > 0, 1.25 * r / (0.25 + r), 0)); return f.mean()
for K in (4, 5, 6, 8, 10, 12, 15):
    m = c.r <= K; print(f"top-{K:2d} by stage1: {m.sum()/len(s1):5.1f} cand/S1 recall {c.y[m].sum()/tot:.4f} oracle {oracle(m.values):.4f}")
for t in (0.001, 0.003, 0.01, 0.03):
    m = c.p >= t; print(f"p>={t:<5}: {m.sum()/len(s1):5.1f} cand/S1 recall {c.y[m].sum()/tot:.4f} oracle {oracle(m.values):.4f}")
