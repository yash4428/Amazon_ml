"""Stage-1 filter with cheap string features: recall retained vs candidates kept (train part, 2-fold)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, lightgbm as lgb, config
from rapidfuzz import fuzz, process
from normalize import load_normalized, build_translit_dict, apply_translit_dict
from io_utils import read_truth
from blocking import generate_candidates
d = os.path.join(config.ROOT, "splits", "val", "train")
s1, pool = load_normalized(d, "train"); truth = read_truth(d)
tl = build_translit_dict(s1, pool, truth); s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
s1 = s1.sample(30000, random_state=3).reset_index(drop=True)
G = {"combo_c": dict(space="combo_c", k=60, max_df=20000), "combo": dict(space="combo", k=30, max_df=20000)}
c = generate_candidates(s1, pool, G, verbose=False)
sid, pid = s1.entity_id.values, pool.entity_id.values
c["y"] = [x in truth.get(s, ()) for s, x in zip(sid[c.i.values], pid[c.j.values])]
nt = np.array([len(truth.get(s, ())) for s in sid])
A = lambda col, src: src[col].values
for nm, col, src_a, src_b in [("name_tset", "name_core", s1, pool), ("addr_tset", "addr_norm", s1, pool)]:
    c[nm] = process.cpdist(src_a[col].values[c.i.values].tolist(), src_b[col].values[c.j.values].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
c["compact_ratio"] = process.cpdist(s1.name_compact.values[c.i.values].tolist(), pool.name_compact.values[c.j.values].tolist(), scorer=fuzz.ratio, workers=-1)
h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[c.i.values]]); h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[c.j.values]])
c["house"] = np.where((h1 == "") | (h2 == ""), 0, np.where(h1 == h2, 1, -1))
F = ["combo_c_sim", "combo_c_rank", "combo_sim", "combo_rank", "name_tset", "addr_tset", "compact_ratio", "house"]
for g in ("combo_c_sim", "combo_sim", "name_tset", "addr_tset"):
    c[g + "_gap"] = c.groupby("i")[g].transform("max") - c[g]; F.append(g + "_gap")
c["addr_empty"] = (pool.addr_norm.values[c.j.values] == "").astype(int); F.append("addr_empty")
fold = c.i.values % 2; p = np.zeros(len(c))
for k in (0, 1):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=8),
                  lgb.Dataset(c.loc[fold != k, F], c.y[fold != k]), 250)
    p[fold == k] = m.predict(c.loc[fold == k, F])
c["p"] = p; c["r"] = c.groupby("i")["p"].rank(ascending=False, method="first")
tot = nt.sum(); print(f"union K60+30: {len(c)/len(s1):.1f} cand/S1, recall {c.y.sum()/tot:.4f}")
def oracle(mask):
    k = np.bincount(c.i[mask], weights=c.y[mask], minlength=len(s1)); r = k / np.maximum(nt, 1)
    return np.where(nt == 0, 1.0, np.where(k > 0, 1.25 * r / (0.25 + r), 0)).mean()
for K in (6, 8, 10, 12, 15, 20):
    m = (c.r <= K).values; print(f"top-{K:2d}: {m.sum()/len(s1):5.1f} cand/S1 recall {c.y[m].sum()/tot:.4f} oracle {oracle(m):.4f}")
for t in (0.001, 0.003, 0.01):
    m = (c.p >= t).values; print(f"p>={t}: {m.sum()/len(s1):5.1f} cand/S1 recall {c.y[m].sum()/tot:.4f} oracle {oracle(m):.4f}")
m = ((c.combo_c_rank < 25) | (c.combo_rank < 15)).values; print(f"current config (c25+15): {m.sum()/len(s1):.1f} cand/S1 recall {c.y[m].sum()/tot:.4f} oracle {oracle(m):.4f}")
