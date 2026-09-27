"""Reverse-engineer the teammate's decision boundary: on OUR candidate pairs (3-seed blend score >= 0.1), fit a shallow
decision tree per country predicting her accept/reject from interpretable pair features; print the rules and how
well they reproduce her (accuracy / agreement)."""
import os, sys, csv, re
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from sklearn.tree import DecisionTreeClassifier, export_text
from normalize import load_normalized
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
M = rd(os.path.join(_ROOT, "submissions/teammate_0.984833/matching_results.tsv"))
her = set((ip[s], jp[x]) for s, ms in zip(M.iloc[:, 0], M.iloc[:, 1]) for x in ms.split(",") if x)
t = pd.read_parquet(os.path.join(_ROOT, "runs", "eda", "blend_scores_cat.parquet"))
t = t.sample(n=min(3_000_000, len(t)), random_state=0).reset_index(drop=True)
a, b = t.i.values, t.j.values
t["her"] = [(x, y) in her for x, y in zip(a, b)]
t["name_tset"] = process.cpdist(s1.name_norm.values[a].tolist(), pool.name_norm.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=8)
t["name_ratio"] = process.cpdist(s1.name_compact.values[a].tolist(), pool.name_compact.values[b].tolist(), scorer=fuzz.ratio, workers=8)
t["addr_tset"] = process.cpdist(s1.addr_norm.values[a].tolist(), pool.addr_norm.values[b].tolist(), scorer=fuzz.token_set_ratio, workers=8)
t["native"] = [bool(re.search(r"[ऀ-෿]", x)) for x in pool.business_name.values[b]]
for k, v in {"eq": 0, "nonum": 1, "empty": 2, "shift": 3, "diff": 4}.items():
    t[f"h_{k}"] = (t.cat.values == k).astype(int)
t["n_cand_s1"] = t.groupby("i").i.transform("size")
t["rank_s1"] = t.groupby("i").score.rank(ascending=False, method="first")
t["best_gap"] = t.groupby("i").score.transform("max") - t.score
t["n_s1_for_rec"] = t.groupby("j").j.transform("size")
t["rank_rec"] = t.groupby("j").score.rank(ascending=False, method="first")
F = ["score", "name_tset", "name_ratio", "addr_tset", "native", "h_eq", "h_nonum", "h_empty", "h_shift", "h_diff",
     "n_cand_s1", "rank_s1", "best_gap", "n_s1_for_rec", "rank_rec"]
for c in ["US", "India", "France"]:
    d = t[t.c == c]
    X, y = d[F].astype(float).values, d.her.values
    ours = d.score.values >= 0.74
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=500, random_state=0).fit(X, y)
    p = tree.predict(X)
    print(f"\n===== {c}: {len(d)} pairs, her accept rate {y.mean():.3f} | agreement ours(score>=0.74) {np.mean(ours == y):.4f} | tree {np.mean(p == y):.4f}")
    imp = sorted(zip(tree.feature_importances_, F), reverse=True)[:8]
    print("   importance:", [(f, round(v, 3)) for v, f in imp])
    print(export_text(tree, feature_names=F, max_depth=3, decimals=2))
