"""Train: are word swaps on copies MEANINGFUL words (both frequent in S1 names) or OCR garbage? True rate by kind."""
import os, sys, json, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from decide import apply_rule
from normalize import load_normalized
from eda_word_swap import swap_type
run = "runs/exp23_test"
o = pd.read_parquet(os.path.join(_ROOT, run, "report", "oof_pairs.parquet"))
prm = json.load(open(os.path.join(_ROOT, run, "report", "oof.json")))["params"]
acc = apply_rule(o, *prm); o["acc"] = False; o.loc[acc.index, "acc"] = True
o = o[o.x.str[:3] != "SYN"].copy()
for split, cty in [("train", None), ("test", "France")]:
    s1, pool = load_normalized(os.path.join(_ROOT, "dataset", split), split, n_jobs=8)
    freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
    if split == "train":
        d = o
        d["i"] = pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(d.s1).values
        d["j"] = pd.Series(np.arange(len(pool)), index=pool.entity_id).reindex(d.x).values
    else:
        d = pd.read_parquet(os.path.join(_ROOT, "runs", "eda", "fr_swaps.parquet"))
    sw = [swap_type(a, b) for a, b in zip(s1.name_norm.values[d.i.values], pool.name_norm.values[d.j.values])]
    d = d.assign(r=[x[0] if x else "" for x in sw], a=[x[1] if x else "" for x in sw])
    d = d[d.r != ""]
    d["fr_r"] = [freq[x] for x in d.r]; d["fr_a"] = [freq[x] for x in d.a]
    d["kind"] = np.where((d.fr_r >= 200) & (d.fr_a >= 200), "both frequent words", np.where(d.fr_a < 5, "added word rare (typo/garbage)", "other"))
    n1 = s1.entity_id.nunique() if cty is None else int((s1.country == cty).sum())
    if split == "train":
        g = d[d.acc].groupby("kind").agg(per100=("y", lambda s: 100 * len(s) / 800000), true=("y", "mean"))
        print("TRAIN accepted swaps by kind:\n", g.round(3).to_string())
        g2 = d.groupby("kind").agg(per100=("y", lambda s: 100 * len(s) / 800000), true=("y", "mean"))
        print("TRAIN all candidate swaps by kind:\n", g2.round(3).to_string())
        ex = d[d.acc & (d.kind == "both frequent words")]
        print("train frequent-word swaps (accepted) top:", (ex.r + "->" + ex.a).value_counts().head(15).to_dict())
    else:
        print("FRANCE accepted swaps by kind (per 100 French S1):", (d.kind.value_counts() * 100 / n1).round(2).to_dict())
    del s1, pool
