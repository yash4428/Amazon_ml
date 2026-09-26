"""EDA: S1 whose accepted matches include shifted look-alikes — split by whether the S1 has another (non-shifted)
accepted match. Train OOF precision vs rates in train and test (US / India)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized, reparse_numbers
from features import cluster_features
from decide import apply_rule
RUN = "runs/exp17_test"; prm = tuple(json.load(open(f"{RUN}/report/oof.json"))["params"])
for split in ("train", "test"):
    s1, pool = load_normalized(f"dataset/{split}", split)
    s1, pool = reparse_numbers(s1, n_jobs=10), reparse_numbers(pool, n_jobs=10)
    if split == "train":
        c = pd.read_parquet(f"{RUN}/report/oof_pairs.parquet"); c = c[~c.x.str.startswith("SY")]
        sid = {e: k for k, e in enumerate(s1.entity_id)}; pid = {e: k for k, e in enumerate(pool.entity_id)}
        c["i"] = c.s1.map(sid).values; c["j"] = c.x.map(pid).values
        n1 = pd.Series(s1.country.values[list(set(c.i))]).value_counts()
        s_nt = pd.read_parquet(f"{RUN}/report/oof_s1.parquet"); nt = dict(zip(s_nt.s1, s_nt.n_true))
    else:
        c = pd.read_parquet(f"{RUN}/test_scores.parquet"); n1 = s1.country.value_counts()
    pr = apply_rule(c, *prm).reset_index(drop=True)
    cf = cluster_features(pr, s1, pool); shifted = cf.branch_sig.values == 1
    pr["shifted"] = shifted; pr["cty"] = s1.country.values[pr.i.values]
    g = pr.groupby("i").agg(n=("shifted", "size"), ns=("shifted", "sum"))
    only = set(g[(g.ns >= 1) & (g.ns == g.n)].index); mixed = set(g[(g.ns >= 1) & (g.ns < g.n)].index)
    for k in ("US", "India"):
        m = pr.cty.values == k
        for lab, grp in (("S1 whose ONLY matches are shifted", only), ("shifted pairs of S1 that ALSO have a normal match", mixed)):
            mm = m & shifted & pr.i.isin(grp).values
            line = f"{split:5s} {k:6s} {lab:52s}: {100 * mm.sum() / n1[k]:5.2f} pairs/100 S1"
            if split == "train":
                line += f"  precision {pr.y.values[mm].mean():.3f}"
                if lab.startswith("S1 whose"):
                    ids = pr.s1.values[mm]; line += f"  (those S1 are singletons: {np.mean([nt.get(x, 0) == 0 for x in set(ids)]):.3f})"
            print(line)
