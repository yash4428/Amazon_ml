"""Blind EDA: 'shifted look-alikes' (same street, house number ±1..30, name token-set ≥ 90) — do they carry an extra
name word? Train (with labels) vs test, per country. All candidates of the run, not only accepted ones."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized, reparse_numbers
from features import cluster_features, pair_extras
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
    else:
        c = pd.read_parquet(f"{RUN}/test_scores.parquet"); n1 = s1.country.value_counts()
    c = c[c.score >= 0.01].reset_index(drop=True)
    acc = set(zip(*[apply_rule(c, *prm)[k].values for k in ("i", "j")]))
    cf = cluster_features(c, s1, pool)
    ts = process.cpdist(s1.name_core.values[c.i.values].tolist(), pool.name_core.values[c.j.values].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    look = (cf.branch_sig.values == 1) & (ts >= 90)
    ex = pair_extras(c[look], s1, pool); extra = np.array([len(e) > 0 for e in ex])
    cc = c[look].assign(extra=extra, cty=s1.country.values[c.i.values[look]],
                        acc=[(a, b) in acc for a, b in zip(c.i.values[look], c.j.values[look])])
    for k in sorted(set(cc.cty)):
        m = cc.cty.values == k
        line = f"{split:5s} {k:7s} shifted look-alikes per 100 S1: {100 * m.sum() / n1[k]:5.2f} | with extra word {cc.extra.values[m].mean():.3f}"
        line += f" | accepted {cc.acc.values[m].mean():.3f}"
        if split == "train":
            for lab, mm in (("plain", m & ~cc.extra.values), ("extra", m & cc.extra.values)):
                line += f" | P(true|{lab}) {cc.y.values[mm].mean():.3f}"
        print(line)
