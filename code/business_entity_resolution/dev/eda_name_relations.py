"""Name-relation categories (after ignoring legal forms / noise suffixes / articles): train accepted rate + precision,
train candidate true rate, and test accepted rate per country. Categories whose test acceptance in one country is far
above train and whose train true rate is low are FP clusters (the word-swap rule came from this)."""
import os, sys, json, csv, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from decide import apply_rule
from normalize import load_normalized
from postprocess import _SWAP_IGNORE

def rel(n1, n2, freq):
    a, b = set(n1.split()) - _SWAP_IGNORE, set(n2.split()) - _SWAP_IGNORE
    rem, add = a - b, b - a
    def fq(w): return "F" if freq[w] >= 200 else "r"
    # pair up typo-like words (similar) so they do not count as add/remove
    for r in list(rem):
        for d in list(add):
            if Levenshtein.normalized_similarity(r, d) >= 0.5 or r in d or d in r:
                rem.discard(r); add.discard(d); break
    if not rem and not add:
        return "same words (mod typos)"
    if not rem and len(add) == 1:
        return "add 1 " + fq(next(iter(add)))
    if not add and len(rem) == 1:
        return "drop 1 " + fq(next(iter(rem)))
    if len(rem) == 1 and len(add) == 1:
        return "swap " + fq(next(iter(rem))) + fq(next(iter(add)))
    if not a & b:
        return "no shared word"
    return "multi-change"

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/exp23_test"
    sub = sys.argv[2] if len(sys.argv) > 2 else "submissions/day3_exp23_rules"
    o = pd.read_parquet(os.path.join(_ROOT, run, "report", "oof_pairs.parquet"))
    prm = json.load(open(os.path.join(_ROOT, run, "report", "oof.json")))["params"]
    acc = apply_rule(o, *prm); o["acc"] = False; o.loc[acc.index, "acc"] = True
    o = o[(o.x.str[:3] != "SYN") & (o.score >= 0.01)].copy()
    s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "train"), "train", n_jobs=8)
    freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
    i = pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(o.s1).values; j = pd.Series(np.arange(len(pool)), index=pool.entity_id).reindex(o.x).values
    o["rel"] = [rel(x, y, freq) for x, y in zip(s1.name_norm.values[i], pool.name_norm.values[j])]
    tr = o.groupby("rel").agg(cand_per100=("y", lambda s: 100 * len(s) / 800000), cand_true=("y", "mean"))
    tr["acc_per100"] = o[o.acc].groupby("rel").size() * 100 / 800000
    tr["acc_prec"] = o[o.acc].groupby("rel").y.mean()
    del s1, pool, o
    s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
    freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
    m = pd.read_csv(os.path.join(_ROOT, sub, "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
    rows = [(ip[s], jp[x]) for s, ms in zip(m.source1_entity_id, m.matched_entity_ids) for x in ms.split(",") if x]
    t = pd.DataFrame(rows, columns=["i", "j"]); t["c"] = s1.country.values[t.i.values]
    t["rel"] = [rel(x, y, freq) for x, y in zip(s1.name_norm.values[t.i.values], pool.name_norm.values[t.j.values])]
    n = s1.country.value_counts()
    for c in ["US", "India", "France"]:
        tr[f"test_{c}_acc_per100"] = t[t.c == c].groupby("rel").size() * 100 / n[c]
    pd.set_option("display.width", 250)
    print(tr.round(3).sort_values("acc_per100", ascending=False).to_string())
    t.to_parquet(os.path.join(_ROOT, "runs", "eda", "test_accepted_rel.parquet"))
