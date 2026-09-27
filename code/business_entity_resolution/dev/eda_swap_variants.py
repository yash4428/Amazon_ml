"""Variants of the France word-swap pattern. For each accepted pair, compare the NON-legal-form word sets and classify
the substitution by the frequency (in S1 names) of the removed / added word, with a reduced ignore list that treats
generic business words (groupe, services, holding, international, france, center...) as real words.
Train: candidate true rate and accepted precision per variant; test: accepted per 100 S1 per country."""
import os, sys, json, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd, csv
from rapidfuzz.distance import Levenshtein
from decide import apply_rule
from normalize import load_normalized

LEGAL = set("inc incorporated llc llp ltd limited corp corporation co company plc lp pvt private opc huf sarl sas sasu sa "
            "eurl sci snc sca cie ste societe the and et of de du des la le les a d l dba aka formerly fka nee known as "
            "doing business trading ta shri sri smt mr ms dr m s".split())

def fb(f):
    return "F" if f >= 200 else ("m" if f >= 20 else ("r" if f >= 3 else "g"))   # frequent / mid / rare / garbage

def variant(n1, n2, freq):
    a, b = set(n1.split()) - LEGAL, set(n2.split()) - LEGAL
    rem, add = a - b, b - a
    for r in list(rem):
        for d in list(add):
            if Levenshtein.normalized_similarity(r, d) >= 0.5 or r in d or d in r:
                rem.discard(r); add.discard(d); break
    if not rem and not add:
        return "same"
    if rem and add:
        k = "swap" if len(rem) == 1 and len(add) == 1 else "multiswap"
        return f"{k} {''.join(sorted(fb(freq[w]) for w in rem))}>{''.join(sorted(fb(freq[w]) for w in add))}"[:24]
    if add:
        return "add " + "".join(sorted(fb(freq[w]) for w in add))[:3]
    return "drop " + "".join(sorted(fb(freq[w]) for w in rem))[:3]

run = sys.argv[1] if len(sys.argv) > 1 else "runs/exp23_test"
sub = sys.argv[2] if len(sys.argv) > 2 else "submissions/day3_exp23_rules_swapFR"
o = pd.read_parquet(os.path.join(_ROOT, run, "report", "oof_pairs.parquet"))
prm = json.load(open(os.path.join(_ROOT, run, "report", "oof.json")))["params"]
acc = apply_rule(o, *prm); o["acc"] = False; o.loc[acc.index, "acc"] = True
o = o[(o.x.str[:3] != "SYN") & (o.score >= 0.01)].copy()
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "train"), "train", n_jobs=8)
freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
i = pd.Series(np.arange(len(s1)), index=s1.entity_id).reindex(o.s1).values; j = pd.Series(np.arange(len(pool)), index=pool.entity_id).reindex(o.x).values
o["v"] = [variant(x, y, freq) for x, y in zip(s1.name_norm.values[i], pool.name_norm.values[j])]
tr = o.groupby("v").agg(tr_cand100=("y", lambda s: 100 * len(s) / 800000), tr_cand_true=("y", "mean"))
tr["tr_acc100"] = o[o.acc].groupby("v").size() * 100 / 800000
tr["tr_acc_prec"] = o[o.acc].groupby("v").y.mean()
del s1, pool, o
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
m = pd.read_csv(os.path.join(_ROOT, sub, "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
ip = pd.Series(np.arange(len(s1)), index=s1.entity_id); jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
rows = [(ip[s], jp[x]) for s, ms in zip(m.source1_entity_id, m.matched_entity_ids) for x in ms.split(",") if x]
t = pd.DataFrame(rows, columns=["i", "j"]); t["c"] = s1.country.values[t.i.values]
t["v"] = [variant(x, y, freq) for x, y in zip(s1.name_norm.values[t.i.values], pool.name_norm.values[t.j.values])]
n = s1.country.value_counts()
for c in ["US", "India", "France"]:
    tr[c] = t[t.c == c].groupby("v").size() * 100 / n[c]
tr["FR_vs_train"] = tr["France"] / tr["tr_acc100"].clip(lower=0.01)
pd.set_option("display.width", 250)
print(tr[tr.France.fillna(0) >= 0.2].round(3).sort_values("France", ascending=False).to_string())
t.to_parquet(os.path.join(_ROOT, "runs", "eda", "test_accepted_variant.parquet"))
fr = t[(t.c == "France") & t.v.str.startswith(("swap", "multiswap", "add F", "add m"))]
for v in ["swap F>m", "swap m>F", "swap F>F", "add F", "add m"]:
    ex = fr[fr.v == v]
    if len(ex):
        print(f"--- France {v}: {len(ex)}")
        for r in ex.sample(min(6, len(ex)), random_state=0).itertuples():
            print(f"   {s1.business_name.values[r.i][:38]:38s} | {s1.business_address.values[r.i][:30]:30s} <- {pool.business_name.values[r.j][:38]:38s} | {pool.business_address.values[r.j][:30]}")
