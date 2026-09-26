"""Blind EDA 2: exact duplicate pool records, singleton predictability from the S1 record, train/test overlap."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from io_utils import read_source, read_truth
from decide import apply_rule
s1 = read_source("dataset/train", "train", 1); truth = read_truth("dataset/train")
pool = pd.concat([read_source("dataset/train", "train", 2), read_source("dataset/train", "train", 3)], ignore_index=True)
owner = {x: s for s, v in truth.items() for x in v}
pool["owner"] = pool.entity_id.map(owner)
# 1) exact duplicates (identical raw name + address) in the pool
key = pool.business_name.str.strip() + "||" + pool.business_address.str.strip()
g = pool.assign(k=key).groupby("k")
sizes = g.size(); dup_keys = sizes[sizes > 1].index
d = pool.assign(k=key)[key.isin(dup_keys)]
same_owner = d.groupby("k").owner.agg(lambda s: s.nunique(dropna=False) == 1)
print(f"1) pool records in an exact-duplicate group: {len(d)} ({len(d)/len(pool):.3f}); groups {len(dup_keys)}; "
      f"groups whose members all share the same owner (or all none): {same_owner.mean():.4f}")
# how often does our OOF prediction accept one twin but not the other?
R = "runs/exp17_test/report"; o = pd.read_parquet(f"{R}/oof_pairs.parquet"); prm = json.load(open(f"{R}/oof.json"))["params"]
o = o[~o.x.str.startswith("SY")]; pr = apply_rule(o, *prm); acc = set(zip(pr.s1, pr.x))
kx = dict(zip(pool.entity_id, key)); d_ids = set(d.entity_id)
cand = o[o.x.isin(d_ids)].copy(); cand["k"] = cand.x.map(kx); cand["acc"] = [(a, b) in acc for a, b in zip(cand.s1, cand.x)]
grp = cand.groupby(["s1", "k"]).agg(n=("acc", "size"), a=("acc", "sum"), y=("y", "sum"))
split = grp[(grp.n >= 2) & (grp.a >= 1) & (grp.a < grp.n)]
print(f"   OOF: (S1, duplicate-group) cases where we accepted SOME but not ALL identical twins: {len(split)} "
      f"(twins truly belong: {(split.y == split.n).mean():.3f})")
# 2) singleton predictability from the S1 record alone
def rec_feats(df):
    n = df.business_name; a = df.business_address; f = pd.DataFrame(index=df.index)
    f["n_len"] = n.str.len(); f["a_len"] = a.str.len(); f["n_words"] = n.str.split().str.len(); f["a_commas"] = a.str.count(",")
    f["a_unit"] = a.str.contains(r"Unit|Apt|Suite", case=False); f["n_legal"] = n.str.contains(r"(?i)\b(llc|inc|ltd|pvt|private|limited|corp|co)\b")
    f["a_digit_first"] = a.str.match(r"^\s*\d"); f["us"] = df.country == "US"; f["n_dot"] = n.str.contains(r"\.")
    f["n_comma"] = n.str.contains(","); f["n_amp"] = n.str.contains("&")
    return f.astype(float)
y = np.array([0 if truth.get(s) else 1 for s in s1.entity_id])     # 1 = singleton
X = rec_feats(s1); fold = np.random.RandomState(0).randint(0, 2, len(y)); p = np.zeros(len(y))
for k in (0, 1):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1, num_threads=8), lgb.Dataset(X[fold != k], y[fold != k]), 200)
    p[fold == k] = m.predict(X[fold == k])
print(f"2) singleton vs non-singleton from the S1 record alone: AUC {roc_auc_score(y, p):.4f} (singleton rate {y.mean():.3f})")
# name duplication as a singleton signal
kn = s1.business_name.str.lower().str.strip(); vc = kn.value_counts(); dup = kn.map(vc).values
for lab, m in (("unique raw name", dup == 1), ("name shared", dup > 1)):
    print(f"   S1 with {lab}: singleton rate {y[m].mean():.3f} (n={m.sum()})")
# 3) train/test overlap of S1 records
t1 = read_source("dataset/test", "test", 1)
ktr = set(s1.business_name.str.strip() + "||" + s1.business_address.str.strip()); kte = t1.business_name.str.strip() + "||" + t1.business_address.str.strip()
print(f"3) test S1 identical (name+address) to a train S1: {kte.isin(ktr).mean():.4f};  same name only: {t1.business_name.isin(set(s1.business_name)).mean():.3f}")
tp = pd.concat([read_source("dataset/test", "test", 2), read_source("dataset/test", "test", 3)], ignore_index=True)
kp = tp.business_name.str.strip() + "||" + tp.business_address.str.strip()
print(f"   test pool records identical to a train pool record: {kp.isin(set(key)).mean():.4f}")
