"""EDA: how do house-number differences look on TRUE copies vs FAKE branches? (train, exp13 OOF pairs)"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from io_utils import read_source
R = "runs/exp13_test/report"; c = pd.read_parquet(f"{R}/oof_pairs.parquet")
c = c[~c.x.str.startswith("SYN")].sample(1_500_000, random_state=0)
s1 = read_source("dataset/train", "train", 1).set_index("entity_id")
pool = pd.concat([read_source("dataset/train", "train", 2), read_source("dataset/train", "train", 3)]).set_index("entity_id")
a1 = s1.loc[c.s1].business_address.values; a2 = pool.loc[c.x].business_address.values
NUM = re.compile(r"\d+")
def first_num(a):
    m = NUM.search(a); return m.group() if m else ""
def rel(x, y):
    if not x or not y: return "missing"
    xs, ys = x.lstrip("0") or "0", y.lstrip("0") or "0"
    if xs == ys: return "equal" if x == y else "equal_after_zero_strip"
    if xs in ys or ys in xs:
        return "prefix/suffix (digit drop/add)"
    d = abs(int(xs[:9]) - int(ys[:9]))
    if len(xs) == len(ys) and d <= 30: return "same_len_offset<=30"
    if d <= 30: return "diff_len_offset<=30"
    if len(xs) == len(ys) and sum(p != q for p, q in zip(xs, ys)) == 1: return "one_digit_changed"
    return "other"
c["rel"] = [rel(first_num(x), first_num(y)) for x, y in zip(a1, a2)]
t = c.groupby("rel").agg(n=("y", "size"), P_true=("y", "mean"), mean_score=("score", "mean"))
t["share_of_true"] = c[c.y].groupby("rel").size() / c.y.sum()
print(t.sort_values("n", ascending=False).round(3).to_string())
for r in ["prefix/suffix (digit drop/add)", "same_len_offset<=30", "one_digit_changed"]:
    s = c[(c.rel == r)]
    print(f"\n== {r}: examples (y, score, S1 addr | cand addr)")
    for q in s.sample(6, random_state=2).itertuples():
        print(f"   {int(q.y)} {q.score:.2f}  {s1.loc[q.s1].business_address[:45]:45s} | {pool.loc[q.x].business_address[:45]}")
