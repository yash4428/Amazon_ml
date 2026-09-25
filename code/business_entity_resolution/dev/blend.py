"""Evaluate / build a blend of two runs' scores (OOF for evaluation, test scores for output).

python blend.py runs/exp05_test runs/exp06_test [--write runs/blend_test]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from decide import apply_rule, tune_rule, macro_f05_fast
from io_utils import read_source, write_outputs
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--write")
ap.add_argument("--w", type=float, default=None)
ap.add_argument("--params-from", default="", help="use fixed decision params from this oof.json (no OOF eval)")
a = ap.parse_args()
if a.params_from:
    best = (a.w, float("nan"), tuple(json.load(open(a.params_from))["params"]))
    print("fixed blend", best)
else:
  oa = pd.read_parquet(f"{a.a}/report/oof_pairs.parquet"); ob = pd.read_parquet(f"{a.b}/report/oof_pairs.parquet")
  m = oa.merge(ob[["s1", "x", "score"]], on=["s1", "x"], how="inner", suffixes=("_a", "_b"))
  print("aligned OOF pairs", len(m), "of", len(oa), len(ob))
  s = pd.read_parquet(f"{a.a}/report/oof_s1.parquet"); nt = pd.Series(s.n_true.values, index=np.arange(len(s)))
  best = (None, -1, None)
  for w in ([a.w] if a.w is not None else [0.0, 0.3, 0.5, 0.7, 1.0]):
      c = m.assign(score=w * m.score_a + (1 - w) * m.score_b)
      prm, f = tune_rule(c, nt, np.arange(len(s)), verbose=False)
      print(f"w_a={w:.1f}  OOF macro {f:.5f}  params {prm}")
      if f > best[1]:
          best = (w, f, prm)
  print("best", best)
if a.write:
    w, f, prm = best
    ta = pd.read_parquet(f"{a.a}/test_scores.parquet"); tb = pd.read_parquet(f"{a.b}/test_scores.parquet")
    t = ta.merge(tb[["s1", "x", "score"]], on=["s1", "x"], how="inner", suffixes=("_a", "_b"))
    t["score"] = w * t.score_a + (1 - w) * t.score_b
    pred = apply_rule(t, *prm)
    s1 = read_source("dataset/test", "test", 1)
    cand = t.sort_values(["i", "score"], ascending=[True, False]).groupby("s1")["x"].apply(list).to_dict()
    mt = pred.sort_values(["i", "score"], ascending=[True, False]).groupby("s1")["x"].apply(list).to_dict()
    write_outputs(a.write, list(s1.entity_id), mt, cand)
    json.dump({"blend_w_a": w, "oof": f, "params": prm, "a": a.a, "b": a.b}, open(f"{a.write}/run_info.json", "w"), indent=2)
    print("wrote", a.write)
