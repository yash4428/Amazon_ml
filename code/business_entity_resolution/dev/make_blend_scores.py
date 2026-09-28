"""Write the 3-seed blend scores of our model runs with house-number categories (input of final_adds.py).

    python make_blend_scores.py runs/exp23_s3 runs/exp23_s4 runs/exp23_s5 --out runs/eda/blend_scores_cat.parquet

Score = equal-weight mean of the runs' test_scores.parquet (identical candidate sets); pairs with score < 0.1 dropped.
cat = house-number relation of the pair: eq / diff / shift (same length, |diff| 1-30) / nonum / empty (pool address).
"""
import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from normalize import load_normalized  # noqa: E402
from postprocess import word_swap  # noqa: E402


def shift(a, b):
    """Same-length numeric house numbers that differ by 1-30."""
    return a.isdigit() and b.isdigit() and len(a) == len(b) and 0 < abs(int(a) - int(b)) <= 30


def main():
    """Blend the runs' scores and tag each pair's house-number category."""
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--test-dir", default=os.path.join(_ROOT, "dataset", "test"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "runs", "eda", "blend_scores_cat.parquet"))
    a = ap.parse_args()
    s1, pool = load_normalized(a.test_dir, "test", n_jobs=8)
    t = None
    for r in a.runs:
        u = pd.read_parquet(os.path.join(r, "test_scores.parquet")).sort_values(["i", "j"]).reset_index(drop=True)
        t = u if t is None else t.assign(score=t.score.values + u.score.values)
    t["score"] = t.score / len(a.runs)
    t = t[t.score >= 0.1].reset_index(drop=True)
    t["c"] = s1.country.values[t.i.values]
    h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[t.i.values]])
    h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[t.j.values]])
    empty = pool.addr_norm.values[t.j.values] == ""
    sh = np.array([shift(x, y) for x, y in zip(h1, h2)])
    t["cat"] = np.where(empty, "empty", np.where((h1 == "") | (h2 == ""), "nonum",
                        np.where(h1 == h2, "eq", np.where(sh, "shift", "diff"))))
    sw = np.array([word_swap(x, y) is not None for x, y in
                   zip(s1.name_norm.values[t.i.values], pool.name_norm.values[t.j.values])])
    t["swap"] = sw & (t.c.values == "France")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t.to_parquet(a.out)
    print(f"wrote {a.out}: {len(t)} pairs")


if __name__ == "__main__":
    main()
