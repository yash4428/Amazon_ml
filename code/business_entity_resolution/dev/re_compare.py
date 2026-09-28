"""How close is a genuine rebuild to the teammate's file? For each run: share of her matched pairs inside the run's
candidate set, and macro F0.5 of the run's final file measured against her file used as a reference (per country).

    python re_compare.py RUN_DIR:FINAL_MATCHING_TSV [RUN_DIR:FINAL_MATCHING_TSV ...]
"""
import collections
import csv
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from normalize import load_normalized  # noqa: E402


def rd(p):
    """Read an id-list TSV as strings."""
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def main():
    """Print candidate coverage of her pairs and macro F0.5 vs her file for every run given."""
    s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
    ip = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    jp = pd.Series(np.arange(len(pool)), index=pool.entity_id)
    cty = s1.country.values

    def pairs(path):
        m = rd(path)
        return set((ip[s], jp[x]) for s, ms in zip(m.iloc[:, 0], m.iloc[:, 1]) for x in ms.split(",") if x)

    ref = pairs(os.path.join(_ROOT, "code", "business_entity_resolution", "inputs", "teammate_matching_results.tsv"))
    ref_n = collections.Counter(i for i, _ in ref)
    for arg in sys.argv[1:]:
        run, final = arg.split(":")
        cand = pd.read_parquet(os.path.join(run, "test_scores.parquet"), columns=["i", "j"])
        cset = set(zip(cand.i.values, cand.j.values))
        P = pairs(final)
        npred = collections.Counter(i for i, _ in P)
        tp = collections.Counter(i for i, j in P if (i, j) in ref)
        out = {}
        for c in ["US", "India", "France"]:
            idx = np.flatnonzero(cty == c)
            f = np.empty(len(idx))
            for k, i in enumerate(idx):
                nt, npd, t = ref_n.get(i, 0), npred.get(i, 0), tp.get(i, 0)
                if nt == 0:
                    f[k] = 1.0 if npd == 0 else 0.0
                elif t == 0:
                    f[k] = 0.0
                else:
                    p, r = t / npd, t / nt
                    f[k] = 1.25 * p * r / (0.25 * p + r)
            cov = np.mean([(i, j) in cset for i, j in ref if cty[i] == c])
            out[c] = (round(f.mean(), 4), round(float(cov), 4))
        allf = sum(out[c][0] * (cty == c).mean() for c in out)
        print(f"{os.path.basename(run):14s} cand/S1 {len(cand) / len(s1):5.1f} | vs-teammate F0.5 ALL {allf:.4f} | "
              + " | ".join(f"{c}: F {v[0]} cover {v[1]}" for c, v in out.items()))


if __name__ == "__main__":
    main()
