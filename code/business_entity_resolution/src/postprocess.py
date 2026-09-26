"""Post-processing rules applied to a run's scored test candidates (test_scores.parquet).

    python postprocess.py --run runs/exp17_test --out submissions/X [--house-rule] [--shift-rule US]

Rules (EDA 26 Sep, see PROGRESS.md §6):
  --house-rule   For every country whose accepted true-copy-like pairs show (almost) no house-number noise —
                 measured WITHOUT labels as the accepted "digit drop/add" rate (3432 -> 432 style) being below
                 --noise-threshold per 100 S1 — drop accepted pairs whose first house number differs from the
                 S1's. In this data that is France (1.05 per 100 S1 vs ~20 in US/India): French true copies keep
                 their house number, fake branches shift it. Public: exp17 0.966-0.968 -> 0.969 (France ≈0.925 -> 0.938).
                 The rule is data-driven, so it also applies to any other unseen country with the same property.
  --shift-rule C Drop accepted pairs whose house number is shifted by 1-30 on the same street for the listed
                 countries (hypothesis: test fake branches that differ only by a shifted number). Submitted as day2_C.
Both rules only REMOVE pairs, so matches stay a subset of candidates. candidate_pairs.tsv = all scored pairs.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from decide import apply_rule  # noqa: E402
from features import cluster_features  # noqa: E402
from io_utils import write_outputs  # noqa: E402
from normalize import load_normalized, reparse_numbers  # noqa: E402

import re  # noqa: E402
import unicodedata  # noqa: E402

# Robust STREET number = the number directly before a street-type word (FR + EN). Used so the house rule does not
# fire on parsing artefacts: postcode / apartment / "CS 71903" / reordered components before the street number
# ("…, 44600 ST NAZAIRE, 34 rue Guy de Maupassant" vs "34 Rue Guy De Maupassant, 44600 …"). EDA 26 Sep: 1,627 of the
# 39,575 French pairs dropped by the first-number rule have EQUAL street numbers (true copies); on kept pairs the
# robust numbers agree 100% of the time.
_STREET = (r"(?:rue|r|avenue|av|ave|bd|boulevard|blvd|boul|allee|all|chemin|ch|chem|cours|cour|crs|route|rte|"
           r"impasse|imp|place|pl|quai|square|sq|esplanade|espl|faubourg|fg|passage|pass|voie|parvis|rond|mail|"
           r"promenade|prom|sentier|villa|cite|hameau|lotissement|lot|residence|res|chaussee|montee|traverse|clos|"
           r"domaine|street|st|road|rd|drive|dr|lane|ln|court|ct|way|highway|hwy|parkway|pkwy|circle|cir|terrace|"
           r"trail|trl|plaza|pike|loop)")
_STREET_NUM = re.compile(r"(?<![\d])(\d+)\s*(?:bis|ter|quater|[a-d])?\s*[,.-]?\s+" + _STREET + r"\b\.?")


def street_number(addr):
    """Number directly preceding a street-type word ('' if none)."""
    a = unicodedata.normalize("NFKD", addr)
    a = "".join(ch for ch in a if not unicodedata.combining(ch)).lower()
    for ch in ("n°", "nº", "#", "(", ")"):
        a = a.replace(ch, " ")
    m = _STREET_NUM.search(a)
    return (m.group(1).lstrip("0") or "0") if m else ""


def main():
    """Load a run's test scores, apply the decision rule and the optional post-processing rules, write outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with test_scores.parquet and report/oof.json")
    ap.add_argument("--params", default="", help="json with 'params' (default: <run>/report/oof.json)")
    ap.add_argument("--test-dir", default=os.path.join(config.ROOT, "dataset", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--house-rule", action="store_true")
    ap.add_argument("--noise-threshold", type=float, default=3.0)
    ap.add_argument("--shift-rule", nargs="*", default=[])
    ap.add_argument("--shift-keep-only", action="store_true",
                    help="shift rule never empties an S1: shifted pairs are dropped only if the S1 keeps a normal match")
    a = ap.parse_args()

    prm = tuple(json.load(open(a.params or os.path.join(a.run, "report", "oof.json")))["params"])
    s1, pool = load_normalized(a.test_dir, "test", n_jobs=config.N_JOBS)
    s1, pool = reparse_numbers(s1, n_jobs=config.N_JOBS), reparse_numbers(pool, n_jobs=config.N_JOBS)
    t = pd.read_parquet(os.path.join(a.run, "test_scores.parquet"))
    pr = apply_rule(t, *prm).reset_index(drop=True)
    cty = s1["country"].values[pr["i"].values]
    cf = cluster_features(pr, s1, pool)
    diff = cf["house_absdiff_log"].values > 0
    drop = np.zeros(len(pr), bool)
    if a.house_rule:
        for c in sorted(set(cty)):
            m = cty == c
            n1 = int((s1["country"].values == c).sum())
            noise = 100 * float((cf["house_digit_substr"].values[m] == 1).sum()) / max(n1, 1)
            apply = noise < a.noise_threshold
            print(f"  {c:8s} accepted digit-drop pairs per 100 S1 = {noise:6.2f} -> house rule "
                  f"{'APPLIED' if apply else 'not applied'}")
            if apply:
                cand = np.flatnonzero(m & diff)
                a1 = s1["business_address"].values[pr["i"].values[cand]]
                a2 = pool["business_address"].values[pr["j"].values[cand]]
                sn1 = np.array([street_number(x) for x in a1]); sn2 = np.array([street_number(x) for x in a2])
                artefact = (sn1 != "") & (sn1 == sn2)          # first numbers differ, street numbers agree
                print(f"  {c:8s} house rule: {len(cand)} first-number mismatches, {int(artefact.sum())} kept "
                      f"(equal street number = parsing artefact)")
                drop[cand[~artefact]] = True
    for c in a.shift_rule:
        sh = (cty == c) & (cf["branch_sig"].values == 1)
        if a.shift_keep_only:
            # EDA 26 Sep: US S1 whose ONLY accepted matches are shifted are 91.5% true in train (0.34 vs 0.59 per
            # 100 S1 on test) — the test excess is entirely in S1 that also have a normal match (3.34 -> 9.20).
            normal = pd.Series(~sh & (cty == c)).groupby(pr["i"].values).transform("any").values
            sh = sh & normal
        drop |= sh
    print(f"  dropped {int(drop.sum())} of {len(pr)} accepted pairs")
    keep = pr[~drop]
    s1_ids = s1["entity_id"].values
    pool_ids = pool["entity_id"].values

    from run_pipeline import group_ids
    write_outputs(a.out, list(s1_ids), group_ids(keep, s1_ids, pool_ids), group_ids(t, s1_ids, pool_ids))
    with open(os.path.join(a.out, "postprocess_info.json"), "w") as f:
        json.dump({"run": a.run, "params": prm, "house_rule": a.house_rule, "shift_rule": a.shift_rule,
                   "shift_keep_only": a.shift_keep_only,
                   "dropped": int(drop.sum())}, f, indent=2)


if __name__ == "__main__":
    main()
