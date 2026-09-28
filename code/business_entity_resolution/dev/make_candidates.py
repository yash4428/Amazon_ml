"""Candidate file of the best (hybrid) submission: for every S1, the union of the final matches and the teammate
model's original matches (the pairs the rule stage evaluated), so matches are a subset of candidates.

    python make_candidates.py <final matching_results.tsv> <teammate matching_results.tsv> <out candidate_pairs.tsv>
"""
import csv
import sys

import pandas as pd


def rd(p):
    """Read an id-list TSV as strings (tab separated, empty fields kept)."""
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def main():
    """Write source1_entity_id / candidate_entity_ids = final matches + teammate matches (deduplicated, in order)."""
    final, teammate, out = sys.argv[1:4]
    F = rd(final)
    T = rd(teammate)
    T = T.set_index(T.columns[0])[T.columns[1]].reindex(F.source1_entity_id).fillna("")
    cand = [",".join(dict.fromkeys(x for x in (a + "," + b).split(",") if x)) for a, b in zip(F.matched_entity_ids, T.values)]
    pd.DataFrame({"source1_entity_id": F.source1_entity_id, "candidate_entity_ids": cand}).to_csv(
        out, sep="\t", index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    main()
