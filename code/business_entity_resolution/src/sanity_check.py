"""Submission sanity checks (CLAUDE.md §9).

    python sanity_check.py --out-dir output --test-dir dataset/test

Prints row counts, empty-prediction share (overall and per country, France
included), average matches per non-empty row, and verifies matches ⊆ candidates,
one row per S1 id, no S1 ids / duplicates / unknown ids in the lists.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import read_source, read_tsv  # noqa: E402


def main():
    """Run all checks and exit 1 if any hard check fails."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    s1 = read_source(args.test_dir, "test", 1)
    country = dict(zip(s1["entity_id"], s1["country"]))
    valid = set()
    for n in (2, 3):
        valid.update(read_source(args.test_dir, "test", n)["entity_id"])
    m = read_tsv(os.path.join(args.out_dir, "matching_results.tsv"))
    c = read_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"))
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print("FAIL:", msg)

    for name, df, col in (("matching", m, "matched_entity_ids"), ("candidate", c, "candidate_entity_ids")):
        if len(df) != len(s1) or set(df["source1_entity_id"]) != set(country):
            fail(f"{name}: {len(df)} rows vs {len(s1)} test S1 ids (or id sets differ)")
        if df["source1_entity_id"].duplicated().any():
            fail(f"{name}: duplicate S1 rows")
        bad = 0
        for v in df[col]:
            ids = v.split(",") if v else []
            if len(ids) != len(set(ids)) or any(x not in valid for x in ids):
                bad += 1
        if bad:
            fail(f"{name}: {bad} rows with duplicate/unknown/S1 ids")
    cand = dict(zip(c["source1_entity_id"], c["candidate_entity_ids"]))
    not_sub = sum(1 for s, v in zip(m["source1_entity_id"], m["matched_entity_ids"])
                  if v and not set(v.split(",")) <= set(cand.get(s, "").split(",")))
    if not_sub:
        fail(f"{not_sub} rows where matches are not a subset of candidates")

    m["country"] = m["source1_entity_id"].map(country)
    m["n"] = m["matched_entity_ids"].map(lambda v: len(v.split(",")) if v else 0)
    c["n"] = c["candidate_entity_ids"].map(lambda v: len(v.split(",")) if v else 0)
    print(f"rows: {len(m)} (test S1 ids: {len(s1)})")
    print(f"empty predictions: {(m.n == 0).mean():.4f}  (train singleton rate 0.056)")
    print(f"avg matches per non-empty row: {m.n[m.n > 0].mean():.3f}; avg candidates/S1: {c.n.mean():.1f}")
    g = m.groupby("country").agg(rows=("n", "size"), empty=("n", lambda x: (x == 0).mean()),
                                 avg_matches=("n", "mean"))
    print(g.round(4).to_string())
    if "France" not in g.index:
        fail("France missing from output")
    print("SANITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
