"""Safe TSV reading and submission writing.

Every source file is read with an explicit tab separator, all columns as strings,
no NaN conversion and no quote handling, so ids like ``S2-00047`` and empty fields
survive unchanged.
"""
import csv
import os

import pandas as pd

SOURCE_FIELDS = ["entity_id", "business_name", "business_address", "country"]
READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False,
               quoting=csv.QUOTE_NONE, encoding="utf-8")


def read_tsv(path):
    """Read a TSV file as all-string columns with empty strings kept as ''."""
    return pd.read_csv(path, **READ_KW)


def read_source(data_dir, prefix, source_no):
    """Read ``<prefix>_source<N>.tsv`` from ``data_dir`` and add a ``src`` column (1/2/3)."""
    df = read_tsv(os.path.join(data_dir, f"{prefix}_source{source_no}.tsv"))
    for col in SOURCE_FIELDS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df = df[SOURCE_FIELDS].copy()
    df["src"] = source_no
    return df


def read_sources(data_dir, prefix):
    """Return (s1, pool) where pool is the concatenation of sources 2 and 3."""
    s1 = read_source(data_dir, prefix, 1)
    pool = pd.concat([read_source(data_dir, prefix, 2), read_source(data_dir, prefix, 3)],
                     ignore_index=True)
    return s1, pool


def read_truth(data_dir, prefix="train"):
    """Read the ground truth as a dict ``s1_id -> set(matched ids)`` (empty set for singletons)."""
    df = read_tsv(os.path.join(data_dir, f"{prefix}_ground_truth.tsv"))
    return {s: set(m.split(",")) if m else set()
            for s, m in zip(df["source1_entity_id"], df["matched_entity_ids"])}


def write_id_lists(path, header, s1_ids, lists):
    """Write one row per S1 id with a comma-joined (no spaces, de-duplicated) id list.

    ``lists`` maps s1_id -> iterable of ids; missing S1 ids get an empty field.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    seen = set()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            if s in seen:
                continue
            seen.add(s)
            ids = []
            used = set()
            for x in lists.get(s, ()):
                if x and x not in used and not x.startswith("S1-"):
                    used.add(x)
                    ids.append(x)
            f.write(f"{s}\t{','.join(ids)}\n")


def write_outputs(out_dir, s1_ids, matches, candidates):
    """Write matching_results.tsv and candidate_pairs.tsv, forcing matches ⊆ candidates."""
    safe_matches = {}
    for s, m in matches.items():
        cand = set(candidates.get(s, ()))
        safe_matches[s] = [x for x in m if x in cand]
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"),
                   ["source1_entity_id", "matched_entity_ids"], s1_ids, safe_matches)
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"),
                   ["source1_entity_id", "candidate_entity_ids"], s1_ids, candidates)
