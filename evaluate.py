#!/usr/bin/env python3
"""
evaluate.py - Local evaluation harness for the Amazon ML Challenge 2026
(Business Entity Resolution).

Stdlib only (no pandas needed), so it runs anywhere and can't be broken by
dependency changes. The pipeline may import `f05`, `macro_f05`,
`load_truth`, `read_tsv` from here for threshold tuning.

WHY THIS FILE EXISTS
  We only get 5 leaderboard submissions per day. Every decision (features,
  thresholds, blocking size) must be made on a LOCAL validation score that
  mimics the official metric exactly: per-Source-1-entity F0.5, macro
  averaged, singletons included.

COMMANDS
  check-data   Profile the training data (singleton rate, match counts,
               one-to-one check, same-country check, S2 vs S3 share...).
  make-split   Hold out a validation split and export it as a fake
               "train dir" + "test dir" so the pipeline runs the SAME code path
               on validation as on the real test set.
                 --mode random        stratified by (country, singleton)
                 --mode loco:<C>      leave-one-country-out: validate on
                                      country C, train on the rest
                                      (simulates the unseen-France problem)
  score        Score a matching_results.tsv (and optionally a
               candidate_pairs.tsv) against a split's hidden ground truth.
               Prints metrics, writes error dumps, appends to experiments.csv.
  selftest     Sanity-check the metric against the example in the problem
               statement.

TYPICAL LOOP
  python3 evaluate.py check-data --train-dir dataset/train
  python3 evaluate.py make-split --train-dir dataset/train --name val --mode random
  python3 evaluate.py make-split --train-dir dataset/train --name loco_india --mode loco:India
  # run the pipeline with --train-dir splits/val/train --test-dir splits/val/test
  python3 evaluate.py score --split val \
      --pred runs/val/matching_results.tsv \
      --cand runs/val/candidate_pairs.tsv \
      --tag exp01 --note "baseline tfidf name only"
"""

import argparse
import csv
import datetime
import json
import os
import random
import sys
from collections import Counter, defaultdict

SPLITS_DIR = "splits"
REPORTS_DIR = "reports"
EXPERIMENTS_CSV = "experiments.csv"
SOURCE_FIELDS = ["entity_id", "business_name", "business_address", "country"]


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def read_tsv(path):
    """Read a TSV robustly: explicit tab split, no quote handling (addresses
    may contain quotes/commas), tolerant of a BOM, CRLF and missing trailing
    empty fields. Returns (header, list_of_dicts)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().split("\n")
    lines = [ln.rstrip("\r") for ln in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return [], []
    header = [h.strip() for h in lines[0].split("\t")]
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        elif len(parts) > len(header):
            # Extra tabs: glue the overflow back into the last column.
            parts = parts[: len(header) - 1] + ["\t".join(parts[len(header) - 1:])]
        rows.append({h: p for h, p in zip(header, parts)})
    return header, rows


def write_tsv(path, header, rows):
    """Write TSV with no quoting. rows = list of lists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")


def parse_ids(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def load_source(path):
    """Returns dict entity_id -> record dict."""
    _, rows = read_tsv(path)
    return {r["entity_id"].strip(): r for r in rows}


def load_sources(data_dir, prefix):
    """prefix is 'train' or 'test'. Returns (s1, s2, s3) dicts."""
    return tuple(
        load_source(os.path.join(data_dir, f"{prefix}_source{i}.tsv")) for i in (1, 2, 3)
    )


def load_truth(path):
    """Returns dict s1_id -> set of matched ids."""
    _, rows = read_tsv(path)
    return {r["source1_entity_id"].strip(): set(parse_ids(r.get("matched_entity_ids", "")))
            for r in rows}


def load_id_lists(path, col):
    """Load matching_results.tsv or candidate_pairs.tsv.
    Returns (dict s1 -> list (order kept, dups kept), list_of_issues)."""
    header, rows = read_tsv(path)
    issues = []
    if "source1_entity_id" not in header or col not in header:
        issues.append(f"{path}: expected columns 'source1_entity_id' and '{col}', got {header}")
    out = {}
    for r in rows:
        s1 = r.get("source1_entity_id", "").strip()
        if s1 in out:
            issues.append(f"duplicate row for {s1}")
        out[s1] = parse_ids(r.get(col, ""))
    return out, issues


# --------------------------------------------------------------------------
# Metric (exactly as specified by the organisers)
# --------------------------------------------------------------------------
def f05(pred, truth):
    """Per-entity F0.5.
    - truth empty & pred empty      -> 1.0  (correct singleton)
    - truth empty & pred non-empty  -> 0.0  (false merge on singleton)
    - truth non-empty & pred empty  -> 0.0  (recall 0)
    - otherwise standard F-beta with beta = 0.5."""
    pred, truth = set(pred), set(truth)
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_by_s1, truth_by_s1):
    """Macro-average over ALL Source 1 entities in truth. Missing predictions
    count as empty lists (the real portal would reject the file instead)."""
    if not truth_by_s1:
        return 0.0
    return sum(f05(pred_by_s1.get(s1, ()), t) for s1, t in truth_by_s1.items()) / len(truth_by_s1)


# --------------------------------------------------------------------------
# check-data
# --------------------------------------------------------------------------
def cmd_check_data(args):
    s1, s2, s3 = load_sources(args.train_dir, "train")
    truth = load_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    pool = {**s2, **s3}

    print("=" * 70)
    print("DATA PROFILE")
    print("=" * 70)
    for name, d in (("S1", s1), ("S2", s2), ("S3", s3)):
        c = Counter(r.get("country", "").strip() for r in d.values())
        print(f"{name}: {len(d):>7} records | countries: {dict(c)}")
        bad = [k for k in d if not k.startswith(name + "-")]
        if bad:
            print(f"  !! {len(bad)} ids without prefix {name}-  e.g. {bad[:3]}")

    print(f"\nGround truth rows: {len(truth)}")
    missing_rows = set(s1) - set(truth)
    extra_rows = set(truth) - set(s1)
    if missing_rows:
        print(f"  !! {len(missing_rows)} S1 ids have no ground-truth row (treated as singletons)")
    if extra_rows:
        print(f"  !! {len(extra_rows)} ground-truth rows reference unknown S1 ids")

    n_single = sum(1 for s in s1 if not truth.get(s))
    print(f"\nSingleton rate (S1 with no matches): {n_single}/{len(s1)} = {n_single / max(len(s1), 1):.3f}")
    print("  -> predicting EMPTY for everything scores exactly this on validation.")

    by_country = defaultdict(lambda: [0, 0])
    for s, r in s1.items():
        c = r.get("country", "").strip()
        by_country[c][0] += 1
        by_country[c][1] += 0 if truth.get(s) else 1
    for c, (n, k) in sorted(by_country.items()):
        print(f"  {c or '<blank>'}: {n} S1, singleton rate {k / max(n, 1):.3f}")

    k_dist = Counter(len(truth.get(s, ())) for s in s1)
    print("\nMatches per S1 entity (count: #entities):")
    for k in sorted(k_dist):
        print(f"  {k}: {k_dist[k]}")

    src_mix = Counter()
    for s in s1:
        t = truth.get(s, set())
        has2 = any(x.startswith("S2-") for x in t)
        has3 = any(x.startswith("S3-") for x in t)
        src_mix[("S2" if has2 else "") + ("+" if has2 and has3 else "") + ("S3" if has3 else "") or "none"] += 1
    print(f"\nWhich sources the matches come from: {dict(src_mix)}")

    # One-to-one check: does each S2/S3 record belong to at most one S1?
    owner = defaultdict(list)
    unknown_ids = 0
    for s, t in truth.items():
        for x in t:
            owner[x].append(s)
            if x not in pool:
                unknown_ids += 1
    multi = {x: o for x, o in owner.items() if len(o) > 1}
    print(f"\nS2/S3 ids appearing in >1 S1 match list: {len(multi)}"
          + ("   -> ONE-TO-ONE ASSIGNMENT IS SAFE" if not multi else "   -> one-to-one NOT strict, check before using it"))
    if unknown_ids:
        print(f"  !! {unknown_ids} matched ids not found in S2/S3 files")
    unmatched = [x for x in pool if x not in owner]
    print(f"S2/S3 records matched to NO S1 (distractors): {len(unmatched)}/{len(pool)} = {len(unmatched) / max(len(pool), 1):.3f}")

    # S2/S3 internal duplicates: how often does one S1 own several S2 records?
    multi_same_src = sum(1 for s, t in truth.items()
                         if sum(x.startswith("S2-") for x in t) > 1 or sum(x.startswith("S3-") for x in t) > 1)
    print(f"S1 entities with >1 match from the SAME source: {multi_same_src}  (sources 2/3 are NOT deduplicated)")

    # Same-country check (decides whether blocking by country is safe).
    same = diff = 0
    for s, t in truth.items():
        c1 = s1.get(s, {}).get("country", "").strip()
        for x in t:
            if x in pool:
                if pool[x].get("country", "").strip() == c1:
                    same += 1
                else:
                    diff += 1
    print(f"\nTrue pairs with same country label: {same}/{same + diff}"
          + ("   -> blocking within country is SAFE" if diff == 0 else f"   -> {diff} cross-country pairs, do NOT hard-block on country"))

    # Cheap signal checks
    def norm(x):
        return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in x).split())
    exact_name = sum(1 for s, t in truth.items() for x in t
                     if x in pool and norm(s1.get(s, {}).get("business_name", "")) == norm(pool[x].get("business_name", "")))
    total_pairs = sum(len(t) for t in truth.values())
    print(f"True pairs with exactly equal normalised name: {exact_name}/{total_pairs}")

    empties = Counter()
    for d in (s1, s2, s3):
        for r in d.values():
            for fld in ("business_name", "business_address", "country"):
                if not r.get(fld, "").strip():
                    empties[fld] += 1
    print(f"Empty fields across all sources: {dict(empties) or 'none'}")

    print("\nSample true matches:")
    rnd = random.Random(0)
    sample = [s for s in truth if truth[s]]
    rnd.shuffle(sample)
    for s in sample[: args.samples]:
        r = s1[s]
        print(f"\n  {s} [{r.get('country', '')}] {r.get('business_name', '')} | {r.get('business_address', '')}")
        for x in sorted(truth[s]):
            if x in pool:
                q = pool[x]
                print(f"    {x} [{q.get('country', '')}] {q.get('business_name', '')} | {q.get('business_address', '')}")
    print("\nSample singletons:")
    singles = [s for s in s1 if not truth.get(s)]
    rnd.shuffle(singles)
    for s in singles[: max(3, args.samples // 3)]:
        r = s1[s]
        print(f"  {s} [{r.get('country', '')}] {r.get('business_name', '')} | {r.get('business_address', '')}")


# --------------------------------------------------------------------------
# make-split
# --------------------------------------------------------------------------
def cmd_make_split(args):
    s1, s2, s3 = load_sources(args.train_dir, "train")
    truth = load_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    rnd = random.Random(args.seed)

    owner = {}
    for s, t in truth.items():
        for x in sorted(t):
            owner.setdefault(x, s)  # first owner wins if (unexpectedly) shared

    if args.mode == "random":
        strata = defaultdict(list)
        for s in sorted(s1):
            strata[(s1[s].get("country", "").strip(), bool(truth.get(s)))].append(s)
        val_s1 = set()
        for key in sorted(strata):
            ids = strata[key]
            rnd.shuffle(ids)
            val_s1.update(ids[: int(round(len(ids) * args.val_frac))])
        hold_country = None
    elif args.mode.startswith("loco:"):
        hold_country = args.mode.split(":", 1)[1]
        val_s1 = {s for s in s1 if s1[s].get("country", "").strip() == hold_country}
        if not val_s1:
            sys.exit(f"No S1 records with country == {hold_country!r}")
    else:
        sys.exit("--mode must be 'random' or 'loco:<Country>'")

    def side_of(x, rec):
        if x in owner:
            return "val" if owner[x] in val_s1 else "train"
        if hold_country is not None:
            return "val" if rec.get("country", "").strip() == hold_country else "train"
        return "val" if rnd.random() < args.val_frac else "train"

    side = {}
    for d in (s2, s3):
        for x in sorted(d):
            side[x] = side_of(x, d[x])

    base = os.path.join(SPLITS_DIR, args.name)
    tr_dir, te_dir = os.path.join(base, "train"), os.path.join(base, "test")

    def dump(d, keep, path):
        rows = [[d[k].get(f, "") for f in SOURCE_FIELDS] for k in sorted(d) if keep(k)]
        write_tsv(path, SOURCE_FIELDS, rows)
        return len(rows)

    n = {}
    n["train_s1"] = dump(s1, lambda k: k not in val_s1, os.path.join(tr_dir, "train_source1.tsv"))
    n["train_s2"] = dump(s2, lambda k: side[k] == "train", os.path.join(tr_dir, "train_source2.tsv"))
    n["train_s3"] = dump(s3, lambda k: side[k] == "train", os.path.join(tr_dir, "train_source3.tsv"))
    n["val_s1"] = dump(s1, lambda k: k in val_s1, os.path.join(te_dir, "test_source1.tsv"))
    n["val_s2"] = dump(s2, lambda k: side[k] == "val", os.path.join(te_dir, "test_source2.tsv"))
    n["val_s3"] = dump(s3, lambda k: side[k] == "val", os.path.join(te_dir, "test_source3.tsv"))

    def gt_rows(keep):
        return [[s, ",".join(sorted(x for x in truth.get(s, set()) if x in side))]
                for s in sorted(s1) if keep(s)]

    write_tsv(os.path.join(tr_dir, "train_ground_truth.tsv"),
              ["source1_entity_id", "matched_entity_ids"], gt_rows(lambda s: s not in val_s1))
    # Hidden answers for the validation "test" set. The pipeline must NEVER read this.
    write_tsv(os.path.join(base, "val_ground_truth.tsv"),
              ["source1_entity_id", "matched_entity_ids"], gt_rows(lambda s: s in val_s1))

    meta = {"name": args.name, "mode": args.mode, "seed": args.seed, "val_frac": args.val_frac,
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "counts": n}
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Split '{args.name}' written to {base}/")
    print(json.dumps(n, indent=2))
    print(f"  pipeline train dir : {tr_dir}   (files named train_*.tsv)")
    print(f"  pipeline test dir  : {te_dir}   (files named test_*.tsv)")
    print(f"  hidden answers     : {base}/val_ground_truth.tsv   (ONLY evaluate.py reads this)")


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
def cmd_score(args):
    base = os.path.join(SPLITS_DIR, args.split)
    truth = load_truth(os.path.join(base, "val_ground_truth.tsv"))
    s1, s2, s3 = load_sources(os.path.join(base, "test"), "test")
    pool = {**s2, **s3}

    pred, issues = load_id_lists(args.pred, "matched_entity_ids")
    fmt_issues = list(issues)

    # --- format checks mirroring the official validator's rules ---
    for s in truth:
        if s not in pred:
            fmt_issues.append(f"missing row for {s}")
    for s, ids in pred.items():
        if s not in truth:
            fmt_issues.append(f"unknown source1 id {s}")
        if len(ids) != len(set(ids)):
            fmt_issues.append(f"duplicate ids in list for {s}")
        for x in ids:
            if not (x.startswith("S2-") or x.startswith("S3-")):
                fmt_issues.append(f"{s}: non S2/S3 id {x}")
            elif x not in pool:
                fmt_issues.append(f"{s}: id {x} not in test pool")

    cand = None
    if args.cand:
        cand, c_issues = load_id_lists(args.cand, "candidate_entity_ids")
        fmt_issues += ["[cand] " + i for i in c_issues]
        for s in truth:
            if s not in cand:
                fmt_issues.append(f"[cand] missing row for {s}")
        for s, ids in pred.items():
            extra = set(ids) - set(cand.get(s, ()))
            if extra:
                fmt_issues.append(f"{s}: {len(extra)} matched ids not in candidates (pipeline bug)")

    # --- metric ---
    per = {s: f05(pred.get(s, ()), t) for s, t in truth.items()}
    macro = sum(per.values()) / max(len(per), 1)

    singles = [s for s, t in truth.items() if not t]
    nonsingles = [s for s, t in truth.items() if t]
    single_acc = sum(per[s] for s in singles) / max(len(singles), 1)
    nonsingle_f = sum(per[s] for s in nonsingles) / max(len(nonsingles), 1)
    false_empty = sum(1 for s in nonsingles if not pred.get(s))

    tp = sum(len(set(pred.get(s, ())) & t) for s, t in truth.items())
    npred = sum(len(set(pred.get(s, ()))) for s in truth)
    ntrue = sum(len(t) for t in truth.values())
    micro_p = tp / npred if npred else 0.0
    micro_r = tp / ntrue if ntrue else 0.0

    by_src = {}
    for src in ("S2-", "S3-"):
        tps = sum(len({x for x in pred.get(s, ()) if x.startswith(src)} & t) for s, t in truth.items())
        nps = sum(len({x for x in pred.get(s, ()) if x.startswith(src)}) for s in truth)
        nts = sum(len({x for x in t if x.startswith(src)}) for t in truth.values())
        by_src[src[:2]] = (tps / nps if nps else 0.0, tps / nts if nts else 0.0)

    by_country = defaultdict(list)
    for s in truth:
        by_country[s1.get(s, {}).get("country", "").strip()].append(per[s])

    # Oracle mode helps separate "blocking problem" from "model problem".
    ceiling = None
    blk = {}
    if cand is not None:
        oracle = {s: set(cand.get(s, ())) & t for s, t in truth.items()}
        ceiling = macro_f05(oracle, truth)
        cov = sum(len(set(cand.get(s, ())) & t) for s, t in truth.items())
        ncand = sum(len(set(cand.get(s, ()))) for s in truth)
        full = sum(1 for s in nonsingles if t_sub(truth[s], cand.get(s, ())))
        blk = {
            "pair_recall": cov / ntrue if ntrue else 0.0,
            "entities_fully_covered": full / max(len(nonsingles), 1),
            "avg_candidates_per_s1": ncand / max(len(truth), 1),
            "reduction_ratio": 1 - ncand / max(len(truth) * max(len(pool), 1), 1),
            "oracle_f05_ceiling": ceiling,
        }

    res = {
        "macro_f05": macro,
        "singleton_acc": single_acc,
        "nonsingleton_f05": nonsingle_f,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "false_empty": false_empty,
        "n_s1": len(truth),
        "n_singletons": len(singles),
        "n_pred_pairs": npred,
        "n_true_pairs": ntrue,
        **blk,
    }

    print("=" * 70)
    print(f"SCORE  split={args.split}  tag={args.tag or '-'}")
    print("=" * 70)
    print(f"MACRO F0.5 (official metric)  : {macro:.5f}")
    print(f"  singleton accuracy          : {single_acc:.4f}   ({len(singles)} singletons)")
    print(f"  F0.5 on non-singletons      : {nonsingle_f:.4f}   ({len(nonsingles)} entities)")
    print(f"  non-singletons predicted [] : {false_empty}")
    print(f"  micro precision / recall    : {micro_p:.4f} / {micro_r:.4f}   ({npred} predicted pairs, {ntrue} true)")
    for k, (p, r) in by_src.items():
        print(f"  {k} precision / recall       : {p:.4f} / {r:.4f}")
    for c, v in sorted(by_country.items()):
        print(f"  country {c or '<blank>':<12}        : {sum(v) / len(v):.4f}   (n={len(v)})")
    if blk:
        print("BLOCKING")
        print(f"  pair recall                 : {blk['pair_recall']:.4f}")
        print(f"  entities fully covered      : {blk['entities_fully_covered']:.4f}")
        print(f"  avg candidates per S1       : {blk['avg_candidates_per_s1']:.1f}")
        print(f"  reduction ratio             : {blk['reduction_ratio']:.5f}")
        print(f"  ORACLE F0.5 ceiling         : {ceiling:.5f}   (perfect model on these candidates)")
    if fmt_issues:
        print(f"\nFORMAT ISSUES ({len(fmt_issues)}) - the portal would REJECT this file:")
        for i in fmt_issues[:20]:
            print("  - " + i)
        if len(fmt_issues) > 20:
            print(f"  ... and {len(fmt_issues) - 20} more")
    else:
        print("\nFormat checks: PASS")

    # --- error dumps for manual error analysis ---
    if args.tag:
        os.makedirs(REPORTS_DIR, exist_ok=True)

        def rec(x, d):
            r = d.get(x, {})
            return [x, r.get("business_name", ""), r.get("business_address", ""), r.get("country", "")]

        hdr = ["s1_id", "s1_name", "s1_address", "s1_country", "other_id", "other_name", "other_address", "other_country"]
        fp, fn, miss = [], [], []
        for s, t in truth.items():
            p = set(pred.get(s, ()))
            for x in sorted(p - t):
                fp.append(rec(s, s1) + rec(x, pool) + ["SINGLETON" if not t else ""])
            for x in sorted(t - p):
                in_cand = cand is None or x in set(cand.get(s, ()))
                (fn if in_cand else miss).append(rec(s, s1) + rec(x, pool))
        write_tsv(os.path.join(REPORTS_DIR, f"{args.tag}_false_pos.tsv"), hdr + ["note"], fp)
        write_tsv(os.path.join(REPORTS_DIR, f"{args.tag}_false_neg.tsv"), hdr, fn)
        if cand is not None:
            write_tsv(os.path.join(REPORTS_DIR, f"{args.tag}_missed_by_blocking.tsv"), hdr, miss)
        print(f"\nError dumps -> {REPORTS_DIR}/{args.tag}_*.tsv  (FP={len(fp)}, FN={len(fn)}, blocking misses={len(miss)})")

        # --- experiment log ---
        row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
               "tag": args.tag, "split": args.split, "note": args.note or "",
               **{k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in res.items()},
               "format_issues": len(fmt_issues)}
        new = not os.path.exists(EXPERIMENTS_CSV)
        fields = list(row.keys())
        if not new:
            with open(EXPERIMENTS_CSV, encoding="utf-8") as f:
                old_fields = next(csv.reader(f), [])
            fields = old_fields + [k for k in row if k not in old_fields]
            if fields != old_fields:  # schema grew: rewrite header
                with open(EXPERIMENTS_CSV, encoding="utf-8") as f:
                    old_rows = list(csv.DictReader(f))
                with open(EXPERIMENTS_CSV, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(old_rows)
        with open(EXPERIMENTS_CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"Logged to {EXPERIMENTS_CSV}")

    if args.json:
        print(json.dumps(res, indent=2))
    return 1 if fmt_issues else 0


def t_sub(t, cand_list):
    return set(t) <= set(cand_list)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def cmd_selftest(_):
    ex = f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert abs(ex - 0.714) < 0.001, ex
    assert f05([], []) == 1.0
    assert f05(["S2-1"], []) == 0.0
    assert f05([], ["S2-1"]) == 0.0
    assert f05(["S2-2"], ["S2-1"]) == 0.0
    assert abs(f05(["S2-1"], ["S2-1", "S2-2", "S3-3"]) - 1.25 * (1 / 3) / (0.25 + 1 / 3)) < 1e-9
    assert macro_f05({"a": [], "b": ["S2-1"]}, {"a": set(), "b": {"S2-1"}}) == 1.0
    print(f"selftest PASS (problem-statement example = {ex:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("check-data")
    a.add_argument("--train-dir", default="dataset/train")
    a.add_argument("--samples", type=int, default=8)

    a = sub.add_parser("make-split")
    a.add_argument("--train-dir", default="dataset/train")
    a.add_argument("--name", default="val")
    a.add_argument("--mode", default="random")
    a.add_argument("--val-frac", type=float, default=0.2)
    a.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("score")
    a.add_argument("--split", default="val")
    a.add_argument("--pred", required=True)
    a.add_argument("--cand")
    a.add_argument("--tag")
    a.add_argument("--note")
    a.add_argument("--json", action="store_true")

    sub.add_parser("selftest")

    args = ap.parse_args()
    fn = {"check-data": cmd_check_data, "make-split": cmd_make_split,
          "score": cmd_score, "selftest": cmd_selftest}[args.cmd]
    rc = fn(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
