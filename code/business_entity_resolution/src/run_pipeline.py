"""End-to-end pipeline CLI.

    python run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir output

Reads train_*.tsv from --train-dir (with train_ground_truth.tsv) and test_*.tsv
from --test-dir, so the same code runs on validation splits (splits/<name>/...)
and on the real test set. Validation answers are never read here.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from blocking import generate_candidates  # noqa: E402
from decide import apply_rule, tune_rule  # noqa: E402
from io_utils import read_truth, write_outputs  # noqa: E402
from normalize import load_normalized  # noqa: E402


def log(msg):
    """Timestamped progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blocking_cached(s1, pool, tag):
    """Run blocking, caching the candidate frame by (record ids, blocking config)."""
    h = hashlib.md5()
    h.update(json.dumps(config.BLOCKING, sort_keys=True).encode())
    h.update(str(len(s1)).encode() + str(len(pool)).encode())
    h.update(",".join(s1["entity_id"].iloc[:: max(1, len(s1) // 1000)]).encode())
    h.update(",".join(pool["entity_id"].iloc[:: max(1, len(pool) // 1000)]).encode())
    path = os.path.join(config.CACHE_DIR, "blocks", f"{tag}_{h.hexdigest()[:12]}.parquet")
    if os.path.exists(path):
        log(f"blocking cache hit {path}")
        return pd.read_parquet(path)
    cand = generate_candidates(s1, pool)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cand.to_parquet(path, index=False)
    return cand


def baseline_score(c):
    """Rule-based pair score: the combined name+address cosine, backed off to name char sim."""
    return np.maximum(c["combo_sim"].values, 0.8 * c["name_c4_sim"].values)


def label_pairs(cand, s1, pool, truth):
    """Add y (is true match) and return n_true per S1 position."""
    s1_ids = s1["entity_id"].values
    pool_ids = pool["entity_id"].values
    cand["y"] = [x in truth.get(s, ()) for s, x in zip(s1_ids[cand["i"].values], pool_ids[cand["j"].values])]
    n_true = pd.Series([len(truth.get(s, ())) for s in s1_ids], index=np.arange(len(s1)))
    return n_true


def group_ids(pairs, s1_ids, pool_ids):
    """Map s1_id -> list of pool ids (best score first) from a pair frame with i, j, score."""
    o = pairs.sort_values(["i", "score"], ascending=[True, False], kind="stable")
    i = o["i"].values
    ids = pool_ids[o["j"].values]
    cuts = np.flatnonzero(np.diff(i)) + 1
    starts = np.concatenate([[0], cuts]) if len(i) else np.empty(0, int)
    return {s1_ids[i[a]]: list(g) for a, g in zip(starts, np.split(ids, cuts))} if len(i) else {}


def main():
    """Parse args, tune on the train part, predict on the test part, write both output files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tune-s1", type=int, default=config.TUNE_S1,
                    help="number of train S1 records used for tuning (full train pool is kept)")
    args = ap.parse_args()
    np.random.seed(config.SEED)

    # ---------------- tuning on the labelled train part
    log("loading train")
    s1_tr, pool_tr = load_normalized(args.train_dir, "train", n_jobs=config.N_JOBS)
    truth = read_truth(args.train_dir, "train")
    if args.tune_s1 and args.tune_s1 < len(s1_tr):
        s1_tr = s1_tr.sample(args.tune_s1, random_state=config.SEED).reset_index(drop=True)
    log(f"blocking train: {len(s1_tr)} S1 x {len(pool_tr)} pool")
    cand_tr = blocking_cached(s1_tr, pool_tr, "train")
    n_true = label_pairs(cand_tr, s1_tr, pool_tr, truth)
    cand_tr["score"] = baseline_score(cand_tr)
    log(f"train candidates {len(cand_tr)}, pair recall {cand_tr.y.sum() / max(n_true.sum(), 1):.4f}")
    params, f_tr = tune_rule(cand_tr, n_true, np.arange(len(s1_tr)))
    del s1_tr, pool_tr, cand_tr

    # ---------------- inference on the test part
    log("loading test")
    s1_te, pool_te = load_normalized(args.test_dir, "test", n_jobs=config.N_JOBS)
    log(f"blocking test: {len(s1_te)} S1 x {len(pool_te)} pool")
    cand_te = blocking_cached(s1_te, pool_te, "test")
    cand_te["score"] = baseline_score(cand_te)
    pred = apply_rule(cand_te, *params)

    s1_ids = s1_te["entity_id"].values
    pool_ids = pool_te["entity_id"].values
    candidates = group_ids(cand_te, s1_ids, pool_ids)
    matches = group_ids(pred, s1_ids, pool_ids)
    write_outputs(args.out_dir, list(s1_ids), matches, candidates)
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump({"params": params, "train_macro_f05": f_tr, "blocking": config.BLOCKING,
                   "n_s1": len(s1_ids), "n_pred_pairs": int(len(pred)),
                   "n_cand_pairs": int(len(cand_te))}, f, indent=2)
    log(f"wrote {args.out_dir}: {len(pred)} matches, {len(cand_te)} candidates, "
        f"empty rows {1 - len(matches) / len(s1_ids):.3f}")


if __name__ == "__main__":
    main()
