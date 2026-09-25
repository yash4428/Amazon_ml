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
from features import chunk_features, global_context  # noqa: E402
from model import predict, train_oof  # noqa: E402
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


def fit_lgbm(args, report_dir):
    """Train-part work for the LightGBM mode: blocking on ALL train S1, features on a
    seeded S1 sample, grouped-CV OOF probabilities, decision rule tuned on OOF."""
    log("loading train")
    s1_tr, pool_tr = load_normalized(args.train_dir, "train", n_jobs=config.N_JOBS)
    truth = read_truth(args.train_dir, "train")
    log(f"blocking train: {len(s1_tr)} S1 x {len(pool_tr)} pool")
    cand = blocking_cached(s1_tr, pool_tr, "train")
    cand = pd.concat([cand, global_context(cand)], axis=1)
    rng = np.random.RandomState(config.SEED)
    n_model = min(args.model_s1, len(s1_tr))
    pos = np.sort(rng.choice(len(s1_tr), n_model, replace=False))
    sub = cand[np.isin(cand["i"].values, pos)].copy()
    del cand
    n_true = label_pairs(sub, s1_tr, pool_tr, truth)
    log(f"model pairs {len(sub)} from {n_model} S1, positives {int(sub.y.sum())}, "
        f"pair recall {sub.y.sum() / max(n_true.iloc[pos].sum(), 1):.4f}")
    X = chunk_features(sub, s1_tr, pool_tr)
    y = sub["y"].values.astype(np.int8)
    log(f"features {X.shape}")
    oof, models, imp = train_oof(X, y, sub["i"].values)
    os.makedirs(report_dir, exist_ok=True)
    pd.Series(imp, index=X.columns).sort_values(ascending=False).to_csv(
        os.path.join(report_dir, "feature_importance.csv"), header=["gain"])
    # tune decision on OOF, with S1 re-indexed to 0..n_model-1 (singletons included)
    local = np.searchsorted(pos, sub["i"].values)
    c = pd.DataFrame({"i": local, "j": sub["j"].values, "y": sub["y"].values, "score": oof})
    nt = pd.Series(n_true.values[pos], index=np.arange(n_model))
    params, f_oof = tune_rule(c, nt, np.arange(n_model))
    return models, list(X.columns), params, f_oof


def predict_lgbm(cand, s1, pool, models, cols):
    """Score test candidates chunk by chunk (whole S1 groups per chunk)."""
    cand = pd.concat([cand, global_context(cand)], axis=1)
    cand = cand.sort_values("i", kind="stable").reset_index(drop=True)
    score = np.zeros(len(cand), np.float32)
    bounds = np.searchsorted(cand["i"].values, np.arange(0, len(s1) + config.INFER_CHUNK_S1,
                                                          config.INFER_CHUNK_S1))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        f = chunk_features(cand.iloc[a:b], s1, pool)
        score[a:b] = predict(models, f[cols])
        log(f"  scored pairs {b}/{len(cand)}")
    cand["score"] = score
    return cand


def main():
    """Parse args, tune on the train part, predict on the test part, write both output files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["lgbm", "rule"], default="rule")
    ap.add_argument("--model-s1", type=int, default=config.MODEL_S1)
    ap.add_argument("--tune-s1", type=int, default=config.TUNE_S1,
                    help="number of train S1 records used for tuning (full train pool is kept)")
    args = ap.parse_args()
    np.random.seed(config.SEED)

    report_dir = os.path.join(args.out_dir, "report")
    if args.mode == "lgbm":
        models, cols, params, f_tr = fit_lgbm(args, report_dir)
        log(f"OOF tuned params {params}, OOF macroF0.5 {f_tr:.4f}")
    else:
        params, f_tr = fit_rule(args)

    # ---------------- inference on the test part
    log("loading test")
    s1_te, pool_te = load_normalized(args.test_dir, "test", n_jobs=config.N_JOBS)
    log(f"blocking test: {len(s1_te)} S1 x {len(pool_te)} pool")
    cand_te = blocking_cached(s1_te, pool_te, "test")
    if args.mode == "lgbm":
        cand_te = predict_lgbm(cand_te, s1_te, pool_te, models, cols)
    else:
        cand_te["score"] = baseline_score(cand_te)
    pred = apply_rule(cand_te, *params)
    write_run(args, s1_te, pool_te, cand_te, pred, params, f_tr)


def fit_rule(args):
    """Rule-baseline tuning on a sample of train S1 (full train pool)."""
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
    return params, f_tr


def write_run(args, s1_te, pool_te, cand_te, pred, params, f_tr):
    """Write matching_results.tsv, candidate_pairs.tsv and run_info.json."""

    s1_ids = s1_te["entity_id"].values
    pool_ids = pool_te["entity_id"].values
    candidates = group_ids(cand_te, s1_ids, pool_ids)
    matches = group_ids(pred, s1_ids, pool_ids)
    write_outputs(args.out_dir, list(s1_ids), matches, candidates)
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump({"mode": args.mode, "params": params, "train_macro_f05": f_tr, "blocking": config.BLOCKING,
                   "n_s1": len(s1_ids), "n_pred_pairs": int(len(pred)),
                   "n_cand_pairs": int(len(cand_te))}, f, indent=2)
    log(f"wrote {args.out_dir}: {len(pred)} matches, {len(cand_te)} candidates, "
        f"empty rows {1 - len(matches) / len(s1_ids):.3f}")


if __name__ == "__main__":
    main()
