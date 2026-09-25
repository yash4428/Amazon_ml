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
from features import (add_dup_counts, chunk_features, cluster_features,  # noqa: E402
                      dup_features, encode_extras, global_context, hmis_features,
                      house_mismatch_token_rates, pair_extras, token_counts)
from model import group_folds, predict, train_full, train_oof  # noqa: E402
from io_utils import read_truth, write_outputs  # noqa: E402
from normalize import apply_translit_dict, build_translit_dict, load_normalized  # noqa: E402


def log(msg):
    """Timestamped progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blocking_cached(s1, pool, tag):
    """Run blocking, caching the candidate frame by (record ids, blocking config)."""
    h = hashlib.md5()
    h.update(json.dumps(config.BLOCKING, sort_keys=True).encode())
    h.update(str(config.NORM_VERSION).encode())
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
    c4 = c["name_c4_sim"].values if "name_c4_sim" in c else 0.0
    return np.maximum(c["combo_sim"].values, 0.8 * c4)


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


def extra_feats(c, s1, pool, neg, pos):
    """Extra-token encoding features for pairs ``c`` given token counts."""
    ex = pair_extras(c, s1, pool)
    mx, sm, unk = encode_extras(ex, neg, pos)
    return pd.DataFrame({"xtok_max": mx, "xtok_sum": sm, "xtok_unknown": unk,
                         "xtok_n": np.array([len(e) for e in ex], np.float32)}, index=c.index), ex


def fit_lgbm(args, report_dir):
    """Train-part work for the LightGBM mode: blocking on ALL train S1, features on a
    seeded S1 sample, grouped-CV OOF probabilities, decision rule tuned on OOF, and a
    final single model on all sampled pairs for inference."""
    log("loading train")
    s1_tr, pool_tr = load_normalized(args.train_dir, "train", n_jobs=config.N_JOBS)
    truth = read_truth(args.train_dir, "train")
    tl = build_translit_dict(s1_tr, pool_tr, truth) if config.USE_TRANSLIT_DICT else {}
    log(f"learned transliteration dict: {len(tl)} tokens")
    s1_tr, pool_tr = apply_translit_dict(s1_tr, tl), apply_translit_dict(pool_tr, tl)
    s1_tr, pool_tr = add_dup_counts(s1_tr, pool_tr)
    log(f"blocking train: {len(s1_tr)} S1 x {len(pool_tr)} pool")
    cand = blocking_cached(s1_tr, pool_tr, "train")
    hm_rates = house_mismatch_token_rates(cand, s1_tr, pool_tr)
    log(f"house-mismatch token rates: {len(hm_rates)} tokens")
    cand = pd.concat([cand, global_context(cand)], axis=1)
    rng = np.random.RandomState(config.SEED + args.sample_seed)
    n_model = min(args.model_s1, len(s1_tr))
    pos = np.sort(rng.choice(len(s1_tr), n_model, replace=False))
    sub = cand[np.isin(cand["i"].values, pos)].copy()
    del cand
    n_true = label_pairs(sub, s1_tr, pool_tr, truth)
    log(f"model pairs {len(sub)} from {n_model} S1, positives {int(sub.y.sum())}, "
        f"pair recall {sub.y.sum() / max(n_true.iloc[pos].sum(), 1):.4f}")
    y = sub["y"].values.astype(np.int8)
    if args.cv == "country":
        cn = s1_tr["country"].values[sub["i"].values]
        fold = pd.factorize(cn, sort=True)[0].astype(np.int8)
    else:
        fold = group_folds(sub["i"].values)
    X = chunk_features(sub, s1_tr, pool_tr)
    X = pd.concat([X, cluster_features(sub, s1_tr, pool_tr), dup_features(sub, s1_tr, pool_tr)], axis=1)
    ex = pair_extras(sub, s1_tr, pool_tr)
    X["hmis_max"], X["hmis_mean"] = hmis_features(ex, hm_rates)
    if config.USE_XTOK:
        xf = np.zeros((len(sub), 3), np.float32)
        for k in np.unique(fold):
            m = fold == k
            neg, pos_ = token_counts([e for e, t in zip(ex, ~m) if t], y[~m])
            mx, sm, unk = encode_extras([e for e, t in zip(ex, m) if t], neg, pos_)
            xf[m] = np.stack([mx, sm, unk], 1)
        X["xtok_max"], X["xtok_sum"], X["xtok_unknown"] = xf[:, 0], xf[:, 1], xf[:, 2]
        X["xtok_n"] = np.array([len(e) for e in ex], np.float32)
        tok_neg, tok_pos = token_counts(ex, y)
    else:
        tok_neg = tok_pos = None
    X = X.drop(columns=[c for c in config.DROP_FEATURES if c in X.columns])
    log(f"features {X.shape}, cv={args.cv} folds={len(np.unique(fold))}")
    if args.fixed_rounds:
        # No CV: one model on all sampled pairs, decision params taken from a validated run.
        prm = tuple(json.load(open(args.params_from))["params"])
        log(f"no-CV mode: {args.fixed_rounds} rounds, decision params {prm} from {args.params_from}")
        final = train_full(X, y, args.fixed_rounds)
        return [final], list(X.columns), prm, float("nan"), (tok_neg, tok_pos), tl
    oof, models, imp = train_oof(X, y, fold)
    os.makedirs(report_dir, exist_ok=True)
    pd.Series(imp, index=X.columns).sort_values(ascending=False).to_csv(
        os.path.join(report_dir, "feature_importance.csv"), header=["gain"])
    # tune decision on OOF, with S1 re-indexed to 0..n_model-1 (singletons included)
    local = np.searchsorted(pos, sub["i"].values)
    c = pd.DataFrame({"i": local, "j": sub["j"].values, "y": sub["y"].values, "score": oof})
    nt = pd.Series(n_true.values[pos], index=np.arange(n_model))
    params, f_oof = tune_rule(c, nt, np.arange(n_model))
    # per-country OOF score with the tuned rule (honest when cv=country)
    from decide import macro_f05_fast
    pr = apply_rule(c, *params)
    cn_s1 = s1_tr["country"].values[pos]
    per = {}
    for cty in sorted(set(cn_s1)):
        idx = np.flatnonzero(cn_s1 == cty)
        per[cty] = round(macro_f05_fast(pr[np.isin(pr["i"].values, idx)], nt, idx), 5)
    log(f"OOF per country {per}")
    info = {"oof_macro_f05": f_oof, "params": params, "per_country": per, "cv": args.cv,
            "n_features": X.shape[1], "best_iters": [m.best_iteration for m in models]}
    with open(os.path.join(report_dir, "oof.json"), "w") as f:
        json.dump(info, f, indent=2)
    dump = c.assign(s1=s1_tr["entity_id"].values[sub["i"].values],
                    x=pool_tr["entity_id"].values[sub["j"].values])
    dump.to_parquet(os.path.join(report_dir, "oof_pairs.parquet"), index=False)
    pd.DataFrame({"s1": s1_tr["entity_id"].values[pos], "n_true": nt.values}).to_parquet(
        os.path.join(report_dir, "oof_s1.parquet"), index=False)
    final = None
    if not args.oof_only:
        rounds = int(np.mean([m.best_iteration for m in models]) * 1.1)
        log(f"training final model on all {len(y)} pairs, {rounds} rounds")
        final = train_full(X, y, rounds)
    return [final] if final else models, list(X.columns), params, f_oof, (tok_neg, tok_pos), tl


def predict_lgbm(cand, s1, pool, models, cols, tok):
    """Score test candidates chunk by chunk (whole S1 groups per chunk)."""
    hm_rates = house_mismatch_token_rates(cand, s1, pool)
    log(f"test house-mismatch token rates: {len(hm_rates)} tokens")
    cand = pd.concat([cand, global_context(cand)], axis=1)
    cand = cand.sort_values("i", kind="stable").reset_index(drop=True)
    score = np.zeros(len(cand), np.float32)
    bounds = np.searchsorted(cand["i"].values, np.arange(0, len(s1) + config.INFER_CHUNK_S1,
                                                          config.INFER_CHUNK_S1))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        c = cand.iloc[a:b]
        f = pd.concat([chunk_features(c, s1, pool), cluster_features(c, s1, pool),
                       dup_features(c, s1, pool)], axis=1)
        ex = pair_extras(c, s1, pool)
        f["hmis_max"], f["hmis_mean"] = hmis_features(ex, hm_rates)
        if tok[0] is not None:
            xf, _ = extra_feats(c, s1, pool, *tok)
            f = pd.concat([f, xf], axis=1)
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
    ap.add_argument("--mode", choices=["lgbm", "rule"], default="lgbm")
    ap.add_argument("--model-s1", type=int, default=config.MODEL_S1)
    ap.add_argument("--cv", choices=["group", "country"], default="group",
                    help="country = leave-one-country-out folds (unseen-country simulation)")
    ap.add_argument("--fixed-rounds", type=int, default=0,
                    help="skip CV; train one model with this many rounds (needs --params-from)")
    ap.add_argument("--params-from", default="", help="oof.json of a validated run (decision params)")
    ap.add_argument("--sample-seed", type=int, default=0, help="offset for the train-S1 sample seed")
    ap.add_argument("--oof-only", action="store_true",
                    help="stop after OOF scoring on the train dir (no test inference)")
    ap.add_argument("--tune-s1", type=int, default=config.TUNE_S1,
                    help="number of train S1 records used for tuning (full train pool is kept)")
    args = ap.parse_args()
    np.random.seed(config.SEED)

    report_dir = os.path.join(args.out_dir, "report")
    if args.mode == "lgbm":
        models, cols, params, f_tr, tok, tl = fit_lgbm(args, report_dir)
        log(f"OOF tuned params {params}, OOF macroF0.5 {f_tr:.4f}")
        if args.oof_only:
            return
    else:
        params, f_tr = fit_rule(args)

    # ---------------- inference on the test part
    log("loading test")
    s1_te, pool_te = load_normalized(args.test_dir, "test", n_jobs=config.N_JOBS)
    if args.mode == "lgbm":
        s1_te, pool_te = apply_translit_dict(s1_te, tl), apply_translit_dict(pool_te, tl)
        s1_te, pool_te = add_dup_counts(s1_te, pool_te)
    log(f"blocking test: {len(s1_te)} S1 x {len(pool_te)} pool")
    cand_te = blocking_cached(s1_te, pool_te, "test")
    if args.mode == "lgbm":
        cand_te = predict_lgbm(cand_te, s1_te, pool_te, models, cols, tok)
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
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame({"s1": s1_ids[cand_te["i"].values], "x": pool_ids[cand_te["j"].values],
                  "i": cand_te["i"].values, "j": cand_te["j"].values,
                  "score": cand_te["score"].values}).to_parquet(
        os.path.join(args.out_dir, "test_scores.parquet"), index=False)
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
