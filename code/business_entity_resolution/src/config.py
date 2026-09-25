"""All tunable numbers, paths and flags for the pipeline, in one place."""
import os

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 2)

# Root of student_resource/ (src -> business_entity_resolution -> code -> root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE_DIR = os.path.join(ROOT, "cache")

# ---------------------------------------------------------------- blocking
# Each generator: feature space, top-K per S1, max document frequency (within
# one country, over S1+pool) for a feature to be used for retrieval.
BLOCK_CHUNK = 2000            # S1 rows per sparse-matmul chunk
# exp05 (with learned translit): combo_c@25 + combo@15 -> recall 0.9813, oracle 0.9938,
# 30.0 cand/S1 (was combo@20 + name_c4@10: 0.9735 @ 26.6). See PROGRESS.md table.
BLOCKING = {
    "combo_c":   dict(space="combo_c",   k=60, max_df=20000),
    "combo":     dict(space="combo",     k=30, max_df=20000),
}
# Stage-1 filter after wide blocking (see stage1.py). 30k-S1 sample: 10.0 cand/S1,
# recall 0.9842, oracle 0.9952 (narrow blocking was 30 cand/S1, recall 0.9814).
STAGE1 = dict(min_p=0.001, max_keep=20, rounds=250, train_s1=60_000)

# ---------------------------------------------------------------- tuning
TUNE_S1 = 300_000             # train S1 sampled for threshold tuning (full train pool kept)

# ---------------------------------------------------------------- model
MODEL_S1 = 300_000            # train S1 records used to fit the pair model (sampled, seeded)
USE_XTOK = True               # learned extra-name-token encoding feature
USE_TRANSLIT_DICT = True      # learned Indic-transliteration -> English token map (train pairs)
DROP_FEATURES = ["addr_len_a", "addr_len_b", "name_len_ratio"]  # country-specific (unseen-country sim +0.0018)
NORM_VERSION = 2              # bump when normalisation changes (invalidates blocking cache)
INFER_CHUNK_S1 = 250_000      # S1 records per feature/predict chunk at inference
N_FOLDS = 5
LGB_ROUNDS = 3000
LGB_EARLY_STOP = 50
LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED,
                  num_threads=N_JOBS, deterministic=True, force_row_wise=True)
