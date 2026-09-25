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
# Chosen on the val train-part (30k S1): combo@20 + name_c4@10 -> pair recall 0.967,
# oracle 0.988, 26.6 cand/S1. Other spaces (name_bi, name_tok, addr_tok) add <=0.002.
BLOCKING = {
    "combo":     dict(space="combo",     k=20, max_df=20000),
    "name_c4":   dict(space="name_c4",   k=10, max_df=5000),
}

# ---------------------------------------------------------------- tuning
TUNE_S1 = 300_000             # train S1 sampled for threshold tuning (full train pool kept)

# ---------------------------------------------------------------- model
MODEL_S1 = 400_000
USE_XTOK = True               # learned extra-name-token encoding feature            # train S1 records used to fit the pair model (sampled, seeded)
INFER_CHUNK_S1 = 250_000      # S1 records per feature/predict chunk at inference
N_FOLDS = 5
LGB_ROUNDS = 3000
LGB_EARLY_STOP = 50
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED,
                  num_threads=N_JOBS, deterministic=True, force_row_wise=True)
