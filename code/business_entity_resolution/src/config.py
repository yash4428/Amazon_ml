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
