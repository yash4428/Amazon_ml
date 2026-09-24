# PROGRESS

## Now
- Current step: Step 1 — Environment + sanity (DONE, 25 Sep 00:35 IST)
- Next action: Step 2 — Stage 0 data profile (`evaluate.py check-data`, test profile incl. France, splits), then 🛑 report to Yash.
- Current best: none yet

## Environment
- Machine: Apple M4 Pro, 12 cores, 24 GB RAM, no NVIDIA GPU (Apple MPS only), ~140 GB free disk.
- Python 3.12 venv at `.venv/` (system python3 is 3.14; 3.12 chosen for reliable wheels). Run everything with `.venv/bin/python`.
- Libs (all permissive): pandas BSD, numpy BSD, scikit-learn BSD, scipy BSD, lightgbm MIT, rapidfuzz MIT, joblib BSD, tqdm MPL-2.0/MIT, pyarrow Apache-2.0. No unidecode.
- Pinned: `code/business_entity_resolution/requirements.txt`.
- `evaluate.py selftest` PASS.
- Git initialised; commits use `-c user.name="Yash Aggarwal"`.

## Key data facts (from Stage 0)
- Sizes (lines incl. header): train S1 2,206,821 / S2 5,034,616 / S3 5,285,603 / GT 2,206,821; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316. Dataset 2.4 GB.
- **Scale is large (~10M candidate-pool records on test).** Blocking must be chunked and memory-aware; full train set likely needs subsampling for model training.

## Decisions made (and why)
- Python 3.12 venv (3.14 too new for some wheels).
- No LLM reranker planned unless time allows: no CUDA GPU, and 1.7M S1 is large.

## Submissions (day, slot, tag, local val, public LB score)
- none

## Ideas backlog (ranked)
- (after Stage 0)

## Known issues
- Validator `--check-ids` loads all S2/S3 ids (a few GB) — fine on 24 GB.
