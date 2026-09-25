# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + LightGBM pair classifier + a tuned decision step. Runs fully offline:
no network calls, no external data, no pretrained models.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce the submission (run from `student_resource/`)

```bash
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py \
    --train-dir dataset/train --test-dir dataset/test --out-dir output --crowd 0.5 --model-s1 800000
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
.venv/bin/python code/business_entity_resolution/src/sanity_check.py --out-dir output --test-dir dataset/test
```

This writes `output/matching_results.tsv`, `output/candidate_pairs.tsv` and `output/run_info.json`.
Intermediate results are cached in `cache/` (normalised text, blocking candidates); delete it for a cold run.

## Validation

`evaluate.py make-split` writes `splits/<name>/{train,test}`, and the same command runs on them:

```bash
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py \
    --train-dir splits/val/train --test-dir splits/val/test --out-dir runs/val --mode lgbm
python3 evaluate.py score --split val --pred runs/val/matching_results.tsv --cand runs/val/candidate_pairs.tsv --tag expNN
```

## Code layout (`src/`)

| file | role |
|---|---|
| `config.py` | every threshold, K, seed and path |
| `io_utils.py` | safe TSV I/O (tab separator, strings only, empty kept) and output writers |
| `normalize.py` | name/address normalisation, Indic-script transliteration, parsed pieces (postcode, house numbers) |
| `blocking.py` | per-country sparse IDF-cosine top-K candidate generation (wide: combo_c@60 + combo@30) |
| `stage1.py` | cheap LightGBM filter after blocking: ~70 → ~12 candidates per S1 at higher recall; its output is `candidate_pairs.tsv` |
| `features.py` | string, address, house-number/branch-signature, learned & unsupervised branch-word, name-duplicate, blocking and context (competition) features; `add_synthetic_branches` (train-time crowd augmentation) |
| `model.py` | LightGBM, 5-fold GroupKFold by S1, OOF predictions |
| `decide.py` | one-to-one assignment, t1/t2/r thresholds tuned for macro F0.5 |
| `run_pipeline.py` | CLI entry point |
| `sanity_check.py` | submission checks (row counts, per-country empty rate, matches ⊆ candidates) |

## Hardware / runtime

Developed on an Apple M4 Pro (12 cores, 24 GB RAM, no GPU). Runtime figures will be filled in for the final configuration.

## Development tools (`dev/`)

| script | purpose |
|---|---|
| `block_dev.py` | blocking recall / candidates / oracle table on a train sample (`--crowd` simulates denser distractors) |
| `stage1_dev2.py` | stage-1 filter recall vs candidates kept |
| `country_cv.py`, `country_params.py` | unseen-country simulator (train on one country, predict the other) |
| `crowd_eval.py` | A/B: normal vs crowd-trained model on normal / crowded validation (`--country-folds` for unseen country) |
| `blend.py` | evaluate a blend of two runs on their common OOF S1 and write a blended submission |
| `package.sh` | validator + sanity check + copy a run to `submissions/<name>/` |

See `PROGRESS.md` at the repository root for the full experiment history and findings.
