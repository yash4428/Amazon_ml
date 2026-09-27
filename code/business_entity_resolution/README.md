# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + LightGBM pair classifier + a tuned decision step. Runs fully offline:
no network calls, no external data, no pretrained models.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce the outputs (run from `student_resource/`)

### A. Fully reproducible pipeline (public 0.972832 single model; 3-seed blend below)

```bash
# 1) three model runs (same settings, different train-S1 samples); ~2.5 h cold for the first, ~1 h each after
#    (blocking results are cached in cache/blocks/)
for s in 3 4 5; do
  .venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train \
      --test-dir dataset/test --out-dir runs/exp23_s$s --crowd 0.9 --sample-seed $s --model-s1 800000 \
      --profile exp17 --rev-k 5
done
# 2) 3-seed blend + decision + label-free post-rules -> output/
.venv/bin/python code/business_entity_resolution/src/postprocess.py --run runs/exp23_s3 runs/exp23_s4 runs/exp23_s5 \
    --params code/business_entity_resolution/final_params.json --out output \
    --house-rule --shift-rule US --swap-rule --type-swap-rule
# 3) checks
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
.venv/bin/python code/business_entity_resolution/src/sanity_check.py --out-dir output --test-dir dataset/test
```

### B. Best leaderboard file (0.989378)

Starts from a teammate model's output (`inputs/teammate_matching_results.tsv`; its code was lost) and applies our rules
and our confident extra pairs:

```bash
.venv/bin/python code/business_entity_resolution/dev/apply_rules_to_file.py inputs/teammate_matching_results.tsv \
    runs/best_rules --house-rule --type-swap --societe-type
# our 3-seed blend scores must exist (step A) -> runs/eda/blend_scores_cat.parquet via dev/eda_mimic.py
.venv/bin/python code/business_entity_resolution/dev/final_adds.py runs/best_rules/matching_results.tsv \
    <our step-A output>/matching_results.tsv --out output_best
```

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
| `blocking.py` | per-country sparse IDF-cosine top-K candidate generation (wide: combo_c@60 + combo@30) + optional reverse top-K per pool record (`--rev-k`) |
| `stage1.py` | cheap LightGBM filter after blocking: ~70 → ~12 candidates per S1 at higher recall; its output is `candidate_pairs.tsv` |
| `features.py` | string, address, house-number/branch-signature, learned & unsupervised branch-word, name-duplicate, blocking and context (competition) features; `add_synthetic_branches` (train-time crowd augmentation) |
| `model.py` | LightGBM, 5-fold GroupKFold by S1, OOF predictions |
| `decide.py` | one-to-one assignment, t1/t2/r thresholds tuned for macro F0.5 |
| `postprocess.py` | data-driven fake-branch rules on saved test scores (France house rule with street-number check, US shift rule) |
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
