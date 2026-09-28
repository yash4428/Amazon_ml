#!/bin/bash
# Regenerate the best leaderboard file (public 0.989378) from
#   (1) the teammate model's output  code/business_entity_resolution/inputs/teammate_matching_results.tsv
#       (her pipeline's source code was lost; this file is its unchanged output, public 0.984833), and
#   (2) our three model runs (README step A: run_pipeline.py with --sample-seed 3/4/5).
# Usage (from student_resource/):  bash code/business_entity_resolution/dev/make_best_file.sh RUN1 RUN2 RUN3 [OUT_DIR]
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CODE="$ROOT/code/business_entity_resolution"
PY="${PY:-$ROOT/.venv/bin/python}"
R1=$1; R2=$2; R3=$3; OUT=${4:-$ROOT/output}
WORK="$ROOT/runs/best_work"; mkdir -p "$WORK" "$OUT"
TM="$CODE/inputs/teammate_matching_results.tsv"
# 1) our 3-seed blend + decision + our label-free rules (France house, US shift, France word-swap, France type-swap)
$PY "$CODE/src/postprocess.py" --run "$R1" "$R2" "$R3" --params "$CODE/final_params.json" --out "$WORK/ours" \
    --house-rule --shift-rule US --swap-rule --type-swap-rule
# 2) blended scores with house-number categories
$PY "$CODE/dev/make_blend_scores.py" "$R1" "$R2" "$R3" --out "$WORK/blend_scores_cat.parquet"
# 3) our France rules applied to the teammate output (house rule, word-swap, in-place type-swap incl. Societe/Cie/Ste)
$PY "$CODE/dev/apply_rules_to_file.py" "$TM" "$WORK/tm_rules" --house-rule --type-swap --societe-type
# 4) add our confident pairs it misses (blend >= 0.99, same / missing house number or empty address)
$PY "$CODE/dev/final_adds.py" "$WORK/tm_rules/matching_results.tsv" "$WORK/ours/matching_results.tsv" \
    --scores "$WORK/blend_scores_cat.parquet" --out "$OUT"
# 5) candidate file = final matches + teammate matches (the set the rule stage evaluated)
$PY "$CODE/dev/make_candidates.py" "$OUT/matching_results.tsv" "$TM" "$OUT/candidate_pairs.tsv"
echo "done: $OUT"
