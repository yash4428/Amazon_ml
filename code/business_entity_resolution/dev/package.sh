#!/bin/zsh
# Validate + sanity-check a run's outputs and copy them to submissions/<name>/.
# Usage: code/business_entity_resolution/dev/package.sh runs/exp09_test day2_best "one-line note"
set -e
RUN=$1; NAME=$2; NOTE=${3:-""}
cd /Users/yashaggarwal/Desktop/student_resource
python3 utils/validate_submission.py --matching $RUN/matching_results.tsv --candidate $RUN/candidate_pairs.tsv --test-dir dataset/test | tail -1
.venv/bin/python code/business_entity_resolution/src/sanity_check.py --out-dir $RUN --test-dir dataset/test | tail -8
mkdir -p submissions/$NAME
cp $RUN/matching_results.tsv $RUN/candidate_pairs.tsv submissions/$NAME/
[ -f $RUN/run_info.json ] && cp $RUN/run_info.json submissions/$NAME/
[ -f $RUN/report/oof.json ] && cp $RUN/report/oof.json submissions/$NAME/
{ echo "# $NAME"; echo "- source run: $RUN (commit $(git rev-parse --short HEAD))"; echo "- $NOTE"; } > submissions/$NAME/NOTE.md
echo "packaged -> submissions/$NAME"
