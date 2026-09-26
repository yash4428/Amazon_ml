#!/bin/zsh
cd /Users/yashaggarwal/Desktop/student_resource
until grep -qE "wrote|Traceback" runs/exp20_test.log; do sleep 20; done
mkdir -p runs/exp21_test
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test \
  --out-dir runs/exp21_test --crowd 0.5 --sample-seed 3 --model-s1 800000 > runs/exp21_test.log 2>&1
