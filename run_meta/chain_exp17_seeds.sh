#!/bin/zsh
# exp18/exp19 = exp17 features on other 800k samples (seeds 4, 5) for a blend; start after exp17 finishes.
cd /Users/yashaggarwal/Desktop/student_resource
until grep -qE "wrote|Traceback" runs/exp17_test.log; do sleep 20; done
for s in 4 5; do
  n=$((14 + s))
  mkdir -p runs/exp${n}_test
  .venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test \
     --out-dir runs/exp${n}_test --crowd 0.5 --sample-seed $s --model-s1 800000 > runs/exp${n}_test.log 2>&1
done
