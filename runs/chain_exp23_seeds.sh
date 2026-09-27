#!/bin/bash
# exp23 config (crowd 0.9 + reverse top-5) with sample seeds 4 and 5 for the final 3-seed blend; blocking caches reused
cd /Users/yashaggarwal/Desktop/student_resource
for s in 4 5; do
  caffeinate -dims .venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir runs/exp23s${s}_test --crowd 0.9 --sample-seed $s --model-s1 800000 --profile exp17 --rev-k 5 > runs/exp23s${s}_test.log 2>&1
done
