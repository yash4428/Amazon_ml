#!/bin/bash
# after exp23: exp22 config with sample seeds 4 and 5 (for a 3-seed blend); blocking caches are reused
cd /Users/yashaggarwal/Desktop/student_resource
while kill -0 86537 2>/dev/null; do sleep 30; done
for s in 4 5; do
  caffeinate -i .venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir runs/exp22s${s}_test --crowd 0.5 --sample-seed $s --model-s1 800000 --profile exp17 --rev-k 5 > runs/exp22s${s}_test.log 2>&1
done
