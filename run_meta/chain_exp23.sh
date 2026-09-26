#!/bin/bash
# wait for exp22, then run exp23 = exp22 + crowd 0.9 (test has ~2x train distractors per S1)
cd /Users/yashaggarwal/Desktop/student_resource
while kill -0 84051 2>/dev/null; do sleep 30; done
caffeinate -i .venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir runs/exp23_test --crowd 0.9 --sample-seed 3 --model-s1 800000 --profile exp17 --rev-k 5 > runs/exp23_test.log 2>&1
