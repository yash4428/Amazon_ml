#!/bin/zsh
# After exp10: exp11 = exp10 config, different train-S1 sample (seed+1), 400k S1, crowd 0.5.
cd /Users/yashaggarwal/Desktop/student_resource
until grep -qE "wrote|Traceback" runs/exp10_test.log; do sleep 20; done
echo "[$(date +%H:%M)] exp10 finished; starting exp11" >> runs/night_chain.log
mkdir -p runs/exp11_test
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test \
  --out-dir runs/exp11_test --crowd 0.5 --sample-seed 1 --model-s1 400000 > runs/exp11_test.log 2>&1
echo "[$(date +%H:%M)] exp11 finished" >> runs/night_chain.log
