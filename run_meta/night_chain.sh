#!/bin/zsh
# Overnight chain (26 Sep): wait for exp08's crowded train blocking cache, stop exp08,
# run exp09 (wide blocking + stage-1), then the crowd A/B test.
cd /Users/yashaggarwal/Desktop/student_resource
until grep -qE "house-mismatch token rates|Traceback" runs/exp08_test.log; do sleep 20; done
pkill -f "out-dir runs/exp08_test"; sleep 5
echo "[$(date +%H:%M)] exp08 stopped after caching crowded blocking; starting exp09" >> runs/night_chain.log
mkdir -p runs/exp09_test
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir runs/exp09_test > runs/exp09_test.log 2>&1
echo "[$(date +%H:%M)] exp09 finished; starting crowd_eval" >> runs/night_chain.log
.venv/bin/python code/business_entity_resolution/dev/crowd_eval.py --model-s1 120000 --crowd 0.5 > runs/crowd_eval.log 2>&1
echo "[$(date +%H:%M)] crowd_eval finished" >> runs/night_chain.log
