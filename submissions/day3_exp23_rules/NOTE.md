# day3_exp23_rules — exp23 + robust France house rule + US shift rule
exp23 = exp22 (reverse top-5 blocking) trained with crowd 0.9 (test-like distractor density). OOF 0.9838 (on the
crowd-0.9 pool; India 0.9818, US 0.9852). Rebuild: postprocess.py --run runs/exp23_test --out <dir> --house-rule --shift-rule US
Validator PASS, sanity PASS (empty FR 5.56% / IN 5.65% / US 5.82%).
