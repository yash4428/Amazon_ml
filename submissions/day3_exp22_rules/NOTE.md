# day3_exp22_rules — exp22 (reverse top-5 blocking) + robust France house rule + US shift rule
Model exp22: exp17 settings (--profile exp17, crowd 0.5, seed 3, 800k S1) + --rev-k 5. OOF 0.9833 (exp17 0.98146).
Rebuild: postprocess.py --run runs/exp22_test --out <dir> --house-rule --shift-rule US
Candidates 18.4M (exp17 20.6M). Validator PASS, sanity PASS (empty FR 5.54% / IN 5.70% / US 5.83%).
Compare with day2_C_us_shift (0.970125; same rules on exp17 minus the ~0.0001 France fix) -> public effect of reverse blocking.
