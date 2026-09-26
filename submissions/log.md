# Submission log
| day/slot | tag | local val | loco_india | loco_us | public LB |
|---|---|---|---|---|---|
| day1_1 | exp01_rule_baseline | 0.7730 | 0.6943 | 0.6951 | 0.679 |
| day1_2 | exp04_lgbm | OOF 0.9736 | IN 0.9662 | US 0.9785 | 0.955 |
| day1_4 | exp05 | OOF 0.9769 | IN 0.9716 | US 0.9805 | 0.963 |
| day1_5 | blend exp06+exp07 | OOF exp06 0.9771 | - | - | 0.964 |
| day2_1 | day2_best (blend exp13/14/15) | OOF ~0.9804 | IN .9776 | US .9822 | 0.966 |
| day2_2 | exp17 France probe | - | - | - | 0.836 (US+IN = 0.9745) |
| day2_3 | Aprime = exp17 minus France house-diff pairs | - | - | - | **0.968745** (France ≈ 0.938) |
| day2_4 | C = Aprime minus US shifted-number pairs | - | - | - | **0.970125** (US +0.0036) |
| day2_5 | day2_final_v2 = 3-seed blend 17/18/19 + robust France house rule + US shift | blend OOF 0.98145 (common S1) | - | - | **0.970385** (+0.00026 vs C) |
