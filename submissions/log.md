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
| day3_1 | day3_exp23_rules = exp23 (reverse top-5 + crowd 0.9) + robust house rule + US shift | OOF 0.9838 (crowd-0.9 pool) | - | - | **0.970601** (+0.00022 vs day2_final_v2, +0.00048 vs C) |
| day3_2 | day3_exp23_rules_swapFR = day3_1 + France word-swap rule (France-only) | - | - | - | **0.972832** (+0.002231 => France +0.0149) |
| day3_3 | teammate file (her pipeline, code lost) | - | - | - | **0.984833** |
| day3_4 | probe_tm_house_swap = teammate + our France house + word-swap rules (France-only) | - | - | - | **0.988587** (+0.003754 => France +0.0251) |
| day3_5 | probe_tm_house_swap_typeswap = day3_4 + in-place type-swap rule (France-only) | - | - | - | **0.989332** (+0.000745 => France +0.0050) |
| day3_6 | final1_A_plus_adds = day3_5 + societe type words + 8,605 of our >=0.99 same/no-number pairs | - | - | - | **0.989378** (+0.000046) |
| day3_7 | final2 = final1 - 11,099 low-score shifted US/India pairs (keep-only) | - | - | - | **0.988827** (-0.000551: those were mostly true copies) |
