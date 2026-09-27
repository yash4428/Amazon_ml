# day3_exp23_rules_swapFR — day3_exp23_rules + France word-swap rule (France-only change)
Drops accepted pairs whose names differ by replacing one frequent word with another frequent word
("Campagne Comite SAS" -> "Campagne Sportive SAS"); label-free switch: France 10.29 per 100 S1, US 0.03, India 0.05
(threshold 1.0). Train: such candidates are almost never true; the model accepts 0.09 per 100 S1 in train vs 8.84 in France.
21,683 French rows differ from day3_exp23_rules (1,212 become empty). ΔFrance = (S - S_day3_exp23_rules)/0.1498.
Rebuild: postprocess.py --run runs/exp23_test --out <dir> --house-rule --shift-rule US --swap-rule
