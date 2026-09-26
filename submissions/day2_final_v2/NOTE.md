# day2_final_v2 — recommended upload for day2 slot 5
= 3-seed blend exp17/18/19 (same features, sample seeds 3/4/5; params runs/blend_17_18_19_params.json)
  + France house rule with the ROBUST street-number check (1,613 French pairs whose first numbers differ only because of
    postcode/appt/reordering artefacts are no longer dropped)
  + US shift rule (as day2_C).
Rebuild: postprocess.py --run runs/blend_17_18_19 --params runs/blend_17_18_19_params.json --house-rule --shift-rule US
Expected: C (0.970125) + blend (+0.0003-0.0004 on common OOF S1) + France fix (+0.0001) ≈ 0.9705.
Validator PASS, sanity PASS (France empty 5.75%, India 5.84%, US 5.89%).
