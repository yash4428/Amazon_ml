# day2_A — exp17 predictions for US + India, exp20 predictions for France (identical candidate sets)
Designed for exact attribution: US/India rows are identical to the exp17 France-probe (public 0.836), so
  France F(exp20) = (A − 0.836)/0.1498 + 0.05  (s_France ≈ 0.05)
exp20 = exp17 − learned branch-word features (blind in France) + capped size-dependent counts.
France: accepted pairs adding a French branch word 23,364 (exp17) → 839 (exp20); France matches/S1 3.447 → 3.244.
Validator PASS, sanity PASS.
