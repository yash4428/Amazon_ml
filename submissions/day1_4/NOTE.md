# day1_4 — exp05 LightGBM
- blocking combo_c@25 + combo@15 (train pair recall 0.9756; 29.5 cand/S1 on test)
- + unsupervised house-mismatch branch-word feature (works for France), 300k train S1
- honest OOF (full train): 0.9769 (India 0.9716, US 0.9805); unseen-country sim ~0.961
- decision: one-to-one, t1=0.78 t2=0.72 r=0
- validator PASS, sanity PASS; empty 0.0585 (France 0.054, India 0.062, US 0.057)
