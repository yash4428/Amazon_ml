# day1_2 — exp04 LightGBM
- code: commit fd7ab3e (exp04), run dir runs/exp04_test
- LightGBM 64 feats (string, house-number cluster, learned branch-word encoding, name-dup, blocking context), 250k train S1, 5-fold GroupKFold, final single model 686 rounds
- learned transliteration dict (567 tokens, from train pairs)
- blocking combo@20 + name_c4@10 (26.5 cand/S1)
- decision: one-to-one, t1=0.74, t2=0.02, r=0.75
- honest OOF (full train): 0.9736 (India 0.9662, US 0.9785)
- validator PASS, sanity PASS; empty 0.0575 (France 0.047, India 0.062, US 0.056)
