# day1_1 — rule baseline (exp01)
- commit: 722dee8
- score = max(combo_sim, 0.8*name_c4_sim); one-to-one; t1=0.58 t2=0.82 r=0 (tuned on 300k train S1)
- local: val 0.7730 | loco_india 0.6943 | loco_us 0.6951 | val oracle 0.9934
- validator PASS, sanity PASS; empty rate 0.024 (France 0.020, India 0.025, US 0.024)
