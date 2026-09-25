# day2_exp10_crowd
- source run: runs/exp10_test (commit f960083)
- exp10 = exp09 + crowd 0.5 training; crowded OOF 0.9789 (IN .9756 US .9811); t1=.72 t2=.70; 11.9 cand/S1
- Chosen over exp09 (submissions/day2_exp09_wide) because: crowd A/B on test-like val +0.0031; on test France,
  exp10 drops 6k branch-like pairs (house diff 58%, branch_sig 44%) and adds 31k same-address pairs
  (same house 76%) that look like true copies with a swapped generic word (98% positive in train).
