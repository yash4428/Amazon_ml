# day1_5 final — blend 0.5*exp06 + 0.5*exp07
- exp06: exp05 + branch-signature (house nudge) feats, minus length feats, 300k S1, 5-fold CV, OOF 0.9771
- exp07: same features, 600k different S1 sample (seed+1), no CV, 750 rounds
- decision params from exp06 OOF: t1=0.76 t2=0.74 r=0, one-to-one
- validator PASS, sanity PASS
