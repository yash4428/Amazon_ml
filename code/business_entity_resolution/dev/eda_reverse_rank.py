"""For OOF true pairs lost at blocking/stage-1: forward rank (x among S1's pool list) and reverse rank
(S1 among x's S1 list) in combo_c and combo spaces (non-crowded train pool)."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)

import sys, time, numpy as np, pandas as pd

import config
from normalize import load_normalized, build_translit_dict, apply_translit_dict
from io_utils import read_truth
from blocking import SPACES, build_matrix
S = _OUT
d = config.ROOT + "/dataset/train"
s1, pool = load_normalized(d, "train", n_jobs=config.N_JOBS); truth = read_truth(d)
tl = build_translit_dict(s1, pool, truth); s1, pool = apply_translit_dict(s1, tl), apply_translit_dict(pool, tl)
fn = pd.read_parquet(S + "fn.parquet")
bm = fn[fn.why == "blocking"].copy()
# also a control sample of true pairs that WERE found
owner = [(s, x) for s, m in truth.items() for x in m]
rng = np.random.RandomState(0)
ctrl = pd.DataFrame([owner[k] for k in rng.choice(len(owner), 30000, replace=False)], columns=["s1", "x"])
s1pos = pd.Series(np.arange(len(s1)), index=s1.entity_id); ppos = pd.Series(np.arange(len(pool)), index=pool.entity_id)
out = []
for tag, df in [("missed", bm), ("random_true", ctrl)]:
    df = df.copy(); df["i"] = s1pos.reindex(df.s1).values; df["j"] = ppos.reindex(df.x).values
    df = df.dropna(subset=["i", "j"]); df["i"] = df.i.astype(int); df["j"] = df.j.astype(int)
    df["cty"] = s1.country.values[df.i]
    for c in ["India", "US"]:
        si = np.flatnonzero(s1.country.values == c); pj = np.flatnonzero(pool.country.values == c)
        li = pd.Series(np.arange(len(si)), index=si); lj = pd.Series(np.arange(len(pj)), index=pj)
        sub = df[df.cty == c]
        for g in ["combo_c", "combo"]:
            t0 = time.time()
            an, cols = SPACES[config.BLOCKING[g]["space"]]
            docs = list(zip(*[pd.concat([s1[cc].iloc[si], pool[cc].iloc[pj]]).tolist() for cc in cols]))
            Sm, Pm = build_matrix(docs, an, config.BLOCKING[g]["max_df"], len(si))
            a = li.reindex(sub.i).values; b = lj.reindex(sub.j).values
            PT = Pm.T.tocsr(); ST = Sm.T.tocsr()
            fr, rr = np.empty(len(sub)), np.empty(len(sub))
            for k0 in range(0, len(sub), 2000):
                A = a[k0:k0 + 2000]; B = b[k0:k0 + 2000]
                F = (Sm[A] @ PT).tocsr(); Rv = (Pm[B] @ ST).tocsr()
                for r in range(len(A)):
                    row = F.getrow(r); v = row.toarray().ravel()[B[r]] if False else None
                    fa, fb = F.indptr[r], F.indptr[r + 1]; ind, dat = F.indices[fa:fb], F.data[fa:fb]
                    sc = dat[ind == B[r]]; sc = sc[0] if len(sc) else 0.0
                    fr[k0 + r] = (dat > sc).sum() if sc > 0 else 9999
                    ra, rb = Rv.indptr[r], Rv.indptr[r + 1]; ind2, dat2 = Rv.indices[ra:rb], Rv.data[ra:rb]
                    rr[k0 + r] = (dat2 > sc).sum() if sc > 0 else 9999
            o = sub[["s1", "x"]].copy(); o["space"] = g; o["fwd"] = fr; o["rev"] = rr; o["set"] = tag; o["cty"] = c
            out.append(o)
            print(f"{tag} {c} {g}: {len(sub)} pairs {time.time()-t0:.0f}s", flush=True)
o = pd.concat(out); o.to_parquet(S + "rev_measure.parquet")
for tag in ["missed", "random_true"]:
    w = o[o.set == tag].pivot_table(index=["s1", "x", "cty"], columns="space", values=["fwd", "rev"]).reset_index()
    w.columns = ["_".join(x).strip("_") for x in w.columns]
    inf = (w.fwd_combo_c < 60) | (w.fwd_combo < 30)
    print(f"\n== {tag}: n={len(w)}  already in fwd top-K (i.e. lost at stage-1): {inf.mean():.3f}")
    nf = w[~inf]
    for k in [1, 2, 3, 5, 10, 20]:
        hit = (nf.rev_combo_c < k) | (nf.rev_combo < k)
        print(f"   not in fwd top-K: reverse rank < {k:2d} in either space: {hit.mean():.3f}  ({int(hit.sum())} pairs)")
    print("   zero similarity (no shared feature) in both:", float(((nf.fwd_combo_c == 9999) & (nf.fwd_combo == 9999)).mean()))
