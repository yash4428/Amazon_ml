"""Apply our label-free France rules (word-swap, optional in-place type-swap) to ANY matching_results.tsv.
Only removes pairs, only in countries where the rule's switch fires (same statistics as postprocess.py)."""
import os, sys, csv, argparse, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import numpy as np, pandas as pd
from normalize import load_normalized
from postprocess import word_swap, type_swap_flags, street_number, _TYPE_IGNORE
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--type-swap", action="store_true"); ap.add_argument("--house-rule", action="store_true"); ap.add_argument("--societe-type", action="store_true", help="societe/cie/ste count as type words in the type-swap rule")
ap.add_argument("--cand", default="", help="candidate_pairs.tsv to copy/extend (matches must stay a subset)")
a = ap.parse_args()
def rd(p): return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", "test"), "test", n_jobs=8)
nm1 = dict(zip(s1.entity_id, s1.name_norm)); nmp = dict(zip(pool.entity_id, pool.name_norm)); cty = dict(zip(s1.entity_id, s1.country))
M = rd(a.inp); M.columns = ["source1_entity_id", "matched_entity_ids"]
freq = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
rows = [(s, x) for s, ms in zip(M.source1_entity_id, M.matched_entity_ids) for x in ms.split(",") if x]
P = pd.DataFrame(rows, columns=["s", "x"]); P["c"] = P.s.map(cty)
sw = [word_swap(nm1[s], nmp[x]) for s, x in zip(P.s, P.x)]
P["swap"] = [w is not None and freq[w[0]] >= 200 and freq[w[1]] >= 200 for w in sw]
n = s1.country.value_counts(); drop = np.zeros(len(P), bool)
if a.house_rule:
    h1d = dict(zip(s1.entity_id, s1.house_numbers)); h2d = dict(zip(pool.entity_id, pool.house_numbers))
    a1d = dict(zip(s1.entity_id, s1.business_address)); a2d = dict(zip(pool.entity_id, pool.business_address))
    h1 = np.array([h1d[s].split(" ", 1)[0] for s in P.s]); h2 = np.array([h2d[x].split(" ", 1)[0] for x in P.x])
    both = (h1 != "") & (h2 != "")
    diff = both & (np.char.lstrip(h1.astype(str), "0") != np.char.lstrip(h2.astype(str), "0"))
    dsub = diff & np.array([(x in y or y in x) for x, y in zip(h1, h2)])
    for c in n.index:
        m = (P.c == c).values; noise = 100 * dsub[m].sum() / n[c]
        print(f"  {c:7s} accepted digit-drop pairs per 100 S1 {noise:6.2f} -> house rule {'APPLIED' if noise < 3.0 else 'no'}")
        if noise < 3.0:
            idx = np.flatnonzero(m & diff)
            sn1 = np.array([street_number(a1d[s]) for s in P.s.values[idx]]); sn2 = np.array([street_number(a2d[x]) for x in P.x.values[idx]])
            keep = (sn1 != "") & (sn1 == sn2)
            print(f"          first-number mismatches {len(idx)}, kept (equal street number) {int(keep.sum())}")
            drop[idx[~keep]] = True
for c in n.index:
    m = (P.c == c).values; rate = 100 * P.swap.values[m].sum() / n[c]
    print(f"  {c:7s} frequent-word swaps per 100 S1 {rate:6.2f} -> {'APPLIED' if rate >= 1.0 else 'no'}")
    if rate >= 1.0: drop |= m & P.swap.values
if a.type_swap:
    for c in n.index:
        ms, mp = s1.country.values == c, pool.country.values == c
        fs = collections.Counter(t for x in s1.name_norm.values[ms] for t in set(x.split()))
        fp = collections.Counter(t for x in pool.name_norm.values[mp] for t in set(x.split()))
        types = {w for w, f in fs.items() if f >= 20 and (fp[w] / mp.sum()) / ((f + 1) / ms.sum()) < 1.3}
        m = np.flatnonzero((P.c == c).values & ~drop)
        ts = type_swap_flags([nm1[s] for s in P.s.values[m]], [nmp[x] for x in P.x.values[m]], types, ignore=(_TYPE_IGNORE - {'societe', 'cie', 'ste'}) if a.societe_type else None)
        rate = 100 * ts.sum() / n[c]
        print(f"  {c:7s} in-place type swaps per 100 S1 {rate:6.2f} -> {'APPLIED' if rate >= 1.0 else 'no'}")
        if rate >= 1.0: drop[m[ts]] = True
print(f"  dropped {int(drop.sum())} of {len(P)} pairs")
K = P[~drop].groupby("s").x.apply(lambda v: ",".join(v))
os.makedirs(a.out, exist_ok=True)
out = M[["source1_entity_id"]].copy(); out["matched_entity_ids"] = out.source1_entity_id.map(K).fillna("")
out.to_csv(os.path.join(a.out, "matching_results.tsv"), sep="\t", index=False, quoting=csv.QUOTE_NONE)
if a.cand:
    import shutil; shutil.copy(a.cand, os.path.join(a.out, "candidate_pairs.tsv"))
