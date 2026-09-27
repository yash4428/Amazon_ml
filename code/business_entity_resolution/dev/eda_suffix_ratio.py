"""Per word (one country): share of POOL names containing it / share of S1 names containing it. Noise suffixes the
generator adds to copies (fils, cie, associes, ...) have ratio >> 1; business-type words inherited from the S1
(club, ecole, bar, immobiliere, ...) have ratio ~1. Label-free; used to define type-word swaps."""
import os, sys, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
import pandas as pd
from normalize import load_normalized
cty = sys.argv[1] if len(sys.argv) > 1 else "France"
split = "test" if cty == "France" else "train"
s1, pool = load_normalized(os.path.join(_ROOT, "dataset", split), split, n_jobs=8)
s1, pool = s1[s1.country == cty], pool[pool.country == cty]
fs = collections.Counter(t for n in s1.name_norm.values for t in set(n.split()))
fp = collections.Counter(t for n in pool.name_norm.values for t in set(n.split()))
d = pd.DataFrame({"s1": pd.Series(fs), "pool": pd.Series(fp)}).fillna(0)
d["ratio"] = (d.pool / len(pool)) / ((d.s1 + 1) / len(s1))
d = d[(d.s1 >= 20) | (d.pool >= 200)]
pd.set_option("display.width", 200)
print(f"{cty}: {len(s1)} S1, {len(pool)} pool")
print("HIGH ratio (added to copies):", d.sort_values("ratio", ascending=False).head(30).round(2).ratio.to_dict())
print("frequent words ratio:", d[d.s1 >= 500].sort_values("s1", ascending=False).head(60).round(2).ratio.to_dict())
d.to_csv(os.path.join(_ROOT, "runs", "eda", f"suffix_ratio_{cty}.csv"))
