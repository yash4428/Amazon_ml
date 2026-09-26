"""Where do true pairs get lost before the model? (train, exp17 run: wide blocking + stage-1, crowded)
Compares the sampled S1's full truth with (a) the pairs that reached the model (oof_pairs)."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from io_utils import read_source, read_truth
R = "runs/exp17_test/report"; o = pd.read_parquet(f"{R}/oof_pairs.parquet", columns=["s1", "x", "y"]); s = pd.read_parquet(f"{R}/oof_s1.parquet")
truth = read_truth("dataset/train"); got = set(zip(o.s1[o.y], o.x[o.y]))
miss = [(a, x) for a in s.s1 for x in truth.get(a, ()) if (a, x) not in got]
tot = sum(len(truth.get(a, ())) for a in s.s1)
print(f"true pairs of the sampled S1: {tot}; reached the model: {len(got)} ({len(got)/tot:.4f}); lost before the model: {len(miss)} ({len(miss)/tot:.4f})")
s1 = read_source("dataset/train", "train", 1).set_index("entity_id")
pool = pd.concat([read_source("dataset/train", "train", 2), read_source("dataset/train", "train", 3)]).set_index("entity_id")
m = pd.DataFrame(miss, columns=["s1", "x"])
a1 = s1.loc[m.s1]; b = pool.loc[m.x]
m["cty"] = a1.country.values
m["empty_addr"] = (b.business_address.values == "")
m["native"] = [bool(re.search(r"[ऀ-ൿ]", t)) for t in b.business_name.values]
m["domain"] = [bool(re.search(r"\.com|^@|^#|com$", t, re.I)) for t in b.business_name.values]
lw = lambda xs: [x.lower() for x in xs]
m["name_ts"] = process.cpdist(lw(a1.business_name.values), lw(b.business_name.values), scorer=fuzz.token_set_ratio, workers=-1)
m["addr_ts"] = process.cpdist(lw(a1.business_address.values), lw(b.business_address.values), scorer=fuzz.token_set_ratio, workers=-1)
cat = np.select([m.empty_addr & (m.name_ts >= 90), m.empty_addr, m.native, m.domain,
                 (m.name_ts >= 80) & (m.addr_ts >= 80), (m.name_ts < 60) & (m.addr_ts >= 80), (m.name_ts >= 80) & (m.addr_ts < 60)],
                ["empty addr, name≥90 (tie-type)", "empty addr, name<90", "native-script name", "domain/handle",
                 "name≥80 & addr≥80 (easy!)", "name<60, addr≥80 (random name)", "name≥80, addr<60"], "both mid/low")
m["cat"] = cat
print((m.groupby(["cat", "cty"]).size().unstack(fill_value=0) / tot * 100).round(3).assign(total=lambda d: d.sum(axis=1)).sort_values("total", ascending=False).to_string())
print("\n(values = % of ALL true pairs of the sampled S1)")
for k in ["name≥80 & addr≥80 (easy!)", "both mid/low", "name<60, addr≥80 (random name)"]:
    print(f"\n== {k}")
    for r in m[m.cat == k].sample(6, random_state=1).itertuples():
        print(f"   {s1.at[r.s1,'business_name']} | {s1.at[r.s1,'business_address']}\n      {pool.at[r.x,'business_name']} | {pool.at[r.x,'business_address']}")
