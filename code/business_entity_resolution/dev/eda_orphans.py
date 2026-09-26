"""26 Sep night EDA (see PROGRESS.md §6c). Run from student_resource/."""
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_sys.path.insert(0, _os.path.join(_ROOT, "code", "business_entity_resolution", "src"))
_OUT = _os.path.join(_ROOT, "runs", "eda") + "/"; _os.makedirs(_OUT, exist_ok=True)
import sys; 
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from normalize import load_normalized
R = _ROOT + "/"
s1, pool = load_normalized(R + "dataset/test", "test", n_jobs=4)
c = pd.read_parquet(R + "cache/blocks/test_96f8b17bd85e.parquet", columns=["j"])
sel = np.zeros(len(pool), bool); sel[c.j.values] = True
pool["sel"] = sel
ns1 = s1.country.value_counts()
print("pool records in NO candidate list (final exp17 candidates) per S1:", (pool[~pool.sel].groupby("country").size() / ns1).round(3).to_dict())
print("  of which empty address:", (pool[~pool.sel & (pool.addr_norm == "")].groupby("country").size() / ns1).round(3).to_dict())
for cty in ["France", "US"]:
    orph = pool[(~pool.sel) & (pool.country == cty) & (pool.addr_norm != "")].sample(4000, random_state=0)
    ss = s1[s1.country == cty]
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, dtype=np.float32)
    X = v.fit_transform((ss.name_norm + " " + ss.addr_norm).tolist())
    Y = v.transform((orph.name_norm + " " + orph.addr_norm).tolist())
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)
    d, k = nn.kneighbors(Y)
    orph["nn_sim"] = 1 - d[:, 0]; orph["nn"] = k[:, 0]
    print(f"\n{cty}: orphan (non-empty addr) nearest-S1 char-tfidf sim quantiles:", np.quantile(orph.nn_sim, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3))
    for r in orph.sort_values("nn_sim", ascending=False).iloc[::200].head(20).itertuples():
        t = ss.iloc[r.nn]
        print(f"  {r.nn_sim:.2f} {r.business_name[:34]:34s} | {r.business_address[:42]:42s} ~ {t.business_name[:34]:34s} | {t.business_address[:42]}")
