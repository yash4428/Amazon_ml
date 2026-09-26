"""Blind EDA: do distractors (pool records owned by NO S1) carry formatting fingerprints vs true copies?
Record-level features from the RAW strings only (no S1 information). Train labels: owned vs not owned."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from io_utils import read_source, read_truth

def feats(df):
    n = df.business_name.fillna(""); a = df.business_address.fillna("")
    f = pd.DataFrame(index=df.index)
    f["src"] = df.src.values
    f["n_len"] = n.str.len(); f["a_len"] = a.str.len()
    f["n_upper"] = (n == n.str.upper()) & n.str.contains("[A-Z]")
    f["a_upper"] = (a == a.str.upper()) & a.str.contains("[A-Z]")
    f["n_lower"] = (n == n.str.lower()) & n.str.contains("[a-z]")
    f["n_title"] = n.str.istitle()
    f["a_commas"] = a.str.count(","); f["n_words"] = n.str.split().str.len()
    f["a_words"] = a.str.split().str.len()
    f["n_dspace"] = n.str.contains("  "); f["a_dspace"] = a.str.contains("  ")
    f["n_trail_sp"] = n.str.endswith(" ") | n.str.startswith(" ")
    for k, pat in {"a_hash": r"#", "a_dhash": r"##", "a_null": r"NULL", "a_unit": r"\bUnit\b|\bUNIT\b",
                   "a_apt": r"\bApt\b|\bAPT\b", "a_pmb": r"PMB|PO BOX|Po Box", "a_zeropad": r"\b0\d{2,}",
                   "a_range": r"\d+-\d+", "a_digit_first": r"^\s*[#(]?\d", "a_state_code_end": r", [A-Z]{2}$",
                   "n_digit": r"\d", "n_paren": r"[()\[\]]", "n_amp": r"&", "n_dot": r"\.",
                   "n_comma": r",", "n_dash": r"-", "n_domain": r"\.com|\.in\b|^@|^#", "a_lower_word": r"\b[a-z]{3,}\b",
                   "a_nonascii": r"[^\x00-\x7f]", "n_nonascii": r"[^\x00-\x7f]", "n_accent": r"[À-ÿ]"}.items():
        f[k] = a.str.contains(pat) if k.startswith("a_") else n.str.contains(pat)
    f["a_empty"] = a.str.strip() == ""
    f["n_first_legal"] = n.str.match(r"(?i)^(llc|inc|ltd|pvt|private|corp|the|sarl|sas|eurl)\b")
    f["n_last_word_len"] = n.str.split().str[-1].str.len().fillna(0)
    return f.astype(float)

s1 = read_source("dataset/train", "train", 1)
pool = pd.concat([read_source("dataset/train", "train", 2), read_source("dataset/train", "train", 3)], ignore_index=True)
truth = read_truth("dataset/train"); owned = {x for v in truth.values() for x in v}
pool["y"] = pool.entity_id.isin(owned).astype(int)            # 1 = true copy of some S1, 0 = distractor
smp = pool.sample(1_500_000, random_state=0)
X = feats(smp); X["country_us"] = (smp.country == "US").astype(float)
y = smp.y.values; fold = np.random.RandomState(1).randint(0, 2, len(y)); pred = np.zeros(len(y)); imp = np.zeros(X.shape[1])
for k in (0, 1):
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=8), lgb.Dataset(X[fold != k], y[fold != k]), 300)
    pred[fold == k] = m.predict(X[fold == k]); imp += m.feature_importance("gain")
print(f"owned-vs-distractor from RAW RECORD FORMAT only: AUC {roc_auc_score(y, pred):.4f}  (base rate owned {y.mean():.3f})")
top = pd.Series(imp, index=X.columns).sort_values(ascending=False).head(15)
for f_, g in top.items():
    col = X[f_].values
    print(f"  {f_:18s} gain {g:12.0f}  mean owned {col[y==1].mean():8.3f}  mean distractor {col[y==0].mean():8.3f}")
# per source & country breakdown of the strongest binary signals
print("\nby source (share of records with feature, owned vs distractor):")
for s in (2, 3):
    mm = smp.src.values == s
    for f_ in ["a_upper", "n_upper", "a_state_code_end", "a_unit", "a_hash", "a_null", "a_zeropad", "a_empty", "n_domain"]:
        print(f"  S{s} {f_:18s} owned {X[f_].values[mm & (y==1)].mean():.3f}  distractor {X[f_].values[mm & (y==0)].mean():.3f}")
