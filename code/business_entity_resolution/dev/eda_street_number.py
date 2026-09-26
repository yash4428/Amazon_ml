"""EDA: robust STREET number (number directly before street words, ignoring unit/apt/suite/floor/# numbers)
vs 'first number'. How cleanly does each separate true copies from fake branches? (train OOF exp21)"""
import os, sys, json, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from io_utils import read_source
UNIT = r"(?:unit|apt|apartment|ste|suite|fl|floor|flr|room|rm|pmb|po box|box|bldg|building|#)"
STREET_NUM = re.compile(r"(?<![\w/-])(\d+)(?:\s*(?:-|/)\s*\d+)?[a-z]?\s+(?!(?:st|nd|rd|th)\b)[a-z]")
def norm(t):
    t = unicodedata.normalize("NFKD", t); t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return t.replace("##", "#")
def street_number(a):
    for comp in norm(a).split(","):
        comp = comp.strip()
        if re.match(UNIT + r"\b", comp) or re.match(r"#\s*\d+\s*$", comp):
            continue
        comp = re.sub(r"^(?:door|h\.?|house|plot|flat|shop|office|n°|no\.?|nº|#)\s*(?:no\.?)?\s*", "", comp)
        m = STREET_NUM.search(comp)
        if m:
            return m.group(1).lstrip("0") or "0"
    return ""
def first_number(a):
    m = re.search(r"\d+", a); return (m.group().lstrip("0") or "0") if m else ""
R = "runs/exp21_test/report"; c = pd.read_parquet(f"{R}/oof_pairs.parquet"); c = c[~c.x.str.startswith("SY")]
c = c[c.score >= 0.05].sample(1_000_000, random_state=0)
s1 = read_source("dataset/train", "train", 1).set_index("entity_id")
pool = pd.concat([read_source("dataset/train", "train", 2), read_source("dataset/train", "train", 3)]).set_index("entity_id")
a1 = s1.loc[c.s1].business_address.values; a2 = pool.loc[c.x].business_address.values
cty = s1.loc[c.s1].country.values
for lab, fn in (("FIRST number", first_number), ("STREET number", street_number)):
    n1 = np.array([fn(x) for x in a1]); n2 = np.array([fn(x) for x in a2])
    both = (n1 != "") & (n2 != "")
    eq = both & (n1 == n2)
    d = np.where(both, [abs(int(p[:9]) - int(q[:9])) if p and q else -1 for p, q in zip(n1, n2)], -1)
    small = both & (d > 0) & (d <= 30); other = both & (d > 30)
    print(f"\n{lab}: coverage (both present) {both.mean():.3f}")
    for k, m in (("equal", eq), ("shift 1-30", small), ("differ >30", other), ("missing", ~both)):
        for cc in ("India", "US"):
            mm = m & (cty == cc)
            print(f"   {cc:6s} {k:12s} share {mm.sum() / (cty == cc).sum():.3f}  P(true) {c.y.values[mm].mean():.3f}")
