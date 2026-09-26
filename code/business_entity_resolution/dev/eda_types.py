"""EDA: record 'noise types' in the pool — frequency train vs test, and model recall per type (train OOF).

Types are detected without labels from the raw pool record:
  empty_addr, null_addr (NULL / ## junk), native (Indic script), domain (.com/@/#/no-space handle),
  acronym (2-5 capital letters only), random (every name token unseen in any S1 name of that country and not a legal
  word), branch_word (extra token with high house-mismatch rate) is left to features.
"""
import os, re, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from io_utils import read_source
from normalize import name_tokens, LEGAL, FILLER
from decide import apply_rule

IND = re.compile(r"[ऀ-ൿ]")
DOM = re.compile(r"(\.com|\.in|\.net|\.org|\.fr)\b|^[@#]", re.I)
ACR = re.compile(r"^[A-Z]{2,5}$")


def types(pool, s1):
    """Return a DataFrame of boolean type flags for each pool record (vocab from S1 names per country)."""
    vocab = {}
    for c, g in s1.groupby("country"):
        v = set()
        for n in g["business_name"].values:
            v.update(name_tokens(n))
        vocab[c] = v
    nm = pool["business_name"].values; ad = pool["business_address"].values; cc = pool["country"].values
    out = {
        "empty_addr": np.array([a.strip() == "" for a in ad]),
        "null_addr": np.array([("NULL" in a) or ("##" in a) for a in ad]),
        "native": np.array([bool(IND.search(n)) for n in nm]),
        "domain": np.array([bool(DOM.search(n)) or (len(n) > 12 and " " not in n.strip() and n.strip().islower()) for n in nm]),
        "acronym": np.array([bool(ACR.match(n.strip())) for n in nm]),
    }
    rnd = []
    for n, c in zip(nm, cc):
        toks = [t for t in name_tokens(n) if t not in LEGAL and t not in FILLER and not t.isdigit()]
        v = vocab.get(c, set())
        rnd.append(len(toks) > 0 and all(t not in v for t in toks) and not IND.search(n))
    out["random_name"] = np.array(rnd)
    return pd.DataFrame(out)


if __name__ == "__main__":
    rows = []
    T = {}
    for d, p in (("dataset/train", "train"), ("dataset/test", "test")):
        s1 = read_source(d, p, 1)
        pool = pd.concat([read_source(d, p, 2), read_source(d, p, 3)], ignore_index=True)
        t = types(pool, s1); t["country"] = pool["country"].values; t["src"] = pool["src"].values
        t["entity_id"] = pool["entity_id"].values
        T[p] = t
        g = t.groupby("country")[[c for c in t.columns if c not in ("country", "src", "entity_id")]].mean()
        g.index = [f"{p}:{i}" for i in g.index]
        rows.append(g)
        per_s1 = t.groupby("country")[["empty_addr", "random_name", "acronym", "domain", "native"]].sum()
        n1 = s1["country"].value_counts()
        ps = per_s1.div(n1, axis=0); ps.index = [f"{p}:{i}" for i in ps.index]
        rows.append(ps.add_suffix("/S1"))
    print("share of pool records by type (and per-S1 counts):")
    print(pd.concat(rows).round(4).to_string())
    # model recall per type on train OOF (exp13)
    R = "runs/exp13_test/report"
    c = pd.read_parquet(f"{R}/oof_pairs.parquet"); prm = json.load(open(f"{R}/oof.json"))["params"]
    pr = apply_rule(c, *prm); got = set(zip(pr.s1, pr.x))
    tt = T["train"].set_index("entity_id")
    pos = c[c.y].copy(); pos["hit"] = [(a, b) in got for a, b in zip(pos.s1, pos.x)]
    neg_pred = pr[~pr.y]
    ty = tt.loc[pos.x]
    print("\nTRAIN OOF (exp13): recall of TRUE pairs by candidate type, and share of all true pairs")
    for col in ["empty_addr", "null_addr", "native", "domain", "acronym", "random_name"]:
        m = ty[col].values
        print(f"  {col:12s} share {m.mean():.3f}  recall {pos.hit.values[m].mean():.3f}  (others {pos.hit.values[~m].mean():.3f})")
    tn = tt.loc[neg_pred.x]
    print("\nFalse-positive pairs by candidate type (share of FPs):")
    for col in ["empty_addr", "null_addr", "native", "domain", "acronym", "random_name"]:
        print(f"  {col:12s} {tn[col].values.mean():.3f}")
