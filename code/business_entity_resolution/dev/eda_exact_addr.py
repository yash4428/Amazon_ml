"""EDA: candidates at the S1's EXACT address (same house number, address token-set >= 95) whose NAME is unlike the
S1 name (random names, acronyms, domains) — accepted rate per country (test) vs truth rate (train OOF)."""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from normalize import load_normalized, reparse_numbers
from decide import apply_rule
RUN = "runs/exp17_test"; prm = tuple(json.load(open(f"{RUN}/report/oof.json"))["params"])
ACR = re.compile(r"^[A-Z]{2,5}$")
for split in ("train", "test"):
    s1, pool = load_normalized(f"dataset/{split}", split)
    s1, pool = reparse_numbers(s1, n_jobs=10), reparse_numbers(pool, n_jobs=10)
    if split == "train":
        c = pd.read_parquet(f"{RUN}/report/oof_pairs.parquet"); c = c[~c.x.str.startswith("SY")]
        sid = {e: k for k, e in enumerate(s1.entity_id)}; pid = {e: k for k, e in enumerate(pool.entity_id)}
        c["i"] = c.s1.map(sid).values; c["j"] = c.x.map(pid).values
        n1 = pd.Series(s1.country.values[list(set(c.i))]).value_counts()
    else:
        c = pd.read_parquet(f"{RUN}/test_scores.parquet"); n1 = s1.country.value_counts()
    c = c.reset_index(drop=True)
    acc = set(zip(*[apply_rule(c, *prm)[k].values for k in ("i", "j")]))
    h1 = np.array([x.split(" ", 1)[0] for x in s1.house_numbers.values[c.i.values]])
    h2 = np.array([x.split(" ", 1)[0] for x in pool.house_numbers.values[c.j.values]])
    at = process.cpdist(s1.addr_norm.values[c.i.values].tolist(), pool.addr_norm.values[c.j.values].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    exact = (h1 != "") & (h1 == h2) & (at >= 95)
    cc = c[exact].copy()
    nt = process.cpdist(s1.name_core.values[cc.i.values].tolist(), pool.name_core.values[cc.j.values].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    raw2 = pool.business_name.values[cc.j.values]
    kind = np.select([np.array([bool(ACR.match(n.strip())) for n in raw2]),
                      np.array([bool(re.search(r"\.com|^@|^#|com$", n, re.I)) for n in raw2]),
                      nt < 50, nt < 90], ["acronym", "domain", "name<50 (random/other)", "name 50-90"], "name>=90")
    cc["kind"] = kind; cc["acc"] = [(a, b) in acc for a, b in zip(cc.i, cc.j)]; cc["cty"] = s1.country.values[cc.i.values]
    for k in ("name<50 (random/other)", "name 50-90", "acronym", "domain"):
        for cty in sorted(set(cc.cty)):
            m = (cc.kind.values == k) & (cc.cty.values == cty)
            line = f"{split:5s} {cty:7s} exact-address {k:24s} {100 * m.sum() / n1[cty]:6.2f}/100 S1  accepted {cc.acc.values[m].mean():.3f}"
            if split == "train":
                line += f"  P(true) {cc.y.values[m].mean():.3f}  P(true|rejected) {cc.y.values[m & ~cc.acc.values].mean():.3f}"
            print(line)
