"""Text normalisation and light field parsing for business names and addresses.

Everything here is hand-written from general language knowledge (legal-form
lists, street-word abbreviations, a phonetic table for Indic scripts). No
external data is looked up. Country is never used to switch rules, so an
unseen country (France in test) goes through exactly the same code.
"""
import os
import re
import unicodedata

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Indic scripts -> Latin.
# All major Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati,
# Oriya, Tamil, Telugu, Kannada, Malayalam) share the same layout inside their
# 128-code-point block, so one offset table transliterates all of them.
# --------------------------------------------------------------------------
_INDIC_BASES = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x62: "li", 0x63: "li"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h"}
_INDIC_RE = re.compile(r"[ऀ-ൿ]")


def _indic_offset(ch):
    """Return the offset of ``ch`` inside its Indic block, or None if not Indic."""
    cp = ord(ch)
    if 0x0900 <= cp < 0x0D80:
        return (cp - 0x0900) % 0x80
    return None


def translit_indic(text):
    """Transliterate any Indic-script characters in ``text`` into rough Latin.

    Consonants carry an inherent 'a' unless followed by a vowel sign or virama;
    a word-final inherent 'a' is dropped (schwa deletion), so ग्लोबल -> global.
    Non-Indic characters are passed through unchanged.
    """
    if not _INDIC_RE.search(text):
        return text
    out = []
    pending_a = False  # consonant emitted, inherent 'a' not yet resolved
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if pending_a and ch.isalnum():
                out.append("a")
            pending_a = False  # word-final inherent 'a' dropped (schwa deletion)
            out.append(ch)
            continue
        if off in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[off])
            pending_a = True
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off == _NUKTA:
            pass
        elif off in _VOWELS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_VOWELS[off])
        elif off in _SIGNS:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(_SIGNS[off])
        elif 0x66 <= off <= 0x6F:
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(str(off - 0x66))
        else:
            if pending_a:
                out.append("a")
                pending_a = False
    # word-final pending 'a' is dropped (schwa deletion)
    s = "".join(out)
    return s


def strip_accents(text):
    """Unicode NFKD and drop combining marks (é -> e). Indic text must be transliterated first."""
    nf = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nf if not unicodedata.combining(c))


def base_clean(text):
    """Lowercase, transliterate Indic scripts, strip accents, map & -> and."""
    t = translit_indic(text)
    t = strip_accents(t).lower()
    t = t.replace("&", " and ")
    return t


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------
LEGAL = {
    # generic / US / UK
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "corp", "corporation", "co",
    "company", "plc", "lp", "pc", "pllc", "lc", "pa", "na",
    # India
    "pvt", "private", "opc", "huf",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "sca", "cie", "ste", "societe",
    "ei", "eirl", "selarl", "scm", "gie", "ets", "etablissements",
}
FILLER = {"the", "and", "of", "m/s", "ms", "shri", "sri", "smt", "dr", "et", "le", "la",
          "les", "de", "des", "du", "d", "l", "www", "com", "net", "org", "in", "dba",
          "fka", "aka", "formerly"}
DESCRIPTORS = {"group", "holdings", "enterprises", "enterprise", "services", "service",
               "solutions", "traders", "trading", "industries", "international", "center",
               "centre", "associates", "agency", "consultants", "labs"}
NAME_ABBR = {
    "pvt": "private", "pvte": "private", "ltd": "limited", "ltda": "limited",
    "corp": "corporation", "co": "company", "intl": "international", "mfg": "manufacturing",
    "bros": "brothers", "assoc": "associates", "assocs": "associates", "svc": "services",
    "svcs": "services", "mgmt": "management", "tech": "technology", "dept": "department",
    "inc": "incorporated", "ctr": "center", "centre": "center", "cntr": "center",
    "grp": "group", "hldgs": "holdings", "ind": "industries", "inds": "industries",
    "natl": "national", "univ": "university", "hosp": "hospital", "mkt": "market",
    "ent": "enterprises", "entp": "enterprises", "sys": "systems", "st": "saint",
    "ste": "societe", "sté": "societe", "cie": "compagnie", "ets": "etablissements",
    "assn": "association", "asso": "association", "praivet": "private", "limited": "limited",
}
_DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|trading as|aka|a\s*/\s*k\s*/\s*a|"
                     r"f\s*/\s*k\s*/\s*a|fka|formerly(?: known as)?)\b")
_ID_JUNK_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|\bid\s*:\s*\d+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
_OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "7": "t", "8": "b"})
_DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|in|net|org|co|biz|info|fr|us)(?:\.[a-z]{2})?\b")


def _fix_ocr_token(tok):
    """Replace digits that sit inside alphabetic words (humme1 -> hummel, de5ert -> desert)."""
    if tok.isalpha() or tok.isdigit():
        return tok
    letters = sum(c.isalpha() for c in tok)
    if letters >= 2 and letters >= len(tok) - 2:
        return tok.translate(_OCR_MAP)
    return tok


def _merge_single_letters(tokens):
    """Join runs of single-letter tokens: ['l','l','c'] -> ['llc'], ['p','e'] -> ['pe']."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def name_tokens(raw):
    """Tokenise a business name: clean, drop id junk, fix OCR digits, merge initials, expand abbreviations."""
    t = base_clean(raw)
    t = _ID_JUNK_RE.sub(" ", t)
    t = _DOMAIN_RE.sub(r" \1 ", t)
    t = t.replace("@", " ").replace("#", " ")
    t = _PUNCT_RE.sub(" ", t)
    toks = [_fix_ocr_token(x) for x in t.split()]
    toks = _merge_single_letters(toks)
    return [NAME_ABBR.get(x, x) for x in toks]


_LEGAL_EXPANDED = {NAME_ABBR.get(x, x) for x in LEGAL} | LEGAL


def normalize_name(raw):
    """Return a dict of name variants for one raw business name.

    name_norm    : cleaned tokens joined by spaces (abbreviations expanded)
    name_core    : name_norm without legal forms / filler words
    name_compact : name_core with spaces removed (matches domains / @handles)
    acronym      : first letters of core tokens
    dba          : ' | '-joined alternative names split on dba/aka/fka/brackets
    """
    toks = name_tokens(raw)
    core = [x for x in toks if x not in _LEGAL_EXPANDED and x not in FILLER]
    if not core:
        core = toks
    # DBA / bracket alternative names
    low = _ID_JUNK_RE.sub(" ", base_clean(raw))
    parts = [p for p in re.split(r"\|", "|".join(_DBA_RE.split(re.sub(r"[()\[\]]", " | ", low))))
             if p.strip()]
    alts = []
    for p in (parts if len(parts) > 1 else []):
        pt = [NAME_ABBR.get(x, x) for x in _PUNCT_RE.sub(" ", p).split()]
        pt = [x for x in pt if x not in _LEGAL_EXPANDED and x not in FILLER]
        if pt and pt != core:
            alts.append(" ".join(pt))
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_compact": "".join(core),
        "acronym": "".join(x[0] for x in core if x and not x.isdigit()) if len(core) > 1 else "",
        "dba": " | ".join(dict.fromkeys(alts)),
    }


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------
# Canonical short forms. Street words are *contracted* (street -> st) rather than
# expanded, so "St", "Street" and the injected "Saint" all agree.
ADDR_ABBR = {
    # English
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "av", "ave": "av",
    "boulevard": "bd", "blvd": "bd", "boul": "bd", "drive": "dr", "lane": "ln", "court": "ct",
    "place": "pl", "plaza": "plz", "crossing": "xing", "highway": "hwy", "parkway": "pkwy",
    "pky": "pkwy", "circle": "cir", "terrace": "ter", "square": "sq", "trail": "trl",
    "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg",
    "bldng": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "fort": "ft", "point": "pt", "heights": "hts", "center": "ctr",
    "centre": "ctr", "expressway": "expy", "freeway": "fwy", "turnpike": "tpke",
    "unit": "unit", "room": "rm", "township": "twp", "junction": "jct", "route": "rte",
    # India
    "mahatma": "mg", "near": "nr", "opposite": "opp", "opp": "opp", "sector": "sec",
    "phase": "ph", "nagar": "ngr", "marg": "mrg", "chowk": "chk", "colony": "col",
    "enclave": "encl", "vihar": "vhr", "mohalla": "mohalla", "gali": "gali", "cross": "crs",
    "main": "mn", "extension": "extn", "ext": "extn", "industrial": "indl", "estate": "est",
    "complex": "cmplx", "apartments": "apts", "society": "soc", "district": "dist",
    "dt": "dist", "post": "po", "village": "vill", "vil": "vill", "taluk": "tq",
    "taluka": "tq", "tehsil": "teh", "tal": "tq", "ground": "gr", "first": "1st",
    "second": "2nd", "third": "3rd",
    # French
    "rue": "r", "avenues": "av", "boulevards": "bd", "impasse": "imp", "chemin": "ch",
    "chem": "ch", "faubourg": "fg", "allee": "all", "place": "pl", "quai": "qu",
    "cours": "crs", "esplanade": "espl", "residence": "res", "sainte": "ste", "route": "rte",
    "rte": "rte", "lieu": "ld", "lieudit": "ld",
}
ADDR_NOISE = {"null", "none", "na", "n/a", "no", "nos", "number", "num", "door", "h", "hno",
              "house", "plot", "flat", "shop", "office", "#", "the", "of", "de", "du", "des",
              "la", "le", "les", "d", "l", "et", "and", "cdp", "n", "house"}
# Keep "n" (north) meaningful only when it is not the "No." marker; we drop "n" since
# "N° 23" / "H.N." are far more frequent than a bare "N" direction after contraction.
_NUMTOK_RE = re.compile(r"\d")
_POSTCODE_RE = re.compile(r"(?<![\d/-])(\d{3}\s?\d{3}|\d{5})(?:-\d{4})?(?![\d/-])")
_HOUSE_RE = re.compile(r"\b\d+[a-z]?(?:[/-]\d+[a-z]?)*\b")
_LANDMARK_RE = re.compile(r"\b(near|nr|opp|opposite|behind|beside|next to|pres de|en face|b/h)\b")


def _addr_token(tok):
    """Canonicalise one address token (strip leading zeros, contract street words)."""
    if tok.isdigit():
        return tok.lstrip("0") or "0"
    return ADDR_ABBR.get(tok, tok)


def normalize_address(raw):
    """Return a dict of address variants and parsed pieces.

    addr_norm     : canonical tokens joined by spaces (order kept)
    postcode      : last standalone 5-6 digit number ('' if none)
    house_numbers : space-joined number-like tokens (e.g. '51/10g/1 12')
    landmark      : 1 if the address uses a landmark phrase (near/opp/behind...)
    """
    t = base_clean(raw)
    t = t.replace("n°", " no ").replace("nº", " no ")
    landmark = 1 if _LANDMARK_RE.search(t) else 0
    pcs = _POSTCODE_RE.findall(t)
    postcode = pcs[-1].replace(" ", "") if pcs else ""
    houses = [h.lstrip("0") or "0" for h in _HOUSE_RE.findall(t)]
    houses = [h for h in houses if h.replace(" ", "") != postcode]
    t = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", t)  # 73rd -> 73 (house/street ordinals)
    t = _PUNCT_RE.sub(" ", t)
    toks = [_addr_token(x) for x in t.split()]
    toks = [x for x in toks if x not in ADDR_NOISE]
    return {
        "addr_norm": " ".join(toks),
        "postcode": postcode,
        "house_numbers": " ".join(dict.fromkeys(houses)),
        "landmark": landmark,
    }


# --------------------------------------------------------------------------
# DataFrame-level helpers (parallel + cached)
# --------------------------------------------------------------------------
NORM_COLS = ["name_norm", "name_core", "name_compact", "acronym", "dba",
             "addr_norm", "postcode", "house_numbers", "landmark"]


def _normalize_block(names, addrs):
    """Normalise parallel lists of names and addresses; returns a dict of column lists."""
    out = {c: [] for c in NORM_COLS}
    for n, a in zip(names, addrs):
        dn = normalize_name(n)
        da = normalize_address(a)
        for c in NORM_COLS:
            out[c].append(dn[c] if c in dn else da[c])
    return out


def normalize_frame(df, n_jobs=1, block=200_000):
    """Add the NORM_COLS columns to a frame with business_name / business_address."""
    from joblib import Parallel, delayed
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    spans = [(i, min(i + block, len(df))) for i in range(0, len(df), block)]
    res = Parallel(n_jobs=n_jobs)(delayed(_normalize_block)(names[a:b], addrs[a:b]) for a, b in spans)
    out = df.copy()
    for c in NORM_COLS:
        vals = [v for r in res for v in r[c]]
        out[c] = np.array(vals, dtype=np.int8) if c == "landmark" else vals
    return out


def normalize_cached(df, cache_path, n_jobs=1):
    """Normalise ``df``, re-using/extending a parquet cache keyed by entity_id.

    Only ids not yet in the cache are normalised; the cache is then rewritten
    with the union, so validation splits re-use the full-train normalisation.
    """
    cached = None
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        todo = df[~df["entity_id"].isin(cached["entity_id"])]
    else:
        todo = df
    if len(todo):
        new = normalize_frame(todo, n_jobs=n_jobs)[["entity_id"] + NORM_COLS]
        cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cached.to_parquet(cache_path, index=False)
    got = cached.set_index("entity_id").loc[df["entity_id"], NORM_COLS]
    out = df.reset_index(drop=True).copy()
    for c in NORM_COLS:
        out[c] = got[c].values
    return out


def load_normalized(data_dir, prefix, n_jobs=1):
    """Read S1 and pool (S2+S3) from ``data_dir`` and attach normalised columns (cached)."""
    from io_utils import read_source
    import config
    frames = []
    for n in (1, 2, 3):
        df = read_source(data_dir, prefix, n)
        frames.append(normalize_cached(df, os.path.join(config.CACHE_DIR, f"norm_s{n}.parquet"),
                                       n_jobs=n_jobs))
    return frames[0], pd.concat(frames[1:], ignore_index=True)


# --------------------------------------------------------------------------
# Learned transliteration dictionary (from TRAIN pairs only).
# Indic-script names are transliterated phonetically above ("pharst oyan
# bildars"); the training ground truth contains many (Latin S1, native-script
# S2/S3) pairs of the same business, whose name tokens align position by
# position. We count source->target token pairs and keep confident mappings
# (pharst -> first, bildars -> builders). Only used on names that contained
# Indic characters. No external data.
# --------------------------------------------------------------------------
_HAS_INDIC = re.compile(r"[ऀ-ൿ]")


def build_translit_dict(s1, pool, truth, min_count=3, min_share=0.6):
    """Learn transliterated-token -> English-token map from true (Latin S1, Indic pool) pairs."""
    from collections import Counter, defaultdict
    s1_name = dict(zip(s1["entity_id"], s1["name_norm"]))
    s1_raw = dict(zip(s1["entity_id"], s1["business_name"]))
    idx = pool.set_index("entity_id")
    native = idx[idx["business_name"].map(lambda t: bool(_HAS_INDIC.search(t)))]
    nat_name = dict(zip(native.index, native["name_norm"]))
    counts = defaultdict(Counter)
    for s, xs in truth.items():
        if s not in s1_name or _HAS_INDIC.search(s1_raw[s]):
            continue
        t1 = s1_name[s].split()
        for x in xs:
            if x in nat_name:
                t2 = nat_name[x].split()
                if len(t1) == len(t2):
                    for a, b in zip(t2, t1):
                        if a != b:
                            counts[a][b] += 1
    out = {}
    for a, cnt in counts.items():
        b, n = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if n >= min_count and n / tot >= min_share:
            out[a] = b
    return out


def apply_translit_dict(df, mapping):
    """Rewrite name fields of Indic-script records using the learned token map (in place copy)."""
    if not mapping:
        return df
    m = df["business_name"].map(lambda t: bool(_HAS_INDIC.search(t))).values
    if not m.any():
        return df
    df = df.copy()
    norm = []
    for n in df.loc[m, "name_norm"].values:
        norm.append(" ".join(mapping.get(t, t) for t in n.split()))
    core, compact, acro = [], [], []
    for n in norm:
        toks = n.split()
        c = [x for x in toks if x not in _LEGAL_EXPANDED and x not in FILLER] or toks
        core.append(" ".join(c))
        compact.append("".join(c))
        acro.append("".join(x[0] for x in c if x and not x.isdigit()) if len(c) > 1 else "")
    df.loc[m, "name_norm"] = norm
    df.loc[m, "name_core"] = core
    df.loc[m, "name_compact"] = compact
    df.loc[m, "acronym"] = acro
    return df
