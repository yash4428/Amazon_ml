# PROGRESS

## Now
- Current step: Step 2 — Stage 0 DONE (25 Sep ~01:00 IST); 🛑 report sent to Yash.
- Next action: Step 3 — normalize.py (incl. Indic-script transliteration via aligned Unicode blocks, US/IN state canonicalisation, FR street abbreviations), then blocking.py with recall table on `val`. io_utils.py already written.
- Current best: none yet

## Environment
- Machine: Apple M4 Pro, 12 cores, 24 GB RAM, no NVIDIA GPU (Apple MPS only), ~140 GB free disk.
- Python 3.12 venv at `.venv/` (system python3 is 3.14; 3.12 chosen for reliable wheels). Run everything with `.venv/bin/python`.
- Libs (all permissive): pandas BSD, numpy BSD, scikit-learn BSD, scipy BSD, lightgbm MIT, rapidfuzz MIT, joblib BSD, tqdm MPL-2.0/MIT, pyarrow Apache-2.0. No unidecode.
- Pinned: `code/business_entity_resolution/requirements.txt`.
- `evaluate.py selftest` PASS.
- Git initialised; commits use `-c user.name="Yash Aggarwal"`.

## Key data facts (from Stage 0)
- Sizes (lines incl. header): train S1 2,206,821 / S2 5,034,616 / S3 5,285,603 / GT 2,206,821; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316. Dataset 2.4 GB.
- **Scale is large (~10M candidate-pool records on test).** Blocking must be chunked and memory-aware; full train set likely needs subsampling for model training.
- Train countries: S1 US 1.32M / India 0.88M. Test S1: India 810k, US 663k, **France 259k (15%)**; France pool S2 703k + S3 732k.
- **Singleton rate 0.056** (same for US & India). Predict-all-empty floor = 0.056. => recall matters a lot; singleton gate is low priority.
- Matches per S1: 0:123k 1:119k 2:375k 3:531k 4:484k 5:322k 6:165k 7:64k 8+:23k (mean ≈3.5). 1.69M S1 have >1 match from the same source.
- **One-to-one: strict** (0 S2/S3 ids in >1 S1 list). Distractors: 26% of pool matched to no S1.
- **All 7.64M true pairs share country label.**
- Exact normalised-name equality in only 22% of true pairs.
- Empty addresses: 0 in S1 (train & test); ~2.5-3% of S2/S3 (≈4% of true-pair partners).
- Postcodes present in only ~7% of addresses (both sides); equal in 90% when both present (noise injected).
- True-pair name/address: India 18% of partner names are native script (Devanagari/Tamil/Gujarati/Bengali...), 23% of partner addresses have native-script parts (often just state name). 24% (India) / 8% (US) of pairs share **no** Latin name token (native script, domains like `revitup.com`/`@handle`, fully random names like "Wexavi"). ~4-5% share no address token. **Never both** (0.0000) => name OR address always carries signal.
- Noise seen: token shuffles (OTE Ltd Pvt Food), dropped tokens, extra suffixes ([Inc], (Corp), Center, Services, "Dr", "Shri", "The"), OCR-ish typos (Muri1lo, Rev lt, Humme1, 0 for o), accent injection (Témplin, Ínc), house-number noise (7 vs 5, 1610 vs 1610-, 0700), "NULL", "##95", address component reorder, abbreviations (St/Saint(!)/Street, Dr, Ave, Xing), UPPERCASE in S2, state full names vs codes vs native script, city variants (Brookhaven/Ronkonkoma, Trappe Borough/Collegeville), "(ID: 63945)" junk.
- 23% of S1 normalised names are duplicated even within a 300k sample => chains/generic names; address must disambiguate.
- France: names are generic French words + legal form (SARL/SAS/EURL/SASU/SCI/SA/EI), e.g. "Pique Ecole EURL"; few cities (Lille, Nantes, Bordeaux, Pessac, Mérignac, Dunkerque, Roubaix, Tourcoing, Calais, Saint-Nazaire, Saint-Herblain); street words Rue/R/R./BD/Av./Cours/Route/Espl, "Bis", "N°"; regions (Hauts-de-France, Pays de la Loire, Nouvelle-Aquitaine) vs départements (Nord, Gironde, Loire-Atlantique); 5-digit postcodes rare.

## Decisions made (and why)
- Python 3.12 venv (3.14 too new for some wheels).
- No LLM reranker planned unless time allows: no CUDA GPU, and 1.7M S1 is large.
- One-to-one assignment: **ENABLED** (0 shared ids).
- Blocking: **hard-block by country** (all true pairs share the label; works for France automatically).
- Singleton rate 5.6% (<40%): singleton gate is NOT top priority; recall of multi-match sets is.
- Many S1 with ≥2 same-source matches: `sibling_support` + lower t2 are HIGH priority.
- Exact-name rate 22% (low): char-n-gram names (incl. space-stripped for domains), address char-TF-IDF blocking + features are top priority.
- Splits (built): val 1.77M/441k S1, val2 same sizes, loco_india 1.32M/883k, loco_us 883k/1.32M. val (random, seed 42), val2 (random, seed 7), loco_india (loco:India), loco_us (loco:US).

## Submissions (day, slot, tag, local val, public LB score)
- none

## Ideas backlog (ranked)
1. Address char-TF-IDF + rare-token blocking (names alone fail for ~8-24% of pairs).
2. Space-stripped name char n-grams so `revitup.com` / `@handle` / `vestsafeintelligencecom` match.
3. Indic-script -> Latin transliteration (one hand-written table; Indic Unicode blocks share offsets) + a native-word->English table learned from TRAIN pairs only.
4. sibling_support + t2 threshold (avg 3.5 matches per S1).
5. Chain disambiguation via house number / street features (23% duplicate S1 names).

## Known issues
- Validator `--check-ids` loads all S2/S3 ids (a few GB) — fine on 24 GB.
