# PROGRESS

## Now (25 Sep 22:00 IST)
- Public LB: day1_1 0.679 (rule), day1_2 0.955 (exp04, OOF 0.9736), day1_4 0.963 (exp05, OOF 0.9769). Leader 0.986; top-100 ~0.97.
- Running: exp06 (runs/exp06_test) = exp05 + house small-offset/branch-signature feats, minus length feats. Blocking cached. ETA ~22:45 -> day1_5 if OOF >= exp05.
- day1_3_probe (exp04 with France emptied) ready; F_France ≈ (0.955 − probe)/0.1498 + ~0.05.
- Local-vs-public gap: exp04 0.019, exp05 0.014 (hmis/France-ready feature helped public more than OOF).

## Unseen-country simulator (dev/country_cv.py, country_params.py) — 25 Sep 21:00
- Train on one country, predict the other (100-150k S1): all feats 0.9608 vs group-CV 0.9752 (drop 0.0144 ≈ public gap).
- no_xtok: same (0.9609) -> learned branch words not the culprit. no_dup: worse. no_cand_ctx: worse (0.9570). **no_len: 0.9626 (+0.0018), group unchanged** -> length feats dropped in exp06.
- Stricter thresholds for the unseen country: ~+0.002 US, 0 India -> not worth a France-specific rule.
- Extra loss on unseen country = FALSE POSITIVES (has_FP 0.0017->0.0054 IN / 0.0019->0.0088 US; singleton_FP x3-6). Recall unchanged.
- FP types: fake branches with house number nudged by 1-30 on the same street (+Partners/Co/Ltd Services); name-only empty-address records owned by a same-name S1.

## KEY FINDING (25 Sep 19:00) — how to evaluate
- Hard negatives are synthetic *sibling branches* of an S1 ("nobody"-owned pool records): same name + branch word (group, holdings, ventures, east, riverside, downtown, infratech, exports...) and a CHANGED house number (30->34, 8520->8527). 80% of singletons' top candidate is such a record.
- In the random `val` split these distractors are assigned randomly (80% go to the train side), so val S1s lose most of their hard negatives -> val is optimistic (rule: val 0.773 vs public 0.679; loco 0.694; train-part OOF tuning 0.680 ≈ public).
- => **Trust OOF on the full dataset/train (all distractors present) and country-CV OOF. Do NOT use val/val2 scores for decisions.**
- Extra-token stats (train, top-5 candidates): tokens like group/holdings/ventures/public/exports/overseas/infratech/east/metro/valley... occur only on negatives (0.0 on positives); center/service/labs/"doing business as" occur on positives. House first number equal: pos 68%, nobody-distractor 6%.

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

## Blocking (Step 3) — chosen config
- Per country; sparse IDF cosine top-K (chunked matmul in processes, per-row argpartition).
- `combo` space = name_core + addr_norm unigrams & bigrams, max_df 20000, K=20; `name_c4` = char 4-grams of space-free core name, max_df 5000, K=10.
- Train-part (30k S1, full 8M pool) table @K=20 each: combo 0.9617 recall/oracle 0.9865; name_bi 0.62; name_c4 0.60; addr_tok 0.85; full union 0.9717 (56 cand/S1).
  Mixes: combo20+c4_5 0.9656 (22.6/S1); **combo20+c4_10 0.9668 / oracle 0.9880 (26.6/S1)**; +addr5 0.9671; +all@10 0.9688 (34/S1).
- On val (evaluate.py): pair recall 0.9813, oracle 0.9934, 26.6 cand/S1 (val pool has fewer distractors per S1 than train part/test, so train-part 0.967 is the realistic number).
- Remaining misses: generic names + empty/partial addresses; rough transliterations ("laiph helthakeyar", "pra li").
- Runtime: val test side blocking ~90 s; normalisation cache (cache/norm_s{1,2,3}.parquet) built once in 2.6 min for all 22M records.

## Organiser email (25 Sep evening)
- candidate_pairs.tsv is part of the final submission; SMALLER candidate sets per S1 are ranked higher (beyond LB). Blocking must scale.

## Analysis 25 Sep 19:30-20:10
- No leakage in file order / id numbers (corr 0.000).
- Test has ~24% more pool per S1 than train in every country (5.5-5.8 vs 4.7).
- France: 16 candidates with combo>=0.5 per S1 (US/India ~5) -> dense look-alikes; same generator: branch words in French (Distribution, Developpement, Participations, Groupe, Associes) + changed house numbers.
- exp04 OOF loss 0.026: recall-only 0.0153 (54% blocking-caused), nonsingle-empty 0.0061, has-FP 0.0033, singleton-FP 0.0016. Pair precision ~0.995.
- Blocking with translit (30k S1): combo20+c4_10 0.9735@26.6; combo_c20+combo10 0.9770@22.5; **combo_c25+combo15 0.9813@30.0**; combo_c30+combo20+c4_10 0.9846@42; big union 0.9887@77.
- Stage-1 filter on blocking-only features is weak (top-10: recall 0.959); needs cheap string features.
- Unsupervised branch-word detector (house-mismatch rate per extra token): holdings .97 riverside .96 group .94 east .88 vs center .38 services .42.

## Experiments
| tag | change | val | loco_india | loco_us | oracle | notes |
|---|---|---|---|---|---|---|
| exp01_rule_baseline | max(combo_sim, 0.8*name_c4_sim), o2o, t1=0.58 t2=0.82 | 0.7730 | 0.6943 | 0.6951 | 0.9934 | public 0.679; tune-OOF 0.680 |
| exp02_lgbm | LightGBM 53 feats, 400k S1, lr .05 | OOF(val-train) 0.9622 | - | - | - | rank_cand dominates gain |
| exp04 | + learned translit dict + name-dup feats, 250k S1, lr .1, full train | OOF 0.9736 | India 0.9662 | US 0.9785 | - | public 0.955 (day1_2) |
| exp05 | blocking combo_c25+combo15 (train recall .9756), hmis branch-word feat, 300k S1 | OOF 0.9769 | India 0.9716 | US 0.9805 | - | public 0.963 (day1_4) |
| exp03 smoke | + xtok encoding + house cluster feats, 20k S1, lr .08 leaves 127 | OOF(val-train) 0.9650 | India OOF 0.951 | US OOF 0.974 | - | t1=0.78 t2=0.02 r=0.75 |

## Submissions (day, slot, tag, local val, public LB score)
- day1_1 | exp01_rule_baseline | val 0.7730 | **public 0.679**
- day1_2 | exp04 | OOF 0.9736 | **public 0.955**
- day1_4 | exp05 | OOF 0.9769 | **public 0.963**

## Ideas backlog (ranked)
1. Address char-TF-IDF + rare-token blocking (names alone fail for ~8-24% of pairs).
2. Space-stripped name char n-grams so `revitup.com` / `@handle` / `vestsafeintelligencecom` match.
3. Indic-script -> Latin transliteration (one hand-written table; Indic Unicode blocks share offsets) + a native-word->English table learned from TRAIN pairs only.
4. sibling_support + t2 threshold (avg 3.5 matches per S1).
5. Chain disambiguation via house number / street features (23% duplicate S1 names).

## Known issues
- Validator `--check-ids` loads all S2/S3 ids (a few GB) — fine on 24 GB.
