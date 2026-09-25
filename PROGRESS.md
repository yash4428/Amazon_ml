# PROGRESS / HANDOVER — Amazon ML Challenge 2026: Business Entity Resolution

This file is the **single source of truth** for the team. It records the current state, everything we learned
(including dead ends), exact numbers, how to run things and what to do next. Read it top to bottom before
changing anything. Rulebooks: `CLAUDE.md` (hard rules, metric, pipeline spec) and `START_HERE.md` (operating plan).

---------------------------------------------------------------------------------------------------------------

## 0. TL;DR (updated 26 Sep 00:15 IST)

- **Best public score so far: 0.964** (day1_5, blend of exp06+exp07). Leader 0.9884, top-15 ≈ 0.984, top-100 ≈ 0.97.
  We were ~300th at end of Day 1.
- **Local metric to trust: OOF macro-F0.5 on the FULL `dataset/train`** (written by every lgbm run to
  `runs/<exp>/report/oof.json`). **Do not use the `val`/`val2` split scores** — they are optimistic (see §4).
- **Diagnosis of our gap** (§6): (a) France ≈ 0.91 vs US/India ≈ 0.97 on test (measured with a probe submission);
  (b) test has 40-55% more fake-branch distractors per S1 than train (France ~9 per S1), so a model trained on
  train density over-accepts on test; (c) blocking recall (~97.6%) caps us — half of all OOF loss.
- **Overnight (26 Sep) running**: `runs/night_chain.sh`:
  exp09 = wide blocking (combo_c@60 + combo@30) + **stage-1 filter** (~10 cand/S1 instead of 30, higher recall)
  → then `dev/crowd_eval.py` (A/B: does training with synthetic fake branches help on test-like data?).
  Status/timing in `runs/night_chain.log`, `runs/exp09_test.log`, `runs/crowd_eval.log`.
- **Morning deliverable**: best validated file in `submissions/day2_best/` (see §9 for what was chosen and why).

---------------------------------------------------------------------------------------------------------------

## 1. How to run (from `student_resource/`)

```bash
# one-time environment (Python 3.12; system 3.14 is too new for some wheels)
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt

# full pipeline: train on dataset/train, predict dataset/test (LightGBM mode is default)
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py \
    --train-dir dataset/train --test-dir dataset/test --out-dir runs/expNN_test

# checks before any upload
python3 utils/validate_submission.py --matching runs/expNN_test/matching_results.tsv \
    --candidate runs/expNN_test/candidate_pairs.tsv --test-dir dataset/test
.venv/bin/python code/business_entity_resolution/src/sanity_check.py --out-dir runs/expNN_test --test-dir dataset/test
```

Useful flags of `run_pipeline.py`:
- `--mode rule|lgbm` (default lgbm). `--model-s1 N` train S1 sampled for the matching model (default config.MODEL_S1=300k).
- `--oof-only` stop after the honest OOF score (no test inference). `--cv country` leave-one-country-out folds.
- `--fixed-rounds R --params-from runs/X/report/oof.json` skip CV (fast; used for exp07). `--sample-seed k` different S1 sample.
- `--crowd f` add f×(distractor count) synthetic fake branches to the TRAIN pool (§6.4).

Outputs per run dir: `matching_results.tsv`, `candidate_pairs.tsv`, `run_info.json`, `test_scores.parquet`
(every candidate pair with its model score → lets you re-decide thresholds / blend without re-running),
`report/oof.json` (OOF score, tuned params, per-country OOF), `report/oof_pairs.parquet`, `report/oof_s1.parquet`,
`report/feature_importance.csv`.

Blending two runs (same OOF sample → evaluated on OOF first):
`.venv/bin/python code/business_entity_resolution/dev/blend.py runs/A runs/B [--w 0.5] [--params-from ...] [--write runs/blend]`

Runtime (M4 Pro, 12 cores, 24 GB, no GPU): full run ≈ 100 min when blocking must be recomputed (train blocking of
2.2M S1 ≈ 30-35 min, test ≈ 20-25 min); ≈ 55 min when both blocking caches exist (`cache/blocks/`).
**Memory: never run two full pipelines at once** (each peaks at 12-15 GB).

Caches (gitignored, safe to delete, rebuilt automatically): `cache/norm_s{1,2,3}.parquet` normalised text for all
22M records (2.6 min to build); `cache/blocks/*.parquet` candidate frames keyed by (config, ids, NORM_VERSION, stage-1 cfg).

---------------------------------------------------------------------------------------------------------------

## 2. Data facts (Stage 0)

- Train: S1 2,206,820 (US 1.32M, India 0.88M) | S2 5.03M | S3 5.29M. Test: S1 1,732,543 (India 810k, US 663k,
  **France 259k = 15%**) | S2 4.89M | S3 5.08M.
- **Test has ~24% more pool per S1 than train in every country** (5.5-5.8 vs 4.7) — see §6.3 for what those are.
- Singleton rate 0.056 (predict-all-empty floor). Matches per S1 mean ≈ 3.5 (0:123k 1:119k 2:375k 3:531k 4:484k 5:322k 6:165k 7+:87k).
- **One-to-one is strict** (no S2/S3 id in two S1 lists) → greedy one-to-one assignment is enabled.
- **All 7.64M true pairs share the country label** → hard-block by country (France gets its own block automatically).
- 26% of the train pool matches no S1 (distractors). Exact normalised-name equality in only 22% of true pairs.
- S1 never has empty addresses; ~3% of S2/S3 do. Postcodes are rare (~7%).
- India: 18% of partner names are native script (Devanagari/Tamil/Telugu/Bengali/Gujarati/Odia…), many addresses have
  native-script state names. 24% (India) / 8% (US) of true pairs share **no** Latin name token (native script, domains
  like `revitup.com`, `@handle`, fully random names like "Wexavi"), but **name OR address always shares a token**.
- Noise in true copies: token shuffles, dropped tokens, added suffixes ([Inc], (Corp), Center, Services, Dr, Shri, The),
  OCR typos (Muri1lo, Humme1, 0↔o), injected accents, house-number noise (7 vs 5, 0700, 1610-, 1976-1978), "NULL",
  "##95", component reordering, St/Street/**Saint**, state code vs name vs native script, city variants.
- **Hard negatives = synthetic "sibling branches"**: same name + a branch word (holdings, group, ventures, east,
  riverside, downtown, infratech, exports…; in France: Distribution, Développement, Participations, Groupe, Associés)
  and the **house number nudged by ~1-30** (30→34, 8520→8527, 46→53). 80% of singletons' top candidate is one.
- No leakage: file order and id numbers are uncorrelated with matches (corr 0.000).
- France: names = generic French words + legal form (SARL/SAS/EURL/SASU/SCI/SA/EI/SNC), few cities, street words
  Rue/R./BD/Av./Cours/Chemin/Quai, "Bis", "N°", regions vs départements. Very dense: **16 candidates with
  blocking sim ≥ 0.5 per S1 vs ~5 in US/India; ~9 branch-like candidates per S1**.

---------------------------------------------------------------------------------------------------------------

## 3. Pipeline (code/business_entity_resolution/src)

1. **normalize.py** — NFKD accent strip, `&`→and, Indic→Latin transliteration with ONE table (all Indic Unicode
   blocks share the same 128-code-point layout), OCR digit fix inside words, merge single-letter runs (L.L.C.→llc),
   abbreviation expansion, legal-form removal (US/IN/FR lists), DBA split, domain/@handle → compact name; address
   contraction (street/st/saint→st, rue→r …), postcode + house numbers + landmark flag. Hand-written lists only.
   **Learned transliteration dict** (`build_translit_dict`): aligns tokens of true (Latin S1, native-script pool)
   TRAIN pairs → 567 mappings (pharst→first, payoniyar→pioneer, enarji→energy). Native-pair token overlap 0.21→0.94.
2. **blocking.py** — per country, sparse IDF-cosine top-K via chunked matmul (processes) + per-row argpartition.
   Spaces: `combo` (name_core + addr unigrams/bigrams), `combo_c` (name char-4-grams + addr uni/bigrams),
   `name_c4`, `addr_tok`, `name_bi`, `name_tok`. max_df prunes common features. Optional per-country `post` hook.
3. **stage1.py** (new, exp09) — cheap LightGBM on blocking sims/ranks + 3 rapidfuzz scores + house agreement +
   per-S1 gaps; keeps p ≥ 0.001 (≤ 20 per S1). Trained on held-out train S1 (not in the matching-model sample).
4. **features.py** — rapidfuzz name/address similarities, compact/acronym/DBA, postcode & house-number states,
   house cluster features, **learned extra-token encoding** (`xtok_*`, OOF target encoding of branch words),
   **unsupervised branch-word detector** (`hmis_*`: per extra token, rate of co-occurring with a changed house
   number, computed on each dataset's own candidates → works for French words), **branch signature**
   (same street, number nudged ≤ 30), name-duplicate counts, context features (within-S1 rank/gap/n_close and
   **candidate-side** rank/gap/#S1 claiming the record, computed on the FULL candidate set).
5. **model.py** — LightGBM (lr 0.1, 127 leaves, deterministic), GroupKFold by S1 → OOF; final single model on all
   sampled pairs with rounds = 1.1 × mean best_iter.
6. **decide.py** — greedy one-to-one, then keep top-1 if p ≥ t1, extras if p ≥ t2 and p ≥ r·best; (t1,t2,r) tuned by
   coordinate descent on OOF macro-F0.5 (numpy, seconds).
7. **run_pipeline.py** — orchestration; **sanity_check.py** — row counts, per-country empty rate (France present),
   matches ⊆ candidates.

---------------------------------------------------------------------------------------------------------------

## 4. How to evaluate (IMPORTANT — we learned this the hard way)

- `evaluate.py make-split --mode random` assigns distractors randomly, so ~80% of a val S1's fake branches land on the
  TRAIN side → **val is optimistic**. Rule baseline: val 0.773 vs public 0.679.
- **Honest local metric = OOF on the full `dataset/train`** (every S1 has all its distractors). Rule tuning-OOF 0.680 ≈
  public 0.679. For the LightGBM it is still ~0.013-0.019 above public because of the train→test shift (§6).
- **Unseen-country simulator**: `dev/country_cv.py` (train on one country, predict the other). Drop vs group CV =
  0.0144 (0.9752→0.9608) ≈ our public gap. Use it to reject features that only work on seen countries.
- **Crowding simulator**: `--crowd` / `dev/crowd_eval.py` add synthetic fake branches (nudged copies of distractors)
  to mimic test density.
- **France probe** (costs a slot): take a submitted file, empty all France rows, resubmit:
  F_France ≈ (score − probe)/0.1498 + s_France (s≈0.05).

---------------------------------------------------------------------------------------------------------------

## 5. Experiments (local = honest OOF on full train unless stated)

| tag | change | local | per-country | public | notes |
|---|---|---|---|---|---|
| exp01 | rule: max(combo, .8·name_c4), o2o | val 0.773 (tune-OOF 0.680) | loco IN 0.694 / US 0.695 | **0.679** | day1_1 |
| exp02 | LightGBM 53 feats, 400k S1 | OOF(val-train) 0.9622 | – | – | cand-side rank dominates |
| exp03 | + xtok + house cluster feats (smoke, 20-40k S1) | OOF(val-train) 0.9650-0.9671 | IN .954 US .976 | – | |
| exp04 | + learned translit + name-dup, 250k S1, full train | **0.9736** | IN .9662 US .9785 | **0.955** | day1_2 |
| probe | exp04 with France rows emptied | – | – | **0.831** | ⇒ France ≈ 0.88, US+IN ≈ 0.968 |
| exp05 | blocking combo_c25+combo15 (train recall .9756, 30/S1) + hmis, 300k S1 | **0.9769** | IN .9716 US .9805 | **0.963** | day1_4; France ≈ 0.91 |
| exp06 | + branch-signature feats, − length feats | 0.9771 | IN .9719 US .9805 | – | changes France preds 2.3× more than US/IN |
| exp07 | exp06 feats, 600k S1 (seed+1), no CV | – | – | – | |
| blend | 0.5·exp06 + 0.5·exp07 | (blend05/06 OOF 0.97726) | – | **0.964** | day1_5 |
| exp08 | exp06 + `--crowd 0.5` | stopped after caching crowded blocking | – | – | used by crowd_eval |
| exp09 | wide blocking c60+c30 + stage-1 (~11/S1, recall .9819) | **0.9795** | IN .9763 US .9817 | – | 2.7× fewer candidates |
| exp10 | exp09 + crowd 0.5 (train-time synthetic branches) | 0.9789 (crowded OOF) | IN .9756 US .9811 | – | expected best on test |

Blocking tables (30k train-part S1, with learned translit):
- combo20+c4_10 0.9735 @26.6/S1 · combo_c20+combo10 0.9770 @22.5 · **combo_c25+combo15 0.9813 @30.0** ·
  combo_c30+combo20+c4_10 0.9846 @42 · union K60/30 0.9888 @72.
- **Stage-1 (cheap string feats) on the K60/30 union: p≥0.001 → 10.0/S1 recall 0.9842 oracle 0.9952;
  top-12 → 0.9833; top-20 → 0.9860.** (Stage-1 on blocking-only features was weak: top-10 0.959.)
- Crowding (+60% synthetic branches): combo_c25+combo15 0.9813→0.9793 only; K60+30 0.9877.

Unseen-country simulator (country folds, 100-150k S1): all 0.9608 · no_xtok 0.9609 · no_dup 0.9588 ·
no_cand_ctx 0.9570 · **no_len 0.9626** (group CV unchanged 0.9751) → length feats removed in exp06.
Stricter thresholds on the unseen country: +0.002 US, 0 India → no France-specific thresholds.
Extra unseen-country loss is **false positives** (has_FP 0.0017→0.0054 IN, 0.0019→0.0088 US), recall unchanged.

OOF loss decomposition (exp05/06, total ≈ 0.023): recall-only 0.0131 (45-54% caused by blocking misses),
non-singleton predicted empty 0.0051, a wrong match included 0.0036, singleton given a match 0.0013.
Pair precision ≈ 0.995; scores are well calibrated (only ~7% of pairs in 0.1-0.9).

Rejected ideas (with evidence): per-country (France) strict thresholds (~0 gain) · expected-F0.5 decision
(0.9763 < 0.9771) · word-substitution as France FP signal (same share in all countries; 98% positive in train) ·
removing xtok (no change) · removing candidate-side context (worse).

---------------------------------------------------------------------------------------------------------------

## 6. What we learned (chronological, most important first)

6.1 **val split is optimistic** (§4). 6.2 **LightGBM ≫ rule** (0.68 → 0.955). Candidate-side context
(`ctx_*_rank_cand`, `gap_cand`) is the top feature: "does another S1 fit this record better".
6.3 **Test is harder than train**: per S1, branch-like candidates (same street, number nudged ≤30):
train US 1.02 / IN 1.37 → **test US 1.58 (+54%), IN 1.89 (+38%), France 9.2**; copy-like candidates (same number,
name tset ≥ 80) unchanged (~2.0; France 2.7). Predicted matches per S1 on test ≈ train. ⇒ extra test pool =
distractors, not copies. The model's odds are learned at train density → over-accepts on test.
6.4 **France is the weakest country** (probe): ≈0.88 (exp04) → ≈0.91 (exp05, thanks to the unsupervised hmis
branch-word feature). US+India ≈ 0.97 on test. Gap to leader ≈ 0.011 from France + 0.014 from US/India.
6.5 French true matches sit deep in candidate ranks 3× more than US (rank 18-24 of 25: 1.26% vs 0.40%; found only by
`combo`: 2.06% vs 0.65%) → narrow K cuts French copies → wide blocking + stage-1 (exp09).
6.6 Unseen-country FPs are branch records with nudged numbers and name-only empty-address records owned by a
same-name S1 → branch-signature features (exp06).

---------------------------------------------------------------------------------------------------------------

## 7. Submissions

| day/slot | file | tag | local | public |
|---|---|---|---|---|
| day1_1 | submissions/day1_1 | exp01 rule | val 0.773 | 0.679 |
| day1_2 | submissions/day1_2 | exp04 | OOF 0.9736 | 0.955 |
| day1_3 | submissions/day1_3_probe | exp04, France emptied (diagnostic) | – | 0.831 |
| day1_4 | submissions/day1_4 | exp05 | OOF 0.9769 | 0.963 |
| day1_5 | submissions/day1_5_final | 0.5·exp06 + 0.5·exp07 | OOF ≈0.9773 | **0.964** |

Rules for slots: max 5/day. Each slot should either improve the honest local score or answer a question (probe).
TSV outputs are gitignored (large); NOTE.md files in each submission folder record config + scores.

---------------------------------------------------------------------------------------------------------------

## 8. Next steps (ranked, for whoever picks this up)

1. **Finish/evaluate exp09** (wide blocking + stage-1). Expect higher recall (+0.3% pairs → ~+0.002-0.004) AND
   3× smaller candidate_pairs.tsv (organiser email: smaller candidate sets rank higher in final evaluation).
2. **Density shift** (§6.3): if `runs/crowd_eval.log` shows model B (trained crowded) beats A on crowded val
   without losing on normal val → rerun exp09 with `--crowd 0.5` (needs new crowded wide blocking ~+50 min).
   Also consider more realistic synthetic branches (S1/copy + learned branch word + nudged number).
3. **France**: measure with another probe only if needed; ideas: relative max_df (France vocab is tiny), France-like
   density in training (crowd), candidate-to-candidate support (true copies agree with each other, branch copies
   agree on the wrong number), transitivity/clustering of pool records.
4. **Model recall** (~2.8% of true pairs missed by the model at p<0.1): name-only empty-address copies with
   ambiguous names, random-name copies at the exact address. Sibling-support stacking is the planned fix.
5. Optional reranker on borderline pairs with a small MIT/Apache model (e.g. multilingual MiniLM, Apache-2.0) —
   ask Yash before adding any model/dependency; no CUDA here (MPS only).
6. **Do NOT**: tune thresholds on public LB, hand-label test, pseudo-label test without asking organisers (rule 7).
7. Package (Day 3 by 18:00): README, requirements, Documentation_template.md (use this file + experiments table),
   clean-room reproduction, zip `<team>_submission.zip`.

---------------------------------------------------------------------------------------------------------------

## 9. Overnight log (26 Sep)

- 00:00 exp09 code: wide blocking + stage-1 filter; mini end-to-end smoke (30k train S1 / 10k test S1) passed.
- 00:15 `runs/night_chain.sh` started: waits for exp08's crowded blocking cache → stops exp08 → exp09 → crowd_eval.
- 00:22 stage-1 trained on 60k held-out train S1: wide union 70.8 cand/S1, recall 0.9864.
- 00:58 train blocking + stage-1: India 60.8M -> 9.9M pairs (11.2/S1), US 94.9M -> 14.6M (11.0/S1).
  **Recall after stage-1 on the 300k model sample: 0.9819 at ~11 cand/S1** (exp05: 0.9756 at 30/S1).
- 01:04 **exp09 OOF 0.9795** (India 0.9763, US 0.9817) vs exp06 0.9771 → +0.0024. Params t1=0.70 t2=0.74 r=0.
- 01:41 exp09 test written: 20.6M candidate pairs = **11.9 cand/S1** (France 16.7, India 11.6, US 10.4; was 29.5).
  Validator PASS, sanity PASS (empty 0.056; France 0.055 / 3.28 matches per row). **Packaged as submissions/day2_best**.
- 01:41 crowd_eval started (A/B of crowded training, 120k S1, 3 folds).
- 02:08 **crowd A/B (dev/crowd_eval.py, 120k S1, 3 folds, exp06-era narrow blocking)**:
  model A (normal training) — normal val 0.9758, **crowded val 0.9710** (−0.0048: density shift hurts);
  model B (trained with +50% synthetic fake branches) — normal val 0.9743, **crowded val 0.9741** (+0.0031 vs A).
  Test is crowded (US +54%, IN +38% branch-like, France ≫) → **use crowd training**. B tunes stricter t1 (0.82 vs 0.76).
- 02:10 exp10 launched = exp09 + `--crowd 0.5` (test candidates reused from exp09 cache).
- 03:01 **exp10 OOF 0.9789** (IN .9756 US .9811) under CROWDED train conditions (+50% synthetic branches);
  recall after stage-1 0.9811 at 12 cand/S1. exp09 0.9795 was measured under normal conditions; the A/B implies an
  exp09-type model loses ~0.005 under crowding (→ ~0.975), so exp10 is expected ~+0.004 better on test.
- 03:05 queued exp11 (exp10 config, sample-seed 1, 400k S1) for a blend.
- 03:11 exp10 test written (11.9 cand/S1, validator+sanity PASS). vs exp09 on test: France +31k/−6k pairs,
  India +22k/−13k, US +20k/−17k. France dropped pairs are branch-like (house diff 58%, branch_sig 44%); added pairs
  are same-address (same house 76%) with a swapped generic word (e.g. "Forge Sport SARL" -> "Forge Parents SARL") —
  in train such swaps are 98% true matches. => France loss was partly RECALL; crowd training recovers it.
  **day2_best := exp10** (exp09 kept as submissions/day2_exp09_wide).
- 03:30 exp11 (exp10 config, seed+1, 400k S1) crowded OOF 0.9794 (IN .9761 US .9816).
- 03:42 blend 0.5·exp10 + 0.5·exp11 on the 54k common OOF S1: 0.97935 (exp10 0.97887, exp11 0.97907).
  Validator+sanity PASS. **day2_best := blend_10_11** (t1 .70 t2 .72 r 0; 11.9 cand/S1).
- 03:44 exp12 launched (seed+2, 400k S1, crowd 0.5) for a 3-way blend.
- (results appended below as they arrive)

---------------------------------------------------------------------------------------------------------------

## 10. Environment, rules and gotchas

- Apple M4 Pro, 12 cores, 24 GB RAM, no NVIDIA GPU. Python 3.12 venv `.venv/`. Libraries all permissive
  (pandas/numpy/sklearn/scipy BSD, lightgbm MIT, rapidfuzz MIT, joblib BSD, tqdm MPL-2.0/MIT, pyarrow Apache-2.0).
  **No unidecode (GPL)**, no external data, no network at runtime, no pretrained models used so far.
- Hand-written dictionaries: legal forms (US/IN/FR), name abbreviations, address abbreviations (EN/IN/FR), the Indic
  transliteration table. Learned from TRAIN only: transliteration dict, extra-token encoding. Unsupervised on each
  dataset's own text (incl. test): IDF / max_df in blocking, hmis branch-word rates, name-duplicate counts — document
  these in the methodology.
- zsh does not word-split `$var` — build commands explicitly (a make-split loop once created a split named
  "val --mode random --seed 42").
- Read TSVs with `io_utils.read_tsv` (tab, dtype=str, keep_default_na=False, QUOTE_NONE).
- The official validator must PASS before any upload; `sanity_check.py` must show France present.
- Git: commits use `-c user.name="Yash Aggarwal" -c user.email=...`; tags `sub-day1-N`. `dataset/`, `splits/`,
  `runs/`, `cache/`, `.venv/`, output TSVs are gitignored. The GitHub repo `yash4428/Amazon_ml` is PUBLIC — do not push
  the dataset or (during the contest) consider making it private before pushing code.
