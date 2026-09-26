# PROGRESS / HANDOVER — Amazon ML Challenge 2026: Business Entity Resolution

**Single source of truth for the team.** It records the current state, every finding (including dead ends), exact
numbers, how to reproduce every submitted file and what to do next. Read it top to bottom before changing code.
Rulebooks: `CLAUDE.md` (hard rules, metric, pipeline spec) and `START_HERE.md` (operating plan).
Last full rewrite: 26 Sep 2026 ~17:40 IST; §0/§1/§4/§6c/§8-§10/§12/§13 updated **27 Sep 00:40 IST**.

---------------------------------------------------------------------------------------------------------------

## 0. TL;DR — where we are (updated 27 Sep 00:40 IST)

| item | value |
|---|---|
| **Best public score** | **0.970385** — `submissions/day2_final_v2` (3-seed blend exp17/18/19 + robust France house rule + US shift rule) |
| **Upload next (day 3, slot 1)** | `submissions/day3_exp22_rules/matching_results.tsv` = **exp22** (reverse top-5 blocking, OOF **0.9833** vs 0.9815) + same rules. Expected ≈0.972-0.973 |
| Running (chained, local machine) | exp23 = exp22 + crowd 0.9 (started 00:20, ETA ~02:30) → exp22 seeds 4, 5 (`runs/exp22s4_test`, `exp22s5_test`, ETA ~04:30) |
| Leaderboard (27 Sep 00:00) | leader 0.9907, top-100 ≈ 0.985-0.986; we are ~600th |
| Honest local score (best model) | **exp22 OOF 0.9833** (India 0.9813, US 0.9846); exp17 0.98146 (same 800k-S1 sample, seed 3) |
| Public decomposition (exact, from probes) | Aprime: **US+India ≈ 0.9735, France ≈ 0.942** (singleton rate 0.0558 in every train country); C: US+IN ≈ 0.975 |
| Deadline | leaderboard closes **27 Sep 23:59 IST**; package (zip) ready by **27 Sep 18:00 IST** (`METHODOLOGY.md` = filled template draft) |

**The story so far.** Blocking + a LightGBM pair model + tuned decision rule gets ~0.98 locally, but every upload
scored 0.011-0.019 below local. Probes located the loss (France ≈0.94, US+India ≈0.975). Fake branches (a different
business with the S1's name and a shifted house number) explained part of it. Two label-free rules gave +0.0014 each:
the France house rule (French copies keep their house number) and the US shift rule (test has 4× the train rate of
"same street, number ±1..30" look-alikes). On 26 Sep night a **loss decomposition** (§6c) showed that 60% of our
local loss is RECALL (S1 with correct but incomplete match lists). Blocking alone lost 52k true pairs, 60% of them S1
whose name is shared by ≥3 S1, so their top-60 list floods with look-alikes. **Reverse top-K blocking** (each pool
record also keeps its 5 closest S1) fixed 30% of those misses: exp22 OOF 0.9815 → 0.9833 with 15% fewer candidates.
Test also has ≈1.9× the distractors per S1 of train (crowd 0.5 only simulates 1.5×) → exp23 trains at crowd 0.9.

**Day 3 plan (5 uploads).** Every upload must teach something AND be a candidate final:
1. `day3_exp22_rules` — public effect of reverse blocking vs C (0.970125 = same rules on exp17).
   ≥0.972 → exp22 is the new base. ≤0.9705 → build exp22 US+IN + day2_final_v2 France rows to separate countries.
2. `day3_exp23_rules` (after exp23 finishes: `postprocess.py --run runs/exp23_test --out submissions/day3_exp23_rules
   --house-rule --shift-rule US`) — public effect of test-like density (compare with slot 1).
3. Blend of the winner's seeds (exp22 + exp22s4 + exp22s5, or exp23 seeds if exp23 wins — then queue
   `--sample-seed 4/5` runs of exp23 in the morning, ~55 min each with blocking caches) via `dev/blend.py`
   (writes `test_scores.parquet` + params) → `postprocess.py --run <blend> --params <params.json> --house-rule --shift-rule US`.
4-5. Final + one spare (e.g. `--shift-keep-only` variant or a France-only probe). Package by 18:00 (§11).

---------------------------------------------------------------------------------------------------------------

## 1. How to run (from `student_resource/`)

```bash
# one-time environment (Python 3.12; the system python3 3.14 is too new for some wheels)
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt

# model run: train on dataset/train, score dataset/test  (~100 min cold, ~55-60 min with blocking caches)
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train \
    --test-dir dataset/test --out-dir runs/expNN_test --crowd 0.5 --sample-seed 3 --model-s1 800000 [--profile exp17]

# post-processing rules on a run's saved scores (minutes) — reproduces the submitted files
.venv/bin/python code/business_entity_resolution/src/postprocess.py --run runs/exp17_test --out output --house-rule
.venv/bin/python code/business_entity_resolution/src/postprocess.py --run runs/exp17_test --out output --house-rule --shift-rule US

# checks before ANY upload
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
.venv/bin/python code/business_entity_resolution/src/sanity_check.py --out-dir output --test-dir dataset/test
```

**Current best model (exp22) and the upload file for day 3 slot 1:**
`run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir runs/exp22_test --crowd 0.5 --sample-seed 3
--model-s1 800000 --profile exp17 --rev-k 5` (~2h25 cold: reverse blocking train 18 min + test 12 min extra) then
`postprocess.py --run runs/exp22_test --out submissions/day3_exp22_rules --house-rule --shift-rule US`.
`--rev-k K` = reverse top-K blocking (0 = off, the default in config). Blocking caches: `cache/blocks/rev_train_*`,
`rev_test_*` (reverse), `train_*`/`test_*` (forward + stage-1; the key includes rk and the stage-1 CONFIG but not the
stage-1 MODEL, so a second run with the same settings reuses the first run's pruned test candidates).

**Reproducing the public-0.969 file (Aprime):**
`run_pipeline.py ... --out-dir runs/exp17_test --crowd 0.5 --sample-seed 3 --model-s1 800000 --profile exp17`
then `postprocess.py --run runs/exp17_test --out <dir> --house-rule`. (`--profile exp17` restores exp17's settings:
learned xtok features kept, no count clipping, original postcode parser. `config.py` defaults are the exp21
settings.) Day2_C = same + `--shift-rule US`. `postprocess.py` was checked to reproduce both submitted files.

`run_pipeline.py` flags: `--mode lgbm|rule`, `--model-s1 N` (train S1 sampled for the pair model), `--sample-seed k`,
`--crowd f` (train-time synthetic fake branches, §6.3), `--s1-branches f` (augmentation v2, built, not yet evaluated),
`--oof-only`, `--cv group|country`, `--fixed-rounds R --params-from X/report/oof.json` (skip CV), `--profile`.

Outputs per run dir: `matching_results.tsv`, `candidate_pairs.tsv`, `run_info.json`, **`test_scores.parquet`**
(every scored test pair → re-decide/blend/post-process without re-running), `report/oof.json` (OOF score, tuned
params, per-country OOF), `report/oof_pairs.parquet`, `report/oof_s1.parquet`, `report/feature_importance.csv`.

Blends: `dev/blend.py runs/A runs/B [--w 0.5] [--params-from P] [--write runs/blend]` (evaluates on the S1 common to
both OOF samples); `dev/blend3_eval.py runs/A runs/B runs/C` (all subsets on the common S1).
Packaging a run: `dev/package.sh runs/X submissions/NAME "note"` (validator + sanity + copy + NOTE.md).

**Runtime / memory (Apple M4 Pro, 12 cores, 24 GB, no GPU):** normalisation cache for all 22M records 2.6 min;
train wide blocking + stage-1 ≈ 35-45 min; test ≈ 25 min; 800k-S1 model 5-fold CV ≈ 20 min; test scoring ≈ 15 min.
**Never run two full pipelines at once** (12-15 GB peak each). Caches (gitignored, safe to delete):
`cache/norm_s{1,2,3}.parquet`, `cache/blocks/*.parquet` (key = blocking config, NORM_VERSION, stage-1 config,
record ids). `runs/`, `splits/`, `cache/`, `dataset/`, `.venv/` and all output TSVs are gitignored — teammates must
regenerate runs locally.

---------------------------------------------------------------------------------------------------------------

## 2. Data facts

- Train: S1 2,206,820 (US 1.32M, India 0.88M) · S2 5.03M · S3 5.29M. Test: S1 1,732,543 (India 810k, US 663k,
  **France 259k = 14.98%**) · S2 4.89M · S3 5.08M. Test has ~24% more pool per S1 in every country (5.5-5.8 vs 4.7);
  the extra ~1 record/S1 is almost all LOW-similarity (claimed records with best sim ≥0.8/S1: US 1.66→1.78, IN 2.14→2.34).
- Singleton rate 0.056 (predict-all-empty floor). Matches per non-singleton S1 ≈ 3.66 (0:123k 1:119k 2:375k
  3:531k 4:484k 5:322k 6:165k 7+:87k). Copies per S1 by source: S2 {0:8%,1:38%,2:31%,3:16%,4+:7%}, S3 similar.
- **One-to-one is strict** (no S2/S3 id in two S1 lists) → greedy one-to-one decision.
- **All 7.64M true pairs share the country label** → hard-block by country (France is its own block).
- 26% of the train pool matches no S1 (distractors). Exact normalised-name equality in only 22% of true pairs.
- S1 never has an empty address; ~3% of S2/S3 do (14-17 empty-address pool records per 100 S1 in every country).
- No leakage: file order and id numbers are uncorrelated with matches (corr ≈ 0.000).
- **US addresses contain essentially no ZIP codes**; 5-digit numbers are house numbers (≈10% of US addresses).
- France: names are generic French words + legal form (SARL/SAS/EURL/SASU/SCI/SA/EI/SNC); few cities; very dense
  (16 candidates with blocking sim ≥0.5 per S1 vs ~5 US/IN); 12% of French S1 share an exact address with another S1
  (US/IN 4-5.6%); 10-20× more acronym records (9.2 vs 0.5-1.1 per 100 S1).

---------------------------------------------------------------------------------------------------------------

## 3. The data generator — what we reverse-engineered

**True copies (S2/S3 records of an S1)** are generated *independently* from the S1 with noise operations:
token shuffle/drop, added suffixes ([Inc], (Corp), Center, Services, Dr, Shri, The), legal-form changes, OCR typos
(Muri1lo, Humme1, 0↔o), injected accents, UPPERCASE (S2 style), domain/handle names (`revitup.com`, `@handle`),
acronyms (RC, PM), fully random made-up names at the exact address (Wexavi, Yumavera), native-script names for India
(Devanagari/Tamil/Telugu/Bengali/Gujarati/Odia; 18% of Indian partner names), empty address (~4% of true pairs),
"NULL"/"##" junk, component reordering, St/Street/**Saint**, state code↔name↔native script, city variants.
**House-number noise on true copies in US/India** (equal 70% of true pairs; digit drop/add 3432→432 4.6%;
zero-padding 0700; ranges 1976-1978; small shifts; unit numbers). **French copies carry almost no house-number noise**
(accepted digit-drop pairs 1.05 per 100 S1 in France vs ~20 in US/India; house-diff share of accepted pairs 4.4% vs
14.4% US / 19.7% IN).

**Fake branches ("sibling branches", the hard negatives)**: a *different* business record = the S1 name (often + a
branch word: holdings, group, ventures, east, riverside, downtown, infratech, exports…; in France Distribution,
Développement, Participations, Groupe, Holding, International) with the **house number shifted by ~1-30**
(30→34, 8520→8527, 46→53). They often come with their own noisy copies. 80% of singletons' top candidate is one.
In test there are more of them per S1 than in train (branch-like candidates per S1: US 1.02→1.58, IN 1.37→1.89,
France ≈9.2).

**Unresolvable ties**: an empty-address copy whose (core) name is shared by k S1 records belongs to one of them with
P≈1/k (exact name shared by 2 → P(true)=0.465, by 3 → 0.31, 4+ → 0.02). Raw-name (legal form/punctuation kept)
breaks 58% of answerable ties with 78% accuracy; the S1's other copies do NOT help (31-38% — copies are independent).

---------------------------------------------------------------------------------------------------------------

## 4. Pipeline (code/business_entity_resolution/src)

1. **normalize.py** — NFKD accent strip, `&`→and; Indic→Latin transliteration with ONE table (all Indic Unicode blocks
   share one 128-code-point layout); OCR digit fix; single-letter-run merge (L.L.C.→llc); abbreviation expansion;
   legal-form removal (US/IN/FR lists); DBA split; domain/@handle → compact name; address contraction
   (street/st/saint→st, rue→r…); postcode / house numbers / landmark flag. **Learned transliteration dict** from TRAIN
   true pairs (567 tokens: pharst→first, payoniyar→pioneer; native-pair token overlap 0.21→0.94).
   **`reparse_numbers`** (exp21 bug fix): postcode only when a comma component on its own, a trailing "STATE 12345",
   or a 6-digit PIN that is not the first number (the old rule threw away 5-digit US house numbers: 8.9% of US S1).
2. **blocking.py** — per country, sparse IDF-cosine top-K (chunked matmul in processes, per-row argpartition).
   Current: `combo_c` (name char-4-grams + address uni/bigrams) K=60 + `combo` (name + address words) K=30,
   max_df 20000; per-country post-hook. **Reverse top-K** (`reverse_candidates`, `--rev-k 5`, exp22): for every pool
   record its 5 most similar S1 per generator, computed against ALL S1 of the country, added before stage-1; a
   reverse-only pair gets its true cosine as `<g>_sim` and features `<g>_rev_rank` (99 = not in reverse top-K).
   Stage-1 training re-indexes the full-S1 reverse pairs to its 60k held-out S1 (`fit_stage1(..., rev)`).
3. **stage1.py** — cheap LightGBM (blocking sims/ranks, 3 rapidfuzz scores, house agreement, per-S1 gaps) trained on
   held-out train S1; keeps p ≥ 0.001, ≤ 20 per S1 → **~12 candidates per S1** (France 16.7). This is the set scored
   by the model and written to candidate_pairs.tsv (organiser email: smaller candidate sets rank higher).
4. **features.py** — ~95 features: name similarities (norm/core/compact/raw), acronym/DBA, address similarities,
   postcode state, house-number relation (equal, |diff|, relative diff, small shift ≤30, digit substring/Levenshtein,
   same length, branch signature, cluster size), learned extra-token encoding `xtok_*` (OOF target encoding —
   DROPPED in exp20/21, see §6.5), label-free branch-word detector `hmis_*` (rate at which an extra token co-occurs with
   a changed house number, computed on each dataset's own candidates), name/address duplicate counts (clipped at 10
   in exp20/21), context features: within-S1 rank/gap/n_close and **candidate-side** rank/gap/#S1 claiming the record
   for blocking sims and for name/compact/address/raw-name similarities (computed on the FULL candidate set).
   Augmentations: `add_synthetic_branches` (crowd v1: nudged copies of distractors), `add_s1_branches` (v2:
   S1 name ± branch word + shifted number; built, NOT evaluated).
5. **model.py** — LightGBM (lr 0.1, 127 leaves, deterministic), GroupKFold by S1 → OOF; final single model on all
   sampled pairs with rounds = 1.1 × mean best_iter.
6. **decide.py** — greedy one-to-one, keep top-1 if p ≥ t1, extras if p ≥ t2 and p ≥ r·best; (t1,t2,r) tuned on OOF
   macro-F0.5 by coordinate descent (typ. t1≈0.72-0.76, t2≈0.70-0.74, r=0).
7. **postprocess.py** — the France house rule and US shift rule (§6.6-6.7). **sanity_check.py** — row counts,
   per-country empty rate, France present, matches ⊆ candidates. **run_pipeline.py** — orchestration.

---------------------------------------------------------------------------------------------------------------

## 5. How to evaluate (learned the hard way — read this)

5.1 **Never use `val`/`val2`** (`evaluate.py make-split --mode random`): distractors are assigned randomly, so ~80% of
a val S1's fake branches land on the train side → optimistic (rule baseline: val 0.773 vs public 0.679).
5.2 **Honest local metric = OOF macro-F0.5 on the FULL `dataset/train`** (all distractors present). Compare models on
the SAME `--sample-seed`/`--model-s1` (identical S1 sample) or with `dev/blend.py` / `blend3_eval.py` on common S1.
5.3 Simulators: `dev/country_cv.py` (train one country, predict the other), `dev/crowd_eval.py [--country-folds]`
(normal vs crowd-trained model on normal vs crowded validation), `dev/adversarial.py` (train-vs-test classifier per
country; its top features = what shifted). **The country simulator missed the French problem** because US and
India share English branch words — adversarial validation found it.
5.4 **Probes and exact attribution** (the most useful tool we found). F0.5 is per S1 and blocking/one-to-one never
cross countries, so rows of different countries are independent:
- France-empty probe P of a file S: US+India F = (P − 0.1498·0.05)/0.8502; France F = (S − P)/0.1498 + 0.05.
- To measure a change in one country exactly: keep the other countries' rows identical to an already-scored file.
- Probes so far: exp04 → US+IN 0.969, France 0.877. exp17 → **US+IN 0.9745**; Aprime → **France 0.938**.
5.5 Local→public gap history: 0.9736→0.955, 0.9769→0.963, 0.9773→0.964, ~0.9804→0.966. Constant ~0.013-0.019 =
France deficit + US fake-branch acceptances.

---------------------------------------------------------------------------------------------------------------

## 6. Findings (most important first; all with evidence)

6.1 **LightGBM ≫ rule** (0.679 → 0.955). Candidate-side context ("does another S1 fit this record better") is the top
feature family.
6.2 **Blocking**: wide blocking + stage-1 filter beats narrow blocking on both recall and size — train recall after
stage-1 0.9819 at ~11 cand/S1 vs 0.9756 at 30/S1 (exp09 OOF 0.9795 vs exp06 0.9771). Table (30k train S1):
combo20+c4_10 0.9735@26.6 · combo_c20+combo10 0.9770@22.5 · combo_c25+combo15 0.9813@30 · union K60/30 0.9888@72 ·
stage-1 on that union p≥0.001 → 0.9842@10.0 (oracle 0.9952).
6.3 **Crowd training** (+50% synthetic fake branches in the train pool): A/B on crowded validation +0.0031 (seen
countries) and +0.0017 (unseen country); normal model loses 0.0048 under crowding. Used since exp10.
6.4 **More data + bagging**: 800k S1 > 300-400k (logloss 0.027→0.025); 3-seed blends +0.0003-0.0004 on common S1.
6.5 **France, learned branch words are blind** (adversarial AUC 0.998): `xtok_sum` train 2.14 vs France 0.36. exp17
accepted **23,364 French pairs whose candidate ADDS a French branch word (9 per 100 S1) vs 0 in US / 20 in India**
("HM Residence SAS | 30 Rue des Lilas" ← "HM RÉSIDENCE DÉVELOPPEMENT SAS | NO 32"). Dropping xtok (exp20) fixed that
(23,364 → 839) BUT also dropped 59,868 French pairs of which 67% had the SAME house number (swapped generic word:
"4l Ecole SARL" → "4l Amicale Sarl") because the label-free hmis detector is polluted on dense French streets
("club" 0.74, "nantes" 0.84) → exp20 France predictions NOT used (idea "A", never submitted).
6.6 **France house rule (SUBMITTED, +0.003 public)**: French true copies keep their house number, so French accepted
pairs whose first house number differs are fake branches. Aprime = exp17 minus 39,575 such pairs (2,044 French S1
become empty; French empty rate 5.0% → 5.8%, matching other countries) → **public 0.969, France 0.925→0.938**.
6.7 **US shifted numbers (TESTING with day2_C)**: with corrected parsing, accepted "same street, number ±1..30" pairs
per 100 S1: US train 3.68 (precision 0.978) vs **US test 9.89**; India 5.87 vs 6.88. The ~6 extra per 100 US S1 would
explain US+IN 0.9745 vs local ~0.981 if they are fake branches — but in train such pairs are true 70% (US) / 91% (IN)
of the time, so only a submission can decide. The 5-digit parsing fix did not change the test count (9.79→9.89).
6.8 **5-digit US house numbers were parsed as postcodes** (bug; 8.9% of US S1 / 6.3% of US pool had no house number)
→ fixed in exp21 (OOF US .9832→.9838).
6.9 **Empty-address copies are the biggest local miss** (recall 0.544 vs 0.99 for every other record type; 3.9% of true
pairs; 64% of model misses) and are mostly unresolvable ties (§3). Raw-name features (exp17) break some ties.
6.10 **House-number EDA** (train): digit drop/add is TRUE 46% of the time (4.6% of true pairs); same-length shift ≤30
is true 6.7%; one digit changed 40%. exp17's digit-level features cut false merges by 18% (11,597→9,500 FP pairs).
6.11 **Size-dependent counts shift** (US test has half the S1 of US train): name-dup mean 35→19 → clipped at 10 (exp20).
6.12 India: learned transliteration fixed native-script names (India OOF 0.954→0.966, exp03→exp04).
6.13 France checks that were NOT the problem: acronyms (95.5% accepted when they equal the S1 initials), co-located
S1 (normal behaviour), name sharing (FR 52% ≈ IN 53%), rejected copy-like records (≤ +0.002 even if all accepted),
blocking-score saturation (no cross-S1 ties), acronym density.

---------------------------------------------------------------------------------------------------------------

## 6b. Blind EDA (26 Sep evening) — fresh look at the raw data (dev/eda_fingerprint.py, eda_blind2.py, eda_shifted.py)

- **Shifted look-alikes** (same street, house number ±1..30, name token-set ≥90), per 100 S1, ALL candidates:
  train US 8.72 (13% carry an extra word; P(true) ≈0.42) vs **test US 35.76 (4.1×; 22% extra word; model accepts
  24.5% ≈ 8.8/100)**; train India 5.13 (P(true) ≈0.81) vs test India 10.25 (2×; accepted 48%); **test France 43.88**.
  ⇒ test has many more fake branches that differ only by the house number, mostly WITHOUT a branch word. At the train
  rate only ~3.6/100 US S1 are true shifted copies → ≈5 wrong merges per 100 US S1. This is the evidence behind
  day2_C (expected ≈ +0.003 overall). India excess ≈1 wrong merge/100 S1 (marginal); France handled by the house rule.
- **Record-format fingerprint** (owned-by-some-S1 vs distractor from the raw record only): AUC 0.78. Empty-address
  records are ~97.7% owned by SOME S1 (4.5% of copies vs 0.3% of distractors); domain names similar (5% vs 0.6%);
  distractors have longer names (4.0 vs 3.3 words — the appended branch word). Already captured by the model.
- Exact duplicate pool records (0.9% of pool): 99.8% share their owner; OOF splits a twin pair only 6 times → nothing.
- **Singletons are unpredictable from the S1 record** (AUC 0.500; singleton rate 0.056 for unique and shared names).
- **No train/test overlap**: 0 identical S1 or pool records across splits (33.8% of test S1 names occur in train S1
  names — generic names, different businesses).

## 6c. Where the loss is (26 Sep night) — loss decomposition, test density, reverse blocking

**OOF loss decomposition** (exp17, 800k S1, loss 0.0185 = 1 − 0.9815): S1 with correct matches but **missing
copies 0.0112 (60%)** · non-singleton predicted empty 0.0038 · extra wrong matches 0.0019 · singleton given a match
0.0012 · FP+FN 0.0004. → our weakness is **recall**, not precision. Missed true pairs (125k): blocking/stage-1
51.9k (gain if all recovered +0.0065), model misses of empty-address copies 48.4k (+0.0056), other model misses
25.1k (+0.0037).
- **Empty-address copies are calibrated and mostly irreducible**: exact name_core shared by k S1 → P(true) 0.97 (k=1,
  we accept 97.4%), 0.47 (k=2), 0.31 (k=3), 0.02 (k≥4); the model's mean score equals P(true) in every k bucket.
- **Blocking misses**: 60% are S1 whose exact name is shared by ≥3 S1 (their top-60 list floods with look-alikes).
  Of the 51.9k: 23% were in forward top-K and dropped by stage-1; of the rest, the S1 is in the pool record's own
  **reverse** top-1/3/5/10 in 19%/29%/34%/42% of cases (8% share no blocking feature at all).
  ⇒ **reverse top-K blocking** (`--rev-k 5`: each pool record also keeps its 5 most similar S1, per generator;
  `blocking.reverse_candidates`, rev ranks become features `<g>_rev_rank`) — **exp22: raw blocking recall 0.9852→0.9905,
  after stage-1 0.9813→0.9869 with 15% FEWER pairs (10.2 vs 12.0 cand/S1); OOF 0.98146→0.9833 (India +0.0025, US
  +0.0013); blocking-lost true pairs 51.9k→36.2k; FP pairs 9.5k→9.8k.** Cost: +18 min train / +12 min test blocking.
- Biggest model-miss slice: same name + same street + different house number (P(true) 0.455; model already
  separates it: 2% of true missed, 0.4% of false accepted).
- Train singleton rate is 0.0558 in BOTH US and India (generator constant) → the France decomposition holds:
  Aprime France ≈ 0.942, US+IN ≈ 0.9735; C → US+IN ≈ 0.975.

**Test density**: pool records per S1 — train 4.68 (US/IN), **test 5.76 US / 5.82 IN / 5.53 FR**. With 3.46 copies
per S1, test has ≈2.3 distractors per S1 vs 1.22 in train (**≈1.9×**; crowd 0.5 only simulates 1.5×). The extra
test records have the same max-similarity profile as ordinary train distractors (mostly combo_c 0.4-0.8), i.e. test ≈
train with doubled distractors ⇒ **exp23 = exp22 + crowd 0.9**.
- Label-free calibration check (sum of scores per S1 = 3.393 vs 3.396 true in OOF): test US 3.577, France 3.680,
  India 3.421. Mid-score mass (0.5-0.9) per 100 S1: OOF 7.9, US 16.6, India 11.0, France 28.6. After C's rules the US
  accepted-score histogram matches OOF closely; France has fewer top-bin (≥0.99) extras (209 vs 218 per 100 S1).
- France: rejected same-number/same-street/similar-name candidates are only 1.36 per 100 French S1 (≤ +0.0002
  overall even if all true). French names follow "<City> <Type> <Legal form>" ("Lille Fetes SARL"), very
  repetitive; pool records selected by no S1: FR 0.38, IN 0.40, US 0.26 per S1 — some are clear true copies
  ("Hospitalier Centre Saint JEAN | 63 Bd. Piere De…, Nantes" of S1 "Centre Hospitalier Saint Jean | 63 Boulevard
  Pierre de Coubertin, Nantes") → reverse blocking.
- French branch words (label-free: enrichment among shifted-number vs same-number accepted pairs): snc 378×,
  participations 155×, groupe 13×, developpement 11×, sa 9×, center 9×, sasu/sas/sarl/sci/eurl 4-6×, france 3.6×;
  noise suffixes on copies (ratio < 1): fils, et/and, compagnie, services, associes, dba/aka/formerly.
  Same-number and no-number accepted pairs carry them at the same ~1% rate → nothing left to remove there.

**Robust France house rule** (`postprocess.py`, now default): the house rule no longer drops pairs whose *street*
number (number right before a street word) is equal although the first parsed numbers differ (postcode / appt /
reordering artefacts): 1,613-1,623 French pairs restored.

## 7. Experiments (local = honest OOF on full train; "crowded" = train pool with synthetic branches)

| tag | change | local OOF | IN / US | public |
|---|---|---|---|---|
| exp01 | rule: max(combo, 0.8·name_c4), one-to-one | tune-OOF 0.680 | loco .694/.695 | **0.679** |
| exp02 | LightGBM 53 feats, 400k S1 | 0.9622 (val-train) | – | – |
| exp03 | + xtok + house cluster feats (smoke) | 0.965-0.967 | .954/.976 | – |
| exp04 | + learned translit + name-dup, 250k S1 | 0.9736 | .9662/.9785 | **0.955** |
| exp05 | blocking combo_c25+combo15, hmis, 300k S1 | 0.9769 | .9716/.9805 | **0.963** |
| exp06 | + branch-signature, − length feats | 0.9771 | .9719/.9805 | – |
| exp06+07 blend | 0.5/0.5 (exp07 = 600k S1, no CV) | ≈0.9773 | – | **0.964** |
| exp09 | wide blocking K60/30 + stage-1 (~12 cand/S1) | 0.9795 | .9763/.9817 | – |
| exp10-12 | exp09 + crowd 0.5 (300-400k S1, seeds 0-2) | 0.9789-0.9794 (crowded) | – | – |
| exp13-15 | crowd 0.5, 800k S1, seeds 3/4/5 | 0.9804 / 0.9800 / 0.9802 | .9776/.9822 (13) | blend 13/14/15 **0.966** |
| exp16 | + cand-side name/address competition, addr dup counts | 0.9807 (seed 3) | .9781/.9824 | – |
| exp17 | + raw-name features + digit-level house relation | **0.98146** (seed 3) | .9788/.9832 | via Aprime **0.969** |
| exp18/19 | exp17 features, seeds 4/5 | 0.9812 / 0.9814 | – | blend 17/18/19 = 0.98145 (105k common S1) |
| exp20 | exp17 − xtok, counts clipped at 10 | 0.9813 (seed 3) | .9787/.9831 | not submitted (France recall loss) |
| exp21 | exp20 + 5-digit house-number parsing fix | **0.9817** (seed 3) | .9785/.9838 | – |
| **exp22** | **exp17 + reverse top-5 blocking** (`--rev-k 5`, profile exp17, seed 3) | **0.9833** | **.9813/.9846** | pending |

Other results: expected-F0.5 decision 0.9763 < tuned thresholds 0.9771 (rejected). Stricter thresholds for an
unseen country: +0.002 US / 0 India (rejected). Unseen-country simulator: all 0.9608, no_xtok 0.9609, no_dup 0.9588,
no_cand_ctx 0.9570, no_len 0.9626 (length feats removed).

---------------------------------------------------------------------------------------------------------------

## 8. Submissions (5 per day; public LB = subset of test)

| day/slot | folder | content | public | what we learned |
|---|---|---|---|---|
| 1/1 | day1_1 | exp01 rule | 0.679 | val split optimistic |
| 1/2 | day1_2 | exp04 | 0.955 | |
| 1/3 | day1_3_probe | exp04, France emptied | 0.831 | US+IN 0.969, France 0.877 |
| 1/4 | day1_4 | exp05 | 0.963 | |
| 1/5 | day1_5_final | 0.5·exp06 + 0.5·exp07 | 0.964 | |
| 2/1 | day2_best | blend exp13/14/15 | 0.966 | |
| 2/2 | day2_exp17_probe_france | exp17, France emptied | 0.836 | **US+IN 0.9745** |
| 2/3 | day2_Aprime | exp17 − French house-diff pairs | **0.969** | **France 0.938** |
| 2/4 | day2_C_us_shift | Aprime − 64,911 US shifted-number pairs | **0.970125** | US +0.0036 (Aprime 0.968745) |
| 2/5 | day2_final_v2 | 3-seed blend 17/18/19 + robust France house rule + US shift rule | **0.970385** | +0.00026 vs C: blend + France fix transfer, small |
| 3/1 | day3_exp22_rules | exp22 (reverse top-5 blocking) + robust house rule + US shift | *to upload* | reverse-blocking effect vs C |

Built but not submitted: day2_exp17 (exp17 alone), day2_A_exp17usin_exp20fr, day2_blend_17_18_19, day2_exp13,
day2_blend_13_14, day2_blend3, day2_blend_10_11, day2_exp10_crowd, day2_exp09_wide, day2_final, day2_probe_france.
`submissions/log.md` has the running log; every folder has a NOTE.md.

---------------------------------------------------------------------------------------------------------------

## 9. Reading the day-2 results (done) and what they mean

- **C (0.970125) − Aprime (0.968745) = +0.00138 overall = +0.0036 on US** (US = 38.3% of S1): the US shifted-number
  pairs were mostly fake branches, but only half of the +0.003 we expected → some were true copies with house noise.
  India version of the rule NOT tried: in India test the excess is only ≈1 per 100 S1 and train P(true) of such pairs
  is 0.91 → expected negative.
- **day2_final_v2 (0.970385) − C = +0.00026**: 3-seed blend (+0.0003-0.0004 on OOF) + robust France house rule
  (+1,613 French pairs, expected +0.0001) both transfer with the right sign, but small. Seed blending is worth doing for
  the final only if compute allows.
- Public vs local: exp17 OOF 0.98146 → C public 0.970125. The gap (0.011) = France (≈0.94, 15% of S1 → 0.006) +
  US+IN (≈0.975 vs 0.981 → 0.005), the latter consistent with test's doubled distractor density.

---------------------------------------------------------------------------------------------------------------

## 10. Next steps (ranked, 27 Sep)

1. Upload `day3_exp22_rules`; if it wins, exp22 (or exp23) is the base of the final (see the §0 day-3 plan).
2. exp23 (crowd 0.9) → `day3_exp23_rules`. Its OOF is on a denser pool and is NOT comparable with exp22's OOF; only
   the public score decides. If exp23 wins, run `--sample-seed 4` / `5` of exp23 for the blend.
3. Remaining recall levers (measured on exp22 OOF, `dev/eda_loss_decomp.py runs/exp22_test`): still 36.2k true pairs
   lost at blocking/stage-1 (22.7k non-empty address), 24.8k non-empty model misses, 49.3k empty-address misses
   (mostly unresolvable name ties). Ideas: reverse top-10 (rank <10 covers 42% vs 34% for <5 of the misses that are not
   in forward top-K), a looser stage-1 (23% of blocking-category misses were in forward top-K and cut by stage-1),
   an exact-address key generator for made-up names at the exact address.
4. France (≈0.94) is the biggest per-country hole; label-free checks found nothing big left in precision (§6c).
   Reverse blocking changes 13% of French rows (+27.5k / −10.5k pairs) — the day3 slot-1 result tells whether that
   helps France; to isolate France, combine exp22 US+IN rows with day2_final_v2 France rows (exact attribution, §5.4).
5. Make the rules model-side (`--s1-branches`) — only if time allows; the post-rules are documented and data-driven.
6. **Do NOT**: tune on the public LB beyond measured, hypothesis-driven steps; hand-label test; pseudo-label test
   without asking the organisers (rule 7); add models/dependencies without asking Yash; run two pipelines at once.

---------------------------------------------------------------------------------------------------------------

## 11. Final package checklist (Day 3, by 18:00 IST)

- `output/matching_results.tsv` + `output/candidate_pairs.tsv` regenerated from code (run_pipeline + postprocess),
  validator PASS, sanity PASS (France present, empty rates sane), matches ⊆ candidates.
- `code/business_entity_resolution/{src/, README.md, requirements.txt}` — README must give the exact commands for the
  chosen final (profile, seeds, postprocess flags) and runtimes.
- `Documentation_template.md` filled: use §2-§8 of this file (data facts, generator, blocking table incl. stage-1
  numbers and candidate size ~12/S1, features, model/CV, decision, post-rules with their measured gains, France/
  unseen-country handling, experiment table, licences, "no external data / no network" statement, hand-written
  dictionaries list, what was learned from train only vs unsupervised on test text).
- Clean-room test: fresh venv, `pip install -r requirements.txt`, run the README commands, diff against `output/`.
- Zip `<team_name>_submission.zip` (ask Yash for the team name).

---------------------------------------------------------------------------------------------------------------

## 12. Files and scripts

- `code/business_entity_resolution/src/` — pipeline (§4). `dev/` — analysis tools:
  `block_dev.py` (blocking table, `--crowd`), `stage1_dev*.py`, `country_cv.py`, `country_params.py`,
  `crowd_eval.py`, `adversarial.py`, `blend.py`, `blend3_eval.py`, `expf.py`, `package.sh`, and the EDA scripts
  `eda_types.py` (record types & recall), `eda_empty.py` (empty-address ties), `eda_house.py` (house-number relations),
  `eda_acronym.py`, `eda_france.py`, `eda_rejected.py`, `eda_sibling.py`, `eda_accepted.py`,
  `eda_branch_cluster.py`, `eda_street_number.py`, blind EDA `eda_fingerprint.py`, `eda_blind2.py`, `eda_shifted.py`,
  `eda_shift_only.py`, `eda_exact_addr.py`, `eda_fr_exact.py`, `eda_block_miss.py`, `eda_plan_checks.py`; 26 Sep night (§6c):
  `eda_loss_decomp.py [runs/X]` (OOF loss by error type, FN/FP causes), `eda_fn_look.py` (gain if each FN type were
  recovered + examples), `eda_reverse_rank.py` (forward/reverse rank of blocking misses), `eda_empty_owner.py`,
  `eda_empty_fn.py` (empty-address ties), `eda_slices.py` (FN/FP by name/address/house slice), `eda_fr_added.py`,
  `eda_fr_branchwords.py`, `eda_fr_rejected.py`, `eda_orphans.py` (pool records nobody retrieves). Outputs go to `runs/eda/`.
- **`run_meta/`** (in git) — small copies of every run's `oof.json` (OOF score, tuned params, per-country), `feature_importance.csv`,
  `run_info.json`, all run logs (`<exp>_test.log`), blend params (`blend_*_params.json`) and chain scripts.
- `runs/<exp>_test/` (local only) — outputs + report per run; `runs/*.log` — run logs; `submissions/<name>/` — files +
  NOTE.md; `reports/` — early evaluate.py dumps; `experiments.csv` — evaluate.py log (val-split era).

---------------------------------------------------------------------------------------------------------------

## 13. Environment, rules and gotchas

- Apple M4 Pro, 12 cores, 24 GB RAM, no NVIDIA GPU (MPS only). Python 3.12 venv `.venv/`. Libraries (all
  permissive): pandas/numpy/scikit-learn/scipy BSD, lightgbm MIT, rapidfuzz MIT, joblib BSD, tqdm MPL-2.0/MIT,
  pyarrow Apache-2.0. **No unidecode (GPL)**, no external data, no network at runtime, **no pretrained models**.
- Hand-written: legal forms (US/IN/FR), name and address abbreviations (EN/IN/FR), the Indic transliteration table,
  fake-branch words for `add_s1_branches`. Learned from TRAIN labels only: transliteration dict, xtok encoding, all
  models and thresholds. Unsupervised on each dataset's own text (incl. test): IDF/max_df, hmis rates, duplicate
  counts, the France house-rule noise statistic (label-free). Document all of these in the methodology.
- Organiser email (25 Sep): candidate_pairs.tsv is part of the final submission; **smaller candidate sets per S1 rank
  higher** beyond the LB → keep the stage-1 filter (~12/S1).
- zsh does not word-split `$var` (a loop once created a split named "val --mode random --seed 42").
- Read TSVs only with `io_utils.read_tsv` (tab, dtype=str, keep_default_na=False, QUOTE_NONE).
- Git: tags `sub-day1-N`. GitHub: `https://github.com/yash4428/Amazon_ml` (pushed 27 Sep 00:45; PUBLIC until Yash
  makes it private). The dataset, `runs/`, `cache/`, all TSVs (incl. the early `reports/*.tsv` error dumps, removed
  from history because they exceed GitHub's 100 MB limit and contain training records) are NOT in git: get
  `dataset/` from the official student_resource zip; regenerate runs locally.
- Background chains: `runs/chain_exp23.sh`, `runs/chain_exp22_seeds.sh` (wait for the previous job, then run).
  Check with `pgrep -fl run_pipeline` and `tail runs/<exp>_test.log`.
