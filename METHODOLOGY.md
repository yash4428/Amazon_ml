# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TEAM NAME]
**Team Members:** [TEAM MEMBERS]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Our pipeline blocks candidates per country with sparse IDF-cosine retrieval (forward top-K per S1 **plus reverse top-K
per S2/S3 record**), filters them with a cheap LightGBM (~10-13 candidates per S1), and scores pairs with a LightGBM
over ~100 string, address, house-number, branch-signature and **candidate-side competition** features. The model is
trained at test-like distractor density (synthetic fake branches) and averaged over 3 seeds. A decision step
(strict one-to-one + thresholds tuned for macro F0.5 on out-of-fold predictions) turns scores into matches. Label-free,
data-driven post-rules then remove "sibling / fake-branch" look-alikes, which the unseen country France exposed.
The best leaderboard file (0.989378) applies these rules to a teammate's model output (see §5.3). The fully
reproducible pipeline alone scores 0.972832.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA)

- Train: 2.21M S1 (US 1.32M, India 0.88M), 10.3M S2+S3. Test: 1.73M S1 (India 810k, US 663k, **France 259k**).
- Singleton rate 0.0558 in every train country (a generator constant); 3.46 copies per S1 on average (0-11).
- **One-to-one is strict** and **all true pairs share the country label**, so blocking is by country (France is simply
  its own block) and the decision uses greedy one-to-one assignment.
- Test has **~1.9× the distractors per S1** of train (5.5-5.8 pool records per S1 vs 4.68).
- Noise on copies: token shuffle/drop, added suffixes (Inc, Corp, Center, Services, Shri; French "et Fils", "& Cie",
  "& Associés", "France", "Développement"), legal-form changes, OCR typos, accents, UPPERCASE, domain names, acronyms,
  made-up names at the exact address, native Indic scripts, empty or junk addresses, reordered components. US/India
  copies carry house-number noise (digit drop, zero padding, small shifts); **French copies keep their street number**.
- Hard negatives: **fake branches** (S1 name ± branch word, house number shifted by 1-30) and, in France, **sibling
  businesses at the same address** whose name differs only by the business-type word ("Campagne Comite SAS" vs
  "Campagne Sportive SAS"). French names follow "<City/Word> <Type> <LegalForm>", so a sibling shares everything but
  one word.
- Empty-address copies whose exact name is shared by k S1 are unresolvable ties (P(true) 0.97 / 0.47 / 0.31 / 0.02 for
  k = 1 / 2 / 3 / ≥4); our model is calibrated on every bucket.

### 2.2 Solution Strategy

**Approach Type:** Blocking (forward + reverse) + two-stage gradient boosting + optimised decision step + label-free
post-rules.
**Core Innovation:** (1) reverse top-K blocking for names shared by many S1; (2) candidate-side competition features;
(3) training at test-like density; (4) label-free, country-agnostic switches for post-rules. Each rule measures its own
trigger statistic on the test predictions of each country, so an unseen country (France) gets exactly the rules its
data calls for.

---

## 3. Candidate Generation (Blocking)

- **Normalisation** (hand-written dictionaries from general language knowledge only): NFKD accent stripping; `&`→and;
  multi-country legal forms (inc, llc, ltd, pvt, private, sarl, sas, sasu, eurl, sci, snc, sa, …); name and address
  abbreviation maps (EN/IN/FR); rule-based transliteration of Indic scripts; a romanisation dictionary learned from
  training pairs; regex postcode / house-number parsing and a street-number parser (number before a street word).
- **Generators** (per country, IDF weights computed on that country's records, unsupervised):
  `combo_c` = name char-4-grams + address word uni/bigrams, top-60 per S1; `combo` = name + address words, top-30;
  **reverse top-5** per pool record in both spaces (computed against all S1 of the country).
- **Stage-1 filter:** 250-tree LightGBM on cheap features (cosines, forward and reverse ranks, 3 RapidFuzz scores,
  house agreement, per-S1 gaps), trained on held-out train S1; keeps p ≥ 0.001 (≤ 20 per S1). Its output is exactly the
  scored set = `candidate_pairs.tsv` (test: **18.4M pairs, 10.6 per S1**).

| configuration (train) | pair recall | cand/S1 |
|---|---|---|
| narrow combo_c@25 + combo@15 | 0.9813 | 30.0 |
| wide combo_c@60 + combo@30 (raw union) | 0.9852 | 70 |
| wide + stage-1 filter (exp17) | 0.9813 | 12.0 |
| **wide + reverse top-5 (raw union)** | **0.9905** | 87 |
| **wide + reverse top-5 + stage-1 (final)** | **0.9869** | **10.2** |

Reverse blocking removed 30% of the true pairs previously lost at blocking (51.9k → 36.2k on the 800k-S1 sample) while
the final candidate set got 15% smaller.

---

## 4. Matching Model

**Features (~100):**
- Name: RapidFuzz ratio / partial / token-sort / token-set / Jaro-Winkler on normalised, core, compact and raw names;
  IDF-weighted token overlap; acronym, DBA and first-token matches; transliteration-aware similarity; learned extra-token
  (branch-word) encoding; label-free branch-word detector.
- Address: token-set / Jaccard / char similarity; postcode state; house-number overlap and **digit-level relations**
  (difference, digit substring, digit Levenshtein, same length, small offset); address-without-numbers similarity;
  **branch signature** (same street, number shifted by 1-30).
- Context / competition: rank and gap of the pair among the S1's candidates and **among all S1 that retrieved the
  record**, #S1 competing for the record, house-number cluster size, name-duplicate counts, blocking provenance
  (forward and reverse ranks).
- No raw country value is used as a feature.

**Top features (gain):** candidate-side combo_c rank, learned branch-word sum, candidate-side combo_c gap, small
house offset, branch-word max, house cluster size, house overlap, address token-set.

**Model:** LightGBM (learning rate 0.1, 127 leaves), 800k train S1 (~8.5M pairs) with the train pool crowded by +90%
synthetic fake branches (test-like density), 5-fold GroupKFold by S1 for out-of-fold probabilities, final model on all
sampled pairs. **3 seeds** (different S1 samples) averaged.
**Decision:** greedy one-to-one; accept the top candidate if p ≥ t1 and extras if p ≥ t2 and p ≥ r·best; (t1, t2, r)
tuned by coordinate search to maximise **macro F0.5 per S1 on OOF predictions** → (0.74, 0.02, 0.75).

**Post-rules (`postprocess.py`, label-free switches computed per country on the test predictions):**

| rule | switch statistic (per 100 S1) | applied to | public gain (measured, France-only changes) |
|---|---|---|---|
| **House rule**: drop pairs whose street number differs | accepted digit-drop pairs < 3 (FR 1.1 vs US/IN ~20) | France | France +0.013 |
| **US shift rule**: drop same-street pairs with number shifted 1-30 | test/train excess of shifted look-alikes | US | US +0.0036 |
| **Word-swap rule**: candidate replaces one frequent word by another | accepted swaps ≥ 1 (FR 10.3 vs US 0.03 / IN 0.05); train true rate of such candidates 0.001 | France | France +0.0149 (our model), +0.0251 (teammate model) |
| **Type-swap rule**: an in-place business-type word replaced by another (type words: frequent in S1 names, pool/S1 share ratio < 1.3) | accepted in-place type swaps ≥ 1 (FR 1.7-2.8 vs US/IN 0.2-0.3) | France | France +0.0050 |

---

## 5. Results & Error Analysis

### 5.1 Scores
- Honest out-of-fold macro F0.5 on train: **0.9840** (3-seed blend; single model 0.9838; India 0.9818, US 0.9852).
- Public leaderboard: 0.679 (rules) → 0.955 → 0.963 → 0.964 → 0.966 → 0.9687 → 0.9701 → 0.9704 → 0.9706 →
  **0.9728 (fully reproducible pipeline)** → 0.9848 (teammate model) → 0.9886 → 0.9893 → **0.989378 (best)**.

### 5.2 Error analysis
- **Loss decomposition (OOF):** 60% of local loss is recall (S1 with correct but incomplete lists); blocking lost 2% of
  true pairs before reverse blocking (1.3% after); empty-address name ties are largely unresolvable.
- **Common false positives:** fake branches with shifted house numbers; French sibling businesses at the same address
  differing only by the type word; empty-address records whose name is shared by several S1.
- **Common false negatives:** empty-address copies of shared names; copies with changed house numbers (digit drop /
  small shift) that look like fake branches; made-up or acronym names at the exact address; native-script names.
- **Unseen country:** probe submissions with one country's rows changed at a time gave exact per-country effects
  (France ≈ 0.94 before the France rules, ≈ 0.96+ after). The France-specific errors were invisible to the model
  trained on US/India and were found by comparing, per name-relation category, how often pairs are accepted in France
  vs US/India vs train.

### 5.3 Teammate model and the best file
A teammate's independent pipeline scored 0.984833. Its source code was lost, but its output file is kept as an input.
The best file = that output + our house, word-swap and type-swap rules (France-only changes) + 8,605 of our own
confident pairs (blend score ≥ 0.99, same / missing house number) that it missed (mostly native-script Indian copies
and French acronym / made-up-name copies at the exact address). `dev/apply_rules_to_file.py` and `dev/final_adds.py`
regenerate it from that file and our run outputs. A last attempt to also remove low-confidence shifted-number pairs from
it lowered the score (0.988827): those pairs were mostly true copies.

---

## 6. Conclusion

Blocking, context features and a tuned decision step gave a strong base (~0.98 locally). The large leaderboard gains
came from the data itself: the generator's fake-branch mechanism, the country-specific house-number behaviour and,
above all, France's sibling businesses that differ by one type word. Probe submissions changing one country at a time
made every rule measurable. Lessons: validate at test-like density, attribute public scores per country, and look for
error categories that one country accepts far more often than the others.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `README.md` (setup + commands), `requirements.txt` (pinned), `final_params.json`,
`src/` (`config.py`, `io_utils.py`, `normalize.py`, `blocking.py`, `stage1.py`, `features.py`, `model.py`, `decide.py`,
`postprocess.py`, `run_pipeline.py`, `sanity_check.py`), `dev/` (analysis and the best-file scripts).

### B. Licences and data
Python 3.12; numpy, pandas, scipy, pyarrow (BSD / Apache-2.0), scikit-learn (BSD-3), LightGBM (MIT), RapidFuzz (MIT),
joblib (BSD-3), tqdm (MIT/MPL). No pretrained models, no GPL libraries (accent stripping via `unicodedata`).
**No external data, lookups or APIs.** All dictionaries are hand-written or learned from the provided training labels.
Unlabelled test text is used only for unsupervised statistics (IDF weights, duplicate counts, the post-rule switch
statistics and the per-country type-word vocabulary).
