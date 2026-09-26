# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [TBD]
**Team Members:** [TBD]
**Submission Date:** 27 September 2026

> Draft (26 Sep night). Items marked [TBD] are filled in once the final configuration is chosen on 27 Sep.

---

## 1. Executive Summary

Country-blocked sparse IDF-cosine retrieval (forward top-K per S1 **and** reverse top-K per S2/S3 record), a cheap
LightGBM stage-1 filter (~12 candidates per S1), and a LightGBM pair classifier with ~100 string, address, house-number,
branch-signature and **competition/context** features. Its out-of-fold probabilities feed a decision step (strict
one-to-one assignment + thresholds tuned for macro F0.5). Two data-driven post-rules remove "fake branch"
look-alikes (same name, shifted house number). The rules are switched on per country from label-free statistics,
so they also cover France, which is unseen in training. Everything runs offline on a laptop CPU: no external data,
no API calls, no pretrained models.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA)

- Train: 2.21M S1 (US 1.32M, India 0.88M), 10.3M S2+S3. Test: 1.73M S1 (India 810k, US 663k, **France 259k**).
- Singleton rate 0.0558 in both countries (a generator constant); 3.46 copies per S1 on average (0-11).
- **One-to-one is strict** (no S2/S3 record belongs to two S1) and **every true pair shares the country label**, so we
  hard-block by country (France automatically becomes its own block) and use greedy one-to-one assignment.
- 26% of the train pool matches no S1 (distractors). Test has ~2× as many distractors per S1 (5.5-5.8 pool records per
  S1 vs 4.68 in train).
- Noise on true copies: token shuffles/drops, added suffixes (Inc, Corp, Center, Services, Shri, The; French "et Fils",
  "& Cie", "Associés"), legal-form changes, OCR typos (0↔o, 1↔l), injected accents, UPPERCASE, domain names / handles,
  acronyms, made-up names at the exact address, native Indic scripts (18% of Indian partner names), empty or junk
  addresses (~4% of copies), reordered address components, St/Street/Saint, state code ↔ name, city variants.
  In US/India, house numbers on copies carry noise (digit drop 3432→432, zero padding, ranges, small shifts). French
  copies keep their house numbers.
- Hard negatives are **fake branches**: a different business with the S1's name (often plus a branch word such as
  Holdings, East, Exports; in France Groupe, Développement, Participations, SNC) and a house number shifted by 1-30.
- Empty-address copies whose name is shared by k S1 records are unresolvable ties (P(true) ≈ 0.97 / 0.47 / 0.31 /
  0.02 for k = 1 / 2 / 3 / ≥4). Our model's scores are calibrated on each of these buckets.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + optimised decision step (+ label-free
post-rules).
**Core Innovation:** (1) candidate-side competition features ("does another S1 explain this record better?");
(2) digit-level house-number relation features and branch signatures against fake branches; (3) train-time
"crowding" (synthetic fake branches) so the model and thresholds are fit at test-like distractor density;
(4) country-agnostic, label-free switches for post-rules, so an unseen country (France) is handled from its own data.

---

## 3. Candidate Generation (Blocking)

- Normalisation (`normalize.py`, hand-written dictionaries only): NFKD accent stripping; `&`→and; legal-suffix
  removal (generic, Indian and French forms: inc, llc, ltd, pvt, private, opc, sarl, sas, sasu, eurl, sci, snc, sa, …);
  abbreviation maps for names (pvt, corp, intl, mfg, bros, svc, mgmt, tech…) and addresses (EN rd/st/ave/blvd…,
  IN mg/nr/opp/sec…, FR bd/av/r/pl/imp/ch/fg/st…); rule-based transliteration of Indic scripts
  (Devanagari/Bengali/Gurmukhi/Gujarati/Odia/Tamil/Telugu/Kannada/Malayalam); a small romanisation dictionary learned
  from training pairs (e.g. "praibhet"→private); postcode / house-number parsing with regular expressions.
- **Blocking keys:** per country, sparse binary features weighted by IDF (computed on that country's records,
  unsupervised) and L2-normalised:
  (a) `combo_c` = name character 4-grams + address word uni/bigrams, top-60 per S1;
  (b) `combo` = name word uni/bigrams + address word uni/bigrams, top-30 per S1;
  (c) [TBD if kept] **reverse top-5**: for every S2/S3 record, its 5 most similar S1 in each space (recovers copies of
  S1 whose own list is flooded by look-alikes).
  Chunked sparse matrix products, processes in parallel, memory bounded.
- **Stage-1 filter:** a 250-tree LightGBM on cheap features (blocking cosines/ranks, 3 RapidFuzz scores, house-number
  agreement, per-S1 gaps), trained on held-out train S1. It keeps pairs with p ≥ 0.001 (max 20 per S1). This is the
  exact set the final model scores, i.e. `candidate_pairs.tsv`.
- **Candidate pairs generated:** test 20.6M (≈11.9 per S1) [TBD final].
- **Recall:** wide union 0.9888 at 72 cand/S1 → after stage-1 0.9842 at 10 cand/S1 (oracle F0.5 ceiling 0.9952,
  30k-S1 sample). Previous narrow blocking: 0.9814 at 30 cand/S1.

| configuration (30k train S1) | pair recall | cand/S1 | oracle F0.5 |
|---|---|---|---|
| combo@20 + name_c4@10 | 0.9735 | 26.6 | – |
| combo_c@20 + combo@10 | 0.9770 | 22.5 | – |
| combo_c@25 + combo@15 | 0.9813 | 30.0 | 0.9938 |
| union combo_c@60 + combo@30 | 0.9888 | 71.7 | – |
| + stage-1 filter (p ≥ 0.001) | 0.9842 | 10.0 | 0.9952 |

---

## 4. Matching Model

**Features used (~99):**
- Name: RapidFuzz ratio / partial / token-sort / token-set / Jaro-Winkler on normalised, core, compact and raw names;
  IDF-weighted token overlap; acronym and first-token match; DBA alternative-name max similarity; native-script and
  transliteration-aware similarity; learned branch-word ("extra token") encoding; label-free branch-word detector
  (co-occurrence of an extra token with a changed house number).
- Address: token-set / Jaccard / char similarity; postcode equal/conflict/missing; house-number overlap/conflict and
  **digit-level relations** (absolute/relative difference, digit substring, digit Levenshtein, same length, small
  offset); address-without-numbers similarity; **branch signature** (same street, number shifted by 1-30).
- Context / competition: rank and gap of this pair among the S1's candidates and **among all S1 that retrieved this
  record** (candidate side), number of S1 competing for the record, sibling support, house-number cluster size,
  name-duplicate counts (how many S1 / pool records share the name).
- Meta: source (S2/S3), empty-field flags, blocking provenance (sims and ranks per generator).
- No raw country value is ever a feature.

**Top features by gain:** candidate-side combo_c rank, learned branch-word sum, candidate-side combo_c gap, small house
offset, branch-word max, house cluster size, house overlap, address token-set, name token-sort, label-free
branch-word rate.

**Model type:** LightGBM (binary, learning rate 0.1, 127 leaves), trained on 800k train S1 (≈9.6M pairs) with the
train pool crowded by +50% [TBD: 90%] synthetic fake branches. 5-fold GroupKFold by S1 → out-of-fold
probabilities; the final model uses the mean CV iteration count. [TBD: 3-seed average]
**Threshold selection method:** greedy one-to-one assignment (each record to its highest-scoring S1), then accept the
top candidate if p ≥ t1, extra candidates if p ≥ t2 and p ≥ r·best. (t1, t2, r) are tuned by coordinate search to
maximise **macro F0.5 per S1 on OOF predictions** (exp17: 0.74 / 0.02 / 0.75). Expected-F0.5 decoding was tested and
rejected (0.9763 vs 0.9771).

**Post-rules (label-free switches, `postprocess.py`):**
- *House rule:* for every country where accepted "digit drop" pairs are rare (< 3 per 100 S1, i.e. copies keep their
  house numbers; measured on that country's own test predictions: France 1.1 vs US/India ~20), drop accepted pairs
  whose street number differs from the S1's. Public score +0.0014 on the total (France ≈ 0.925 → 0.938).
- *Shift rule (US):* drop accepted pairs whose house number is shifted by 1-30 on the same street (test has 4.1×
  the train rate of such look-alikes). Public score +0.0014 on the total (US +0.0036).

---

## 5. Results & Error Analysis

- **F0.5 (macro), honest out-of-fold on train:** 0.9815 (US 0.9832, India 0.9788) [TBD final]. Public leaderboard
  best: [TBD] (history: 0.679 rule baseline → 0.955 → 0.963 → 0.964 → 0.966 → 0.9687 → 0.9701 → [TBD]).
- **Loss decomposition (OOF):** 60% of the loss is S1 with correct but incomplete match lists (recall). Missed pairs:
  blocking 1.9% of true pairs, empty-address ties (largely unresolvable), noisy copies with changed house numbers.
- **Common false positives:** fake branches (same name, house number shifted by a few units, optional branch word),
  co-located different businesses, and empty-address records whose name is shared by several S1.
- **Common false negatives:** empty-address copies of S1 that share their name with other S1; copies with a changed
  house number (digit drop / small shift) that look like fake branches; made-up names or domain names at the exact
  address; copies of very common names that fall outside the forward top-K.
- **Unseen country (France):** multi-country dictionaries and accent stripping; no country features; the country-held-out
  simulator (train on US → test India and vice versa) measured each feature group's robustness; post-rules switched on
  from France's own label-free statistics. Probe submissions estimate France at ≈ 0.94 vs US+India ≈ 0.975.

[TBD: experiment table from experiments.csv / PROGRESS.md §7]

---

## 6. Conclusion

A carefully blocked two-stage gradient-boosting pipeline gets most of the way. The decisive gains came from the data:
the fake-branch mechanism, the country-specific house-number behaviour and the distractor density. Competition features
and label-free, country-agnostic rules turned these into score. The main lessons: validate on a split that reproduces
test conditions (density, unseen country), and use probe submissions to attribute the public score per country.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `README.md` (setup + one command), `requirements.txt` (pinned), `src/`:
`config.py` (all constants), `io_utils.py`, `normalize.py`, `blocking.py`, `stage1.py`, `features.py`, `model.py`,
`decide.py`, `postprocess.py`, `run_pipeline.py` (entry point), `sanity_check.py`.

```bash
.venv/bin/python code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train \
    --test-dir dataset/test --out-dir runs/final [TBD flags]
.venv/bin/python code/business_entity_resolution/src/postprocess.py --run runs/final --out output --house-rule --shift-rule US
```

### B. Licences and data

Python 3.12; numpy, pandas, scipy, pyarrow (BSD/Apache-2.0), scikit-learn (BSD-3), LightGBM (MIT), RapidFuzz (MIT),
joblib (BSD-3), tqdm (MIT/MPL). No pretrained models, no GPL libraries (accent stripping via `unicodedata`).
**No external data, lookups or APIs of any kind were used.** All dictionaries are hand-written from general language
knowledge or learned from the provided training labels. Unlabelled test text is used only for unsupervised statistics
(IDF weights, name-duplicate counts, label-free rule switches).
