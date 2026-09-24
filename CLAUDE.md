# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

You are the engineer on Yash's team for the Amazon ML Challenge 2026. Read this whole file before writing any code, and re-read the **Hard rules** section before every submission.

**Goal:** finish in the **Top 100** on the **private** leaderboard.
**Deadline:** challenge closes **27 Sept 2026, 11:59 PM IST**. The final zip (code + outputs + methodology) must be ready before then.
**Submissions:** max **5 per day** (15 total). Never spend one on an idea that has not already improved the local validation score.

---

## 1. The problem in one paragraph

There are three sources of business records, each with `entity_id`, `business_name`, `business_address` and `country`. Source 1 (S1) is a **deduplicated** reference list. For **every S1 record**, output every Source 2 / Source 3 record that is the same real-world business. That can be zero, one or many. The data is noisy: abbreviations (Pvt/Private, Ltd/Limited, Corp/Corporation, Rd/Road, St/Street), legal-suffix differences, DBA/trade names, & vs "and", word-order swaps, typos, transliterations, missing PIN/ZIP/state, landmark addresses ("Near SBI ATM"), municipal numbering formats and reordered address components. Sources 2 and 3 are **not** deduplicated, so one S1 record can own several S2 records.

---

## 2. HARD RULES (breaking any of these = rejection or disqualification)

1. **No external data lookup of any kind.** No geocoding APIs, no business registries, no web search, no commercial ER services, and no downloading gazetteers or address databases. The pipeline must make **zero network calls at runtime**. The only exception is downloading permitted pretrained model weights once during setup. Hand-written normalisation dictionaries built from general language knowledge are fine; document them in the methodology. Scraped lists are not.
2. **Model licence:** any model used for the final result must be **MIT or Apache 2.0** and **≤ 8B parameters**. Check the licence on the model card before using any model, because licences differ within the same family. Known OK: `intfloat/multilingual-e5-small`/`-base` (MIT), `BAAI/bge-m3` (MIT), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0), `Qwen/Qwen2.5-7B-Instruct` / `-1.5B-Instruct` / `-0.5B-Instruct` (Apache 2.0). **NOT OK:** Llama (custom licence), Gemma (custom), Qwen2.5-3B (non-Apache). LightGBM (MIT), XGBoost (Apache), scikit-learn (BSD) are fine. Prefer permissively licensed libraries in general. **Avoid `unidecode` (GPL)** and use `unicodedata` NFKD for accent stripping.
3. **Country is an open set.** Training has US and India. **Test also has France**, which never appears in training. Never hard-code, filter, one-hot or special-case `{US, India}` in a way that breaks for an unseen label. Every test S1 record, France included, must appear in the output.
4. **Output format** (both files tab-separated, UTF-8, header row, no quoting):
   - `matching_results.tsv`: columns `source1_entity_id`, `matched_entity_ids`
   - `candidate_pairs.tsv`: columns `source1_entity_id`, `candidate_entity_ids`
   - Exactly **one row per test S1 id**, with no duplicate rows.
   - ID lists are comma-separated with **no spaces**. They contain **only S2-/S3- ids that exist in the test files**, with no duplicates in a list and never an S1 id.
   - An empty list means an empty field (`S1-00003\t`).
   - `candidate_pairs.tsv` = the **exact final set the model scored** (the last filtering stage), not an earlier, bigger blocking pass.
   - **Every matched id must also be a candidate** (matches ⊆ candidates).
5. **Always read TSVs with `sep="\t"`** (pandas: also `dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE`, or empty fields become NaN and quotes break parsing).
6. **Never read the validation answers** (`splits/*/val_ground_truth.tsv`) from pipeline code. Only `evaluate.py` may read them.
7. **Never tune on the test set** and never hand-label test records. Using the unlabeled test text for unsupervised things like fitting TF-IDF vocab or IDF is allowed, but document it.
8. **Reproducible:** fixed seeds everywhere, a pinned `requirements.txt`, and one command regenerates both output files from the raw data.
9. **Before any leaderboard upload,** run the official validator and confirm PASS:
   ```
   python3 utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```

---

## 3. The metric and what it means for every decision

- **F0.5 is computed per S1 entity, then macro-averaged** over all S1 entities, including singletons.
- **Singleton** (no true matches): predicting `[]` scores **1.0**, and predicting anything scores **0.0**.
- **Non-singleton**: predicting `[]` scores **0.0**. Otherwise F0.5 = 1.25·P·R / (0.25·P + R).
- Precision counts roughly **2× recall**.

Worked intuition (truth has 3 matches):

| Prediction | P | R | F0.5 |
|---|---|---|---|
| 1 correct only | 1.00 | 0.33 | **0.71** |
| 2 correct only | 1.00 | 0.67 | 0.91 |
| 3 correct + 1 wrong | 0.75 | 1.00 | 0.79 |
| 1 correct + 1 wrong | 0.50 | 0.33 | 0.45 |

**Consequences:**
- A single confident correct match already earns ~0.7. Adding a doubtful extra is usually a bad bet.
- Getting singletons right is a large, cheap share of the score. **Predicting `[]` for everything scores exactly the singleton rate.** That is the floor every model must beat.
- Thresholds must be tuned on **macro F0.5 per entity**, never on pair-level accuracy, AUC or F1.

---

## 4. Repository layout (build exactly this)

```
student_resource/                  # the unzipped official resource (dataset/, utils/, Documentation_template.md)
├── CLAUDE.md                      # this file
├── evaluate.py                    # PROVIDED, already tested. Do not rewrite its metric.
├── experiments.csv                # auto-appended by evaluate.py
├── splits/                        # created by evaluate.py make-split
├── reports/                       # error dumps from evaluate.py
├── runs/<split>/                  # pipeline outputs on validation splits
├── output/                        # FINAL test outputs (matching_results.tsv, candidate_pairs.tsv)
└── code/business_entity_resolution/
    ├── README.md                  # exact end-to-end run instructions
    ├── requirements.txt           # pinned versions
    └── src/
        ├── config.py              # all thresholds, K values, seeds, paths, feature flags in ONE place
        ├── io_utils.py            # safe TSV read/write, output writers
        ├── normalize.py           # text cleaning + field parsing
        ├── blocking.py            # candidate generation
        ├── features.py            # pair features
        ├── model.py               # training (grouped CV, OOF preds) + inference
        ├── decide.py              # post-processing: one-to-one, thresholds, singleton gate
        ├── llm_rerank.py          # OPTIONAL borderline reranker (only if it helps on validation)
        └── run_pipeline.py        # CLI: --train-dir --test-dir --out-dir [--split-name]
```

`run_pipeline.py` must take `--train-dir` and `--test-dir`, reading `train_*.tsv` from the first and `test_*.tsv` from the second. Because `evaluate.py make-split` exports validation data in exactly this layout, **the same code runs on validation and on the real test set**. There is no separate "validation mode" to go out of sync.

Every stage must accept any list of countries seen in the data, and must also handle a country **not seen in training** without crashing or silently dropping rows.

---

## 5. Evaluation harness: `evaluate.py` (provided, tested)

```
python3 evaluate.py selftest
python3 evaluate.py check-data --train-dir dataset/train
python3 evaluate.py make-split --train-dir dataset/train --name val --mode random --val-frac 0.2 --seed 42
python3 evaluate.py make-split --train-dir dataset/train --name loco_india --mode loco:India
python3 evaluate.py make-split --train-dir dataset/train --name loco_us --mode loco:US
python3 evaluate.py score --split val --pred runs/val/matching_results.tsv \
      --cand runs/val/candidate_pairs.tsv --tag expNN --note "what changed"
```

- Use the **exact** country strings printed by `check-data` for the `loco:` splits.
- `score` prints: macro F0.5, singleton accuracy, non-singleton F0.5, micro P/R, P/R for S2 and S3 separately, per-country scores, **blocking pair recall**, avg candidates per S1, reduction ratio, and the **oracle F0.5 ceiling** (the score a perfect model would get on your candidates).
- It writes `reports/<tag>_false_pos.tsv`, `_false_neg.tsv` and `_missed_by_blocking.tsv`, and appends a row to `experiments.csv`.
- The split is by **S1 entity**. Every S2/S3 record goes with its owner, and unmatched distractor records are split randomly (or by country in `loco` mode). No leakage.
- The pipeline may `from evaluate import f05, macro_f05` for threshold tuning on **OOF predictions of the training part only**.

**The `loco_*` splits are the France simulator.** A model trained on US and tested on India (and vice versa) shows how the pipeline behaves on an unseen country. Treat `loco` scores as a first-class metric. A change that improves `val` but hurts `loco` is suspect.

---

## 6. Pipeline design

### Stage 0: Data profiling (do this FIRST, before any modelling)
Run `check-data` and report to Yash:
- Singleton rate (the "predict all empty" floor).
- Distribution of matches per S1.
- **Is matching one-to-one?** Does every S2/S3 id belong to at most one S1? This decides whether the one-to-one assignment in Stage 5 is safe.
- **Do true pairs always share a country label?** This decides whether hard-blocking by country is safe.
- Share of S2/S3 records that match nothing (distractors).
- The empty-field rate.
- 20 random true pairs and 10 singletons, read by eye. Summarise the noise patterns actually present.

### Stage 1: Normalisation (`normalize.py`)
Build these for every record, and keep the raw text too:
- `name_norm`: lowercase, Unicode NFKD + strip accents, `&`→`and`, punctuation→space, collapse spaces.
- `name_core`: `name_norm` with **legal suffixes removed**. This must be a multi-country list, for example:
  - generic: `inc, incorporated, llc, llp, ltd, limited, corp, corporation, co, company, plc, lp`
  - India: `pvt, private, pvt ltd, opc, huf`
  - France: `sarl, sas, sasu, sa, eurl, sci, snc, sca, et cie, cie, ste, societe`
  - common descriptors to flag rather than delete: `the, and, group, holdings, enterprises, services, solutions, traders, industries, international`
- `name_tokens` with **abbreviation expansion** (pvt→private, ltd→limited, corp→corporation, co→company, intl→international, mfg→manufacturing, bros→brothers, assoc→associates, svc/svcs→services, mgmt→management, tech→technology, dept→department).
- `dba_parts`: split on `dba`, `d/b/a`, `t/a`, `trading as`, `aka`, `formerly`, `(`…`)`, and produce each part as an alternative name.
- `acronym`: first letters of `name_core` tokens (catches "IBM" vs "International Business Machines").
- `addr_norm` with expansions:
  - EN: `rd→road, st→street, ave→avenue, blvd→boulevard, dr→drive, ln→lane, ct→court, hwy→highway, ste→suite, apt→apartment, fl→floor, bldg→building, n/s/e/w→north/south/east/west`
  - IN: `mg→mahatma gandhi, nr/near→near, opp→opposite, bldg, flr, sec/sector, ph/phase, colony, nagar, marg, chowk, gali, mohalla, vihar, enclave` (keep them as tokens; do not delete)
  - FR: `bd→boulevard, av→avenue, r→rue, pl→place, imp→impasse, chem→chemin, fg→faubourg, st/ste→saint/sainte` (context: before a name), `cedex`
- Parsed address pieces (heuristic regex, never an external parser):
  - `postcode`: IN 6-digit PIN; US 5-digit ZIP (+4 optional); FR 5-digit code. The generic rule is "5–6 digit standalone number". Store the raw digits; don't classify by country.
  - `house_numbers`: every number token that isn't the postcode (unit numbers, plot numbers, `12/3`, `12-A`, `bis`/`ter`).
  - `landmark_flag`: address contains `near/opp/opposite/behind/beside/next to/pres de/en face`.
  - `city_tokens`: the last 1–3 non-numeric tokens (heuristic).
- **Number handling matters.** Mismatched house numbers or postcodes are among the strongest "not a match" signals. Mismatched chain branches (same name, different address) are the #1 expected false-positive source.

### Stage 2: Blocking / candidate generation (`blocking.py`)
Goal: **pair recall ≥ 0.97** with a manageable candidate count. Union these generators, each producing top-K neighbours per S1 from the S2+S3 pool:
1. **Char n-gram TF-IDF (3–5 grams, `char_wb`) on `name_norm`**, cosine top-K. This is the workhorse and is typo- and language-robust.
2. **Word TF-IDF on `name_core`**, top-K (catches reordered words).
3. **Char TF-IDF on `addr_norm`**, top-K (catches DBA/trade-name cases where the names differ entirely).
4. **Exact keys:** same postcode + shared rare name token; same `acronym`; same `name_core`.
5. *(Optional, if it adds recall)* multilingual embedding ANN (e5 / bge-m3) on `name + " | " + address`.

Rules:
- If Stage 0 shows true pairs **always** share a country label, restrict neighbours to the same country. If not, don't hard-block. Use country equality as a feature instead.
- Fit vectorisers on **all source text (train + test, S1+S2+S3)**. This is unsupervised and keeps the vocab consistent for France.
- Use sparse matrix top-K via `sparse_dot_topn`-style chunked multiplication, or sklearn `NearestNeighbors(metric="cosine")`. Chunk it so memory stays bounded.
- Tune K per generator on `val` using **pair recall** and **oracle ceiling** from `evaluate.py`. Report the recall/size trade-off table to Yash.
- Keep, per candidate, **which generators found it and its rank in each**. These become features.

### Stage 3: Pair features (`features.py`)
For each (S1, candidate) pair:
- **Name:** rapidfuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, Jaro-Winkler, and normalised Levenshtein on `name_norm` and on `name_core`; char-TF-IDF cosine; word Jaccard; **IDF-weighted token overlap** (rare shared tokens count, generic words like "traders" don't); the **max similarity over DBA alternative names**; acronym match; first-token match; length ratio; whether a legal suffix is present on either side.
- **Address:** char-TF-IDF cosine; token-set ratio; Jaccard; `postcode_equal` / `postcode_conflict` / `postcode_missing` (three-state, not one boolean); house-number overlap / conflict / missing; city-token overlap; landmark flags on each side; address length on each side.
- **Context (the strong ones):**
  - `rank_in_s1`: this candidate's rank among the S1 record's candidates by name sim, and by combined sim.
  - `gap_to_best`: best score for this S1 minus this pair's score.
  - `n_close_candidates`: how many candidates for this S1 score within δ of the best (ambiguity).
  - `rank_in_candidate`: from the candidate's side, how this S1 ranks among **all S1 records that retrieved this candidate**, plus the margin to the second-best S1. This detects "this S2 record fits a different S1 better".
  - `s1_neighbour_density`: how many *other S1 records* are very similar to this S1 by name (chains/franchises). This is high-risk for false merges.
  - `sibling_support`: max similarity between this candidate and the other high-scoring candidates of the same S1. S2/S3 duplicates of one business look alike.
  - Blocking provenance: which generators found it, and its rank in each.
- **Meta:** candidate source (S2 vs S3), `country_equal`, missing-field flags.
- **Never** use the raw country string as a categorical feature. A model trained on {US, India} would have no value for "France".

### Stage 4: Model (`model.py`)
- **LightGBM binary classifier** on pairs, label = candidate is in the S1's ground-truth list.
- **5-fold GroupKFold grouped by S1 id** on the training part → **out-of-fold (OOF) probabilities** for every training pair. All threshold and decision tuning uses OOF predictions, never in-sample ones.
- Final model = trained on all training pairs with the chosen hyper-parameters. Optionally average the 5 fold models for test inference.
- Start with sensible defaults (`num_leaves` 31–63, `learning_rate` 0.05, early stopping on the fold's validation set, `min_data_in_leaf` ≥ 20). Only small tuning is needed; features matter far more.
- Log feature importance (gain) each run. Drop features that are useless, and investigate any single feature that dominates suspiciously (possible leakage).
- **Unseen-country robustness:** check the `loco_*` splits. If the scores collapse, the model is relying on country-specific quirks. Remove or neutralise those features.

### Stage 5: Decision step (`decide.py`) — where most teams lose points
Apply in this order and measure each step separately on `val`:
1. **One-to-one assignment** (only if Stage 0 confirmed it): each S2/S3 record may be assigned to **at most one S1**, whichever gives it the highest probability. Remove it from all other S1 lists. Optionally require a margin over the runner-up.
2. **Threshold:** keep candidates with `p ≥ t`. Tune `t` on OOF predictions to maximise **macro F0.5** with `evaluate.macro_f05`. Grid 0.05–0.95, step 0.01.
3. **Two-threshold rule:** accept the top-1 candidate if `p ≥ t1`, and additional candidates only if `p ≥ t2` (t2 may differ from t1). Tune jointly.
4. **Singleton gate:** if the best candidate is below `t1`, output `[]`. Optionally train a small "does this S1 have any match?" classifier on entity-level features (max p, gap between top-1 and top-2, number of candidates above 0.3, neighbour density), and gate on it.
5. **Expected-F0.5 optimisation (advanced):** per S1, sort candidates by p and pick the prefix size k (k = 0 means empty) that maximises the **expected** F0.5 under the model's probabilities, estimated with ~200 Monte-Carlo samples of which candidates are true. This requires calibrated probabilities, so check calibration on OOF first and apply isotonic calibration if needed. Keep it only if it beats the tuned thresholds on `val` **and** `loco`.
6. **Per-country thresholds:** usually overfit. Don't use them unless the gain is large and stable across seeds. **Unseen countries always use the global thresholds.**

### Stage 6 (optional, Day 2–3): LLM reranker for borderline pairs (`llm_rerank.py`)
- Only for pairs with p in an uncertainty band (e.g. 0.25–0.75), which is a small slice.
- Model: `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0) on GPU, or `Qwen2.5-1.5B-Instruct` if resources are tight. Run locally. **No API calls.**
- Prompt: both records' names and addresses, then ask "Same real-world business? Answer YES or NO". Take the logit of YES vs NO as a score and **feed it as an extra feature** into a second-stage model or a blend. Don't use it as a hard override.
- Keep it only if `val` **and** `loco` improve. Cache all LLM outputs to disk so reruns are free.
- If there's no GPU, skip this stage rather than burn hours.

---

## 7. The edge playbook (why we beat teams using the same tools)

1. **Trust local validation, not the leaderboard.** Every idea gets a `val` + `loco` score before it can be submitted. Aim for 30+ logged experiments and ≤ 15 submissions.
2. **Error analysis loop after every model change.** Read `reports/<tag>_false_pos.tsv` (prioritise rows marked `SINGLETON`) and `_false_neg.tsv`. Categorise the top 30 errors into patterns, then propose **one** targeted fix per pattern. Show Yash the categories with counts before implementing.
3. **Blocking-vs-model diagnosis:** if `oracle ceiling − macro F0.5` is large, the model or decision step is the problem. If the oracle ceiling itself is low, blocking is the problem.
4. **Decision step as a first-class model:** one-to-one assignment, tuned thresholds, singleton gating and expected-F optimisation. Each step gets its own measured gain.
5. **France readiness:** multi-country normalisation lists, accent stripping, no country categoricals, `loco` validation, and vectorisers fit on all text including test.
6. **Context features** (rank, gap, candidate-side competition, neighbour density, sibling support) carry most of the lift over plain string similarity.
7. **Stability:** for the final submission, prefer the configuration that is good on `val`, `loco_india` and `loco_us`, and across 2–3 split seeds. **Do not** chase a public-leaderboard bump that local validation doesn't support. The private leaderboard decides.

---

## 8. How to work with Yash (workflow rules)

- **Build in this order, one stage per step:** Stage 0 report → io + normalisation → blocking (report the recall table) → baseline decision (name similarity + threshold, no ML) as a **Day-1 submission** → features + LightGBM → decision step → extras. Don't write the whole pipeline in one go.
- **After every experiment**, reply with this block:
  ```
  EXP <tag>: <one-line change>
  val   macroF0.5 = x.xxxx (Δ vs best = +/-x.xxxx) | singleton acc x.xx | oracle ceiling x.xx | pair recall x.xx
  loco  india x.xxxx | us x.xxxx
  verdict: KEEP / REVERT, and why
  next: <the single next thing to try>
  ```
- **One change per experiment,** so gains can be attributed.
- **git commit** after every KEEP, with the tag and score in the message. Tag submitted versions `sub-dayN-M`.
- Put every magic number in `config.py`, never inline.
- If a run will take > 10 minutes, say so first and propose a faster approximation (subsample, smaller K).
- **Ask Yash before:** submitting to the leaderboard (he uploads it himself), adding any new pretrained model or dependency, or changing the split protocol.
- If anything seems to contradict the Hard rules, stop and ask. When unsure whether something counts as "external data", the answer is **don't**, and Yash can ask the organisers through their query form.
- Never claim a score you didn't compute with `evaluate.py`.

---

## 9. Submission checklist (run every time)

1. Regenerate from scratch: `python3 code/business_entity_resolution/src/run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir output`
2. Official validator → **PASS**.
3. Sanity checks, printed:
   - row count == number of test S1 ids
   - share of empty predictions (compare with the training singleton rate)
   - avg matches per non-empty row
   - rows per country, with **France present**
   - France's empty-prediction rate is not wildly different from US/India. If France is ~100% empty or ~0% empty, something is broken.
4. The matched ⊆ candidates check passes.
5. Copy the outputs to `submissions/<tag>/` with a note of its val/loco scores and the git commit hash.
6. Yash uploads `output/matching_results.tsv`. Record the public score next to the local score in `submissions/log.md`.

---

## 10. Final package (must exist before the deadline; build it by Day 3 afternoon)

```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md     # filled in
```
- README: environment setup, model-weight download step (if any), one command end-to-end, expected runtime and hardware.
- `requirements.txt`: pinned versions (`pip freeze` of the actual env, trimmed).
- Code: docstrings on every function. The guidelines explicitly ask for commented source.
- **Methodology document** (fill the official template): methodology overview; normalisation (list the dictionaries and state they are hand-written, not looked up); blocking strategy with the recall / reduction-ratio / oracle-ceiling table; features grouped by type with top importances; model and CV scheme; decision step (one-to-one, thresholds, singleton gate) with the measured gain of each; France/unseen-country handling and `loco` results; the experiment history table from `experiments.csv`; licences of every model and library used; a statement that no external data or APIs were used.
- Do a clean-room test: a fresh virtualenv, install `requirements.txt`, run the README command, and diff the outputs against `output/`.

---

## 11. Timeline

| When | Target |
|---|---|
| Day 1 (25 Sep) | Stage 0 report; splits; normalisation; blocking with recall ≥ 0.95; rule-based baseline → **first submission**; start features. |
| Day 2 (26 Sep) | Full features + LightGBM + OOF; decision step (one-to-one, thresholds, singleton gate); 2–3 error-analysis loops; France hardening via `loco`; 2–4 submissions. |
| Day 3 (27 Sep) | Optional LLM reranker / expected-F; stability check across seeds; pick the final config by local validation; **package + methodology by 6 PM IST**; the final submissions after that. |

---

## 12. Common bugs to guard against

- Reading TSVs without `sep="\t"`, or letting pandas turn empty strings into NaN / "nan".
- IDs like `S2-00047` getting parsed as something other than strings. Always use `dtype=str`.
- Writing `nan` or spaces into ID lists.
- Train/validation leakage: fitting anything that uses **labels** on data that includes the validation S1 records, or tuning thresholds on in-sample predictions.
- Blocking that silently drops S1 rows with empty names or addresses. Every S1 needs a row.
- France rows dropped by a country filter or mapping dict.
- Candidate file written from an earlier stage than the one the model actually scored.
- One-to-one assignment applied when Stage 0 showed it is not strictly true.
