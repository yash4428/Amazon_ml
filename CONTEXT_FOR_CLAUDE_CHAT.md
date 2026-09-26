# FULL CONTEXT BRIEF — Amazon ML Challenge 2026: Business Entity Resolution
*(paste this whole document into a chat; it is written so a model with no file access has the complete picture)*

---

## 0. What I want from you (the chat assistant)

I am competing in the **Amazon ML Challenge 2026 — Business Entity Resolution**. I have built a full pipeline
and done two days of analysis with a coding agent. We are at **public F0.5 = 0.970125** (~600th). The public
leaderboard top score is **0.990748**; the top-100 cut-off is between **0.986 and 0.990**. I need ideas that can
realistically move us **+0.01 to +0.02** within the remaining time. Please:

1. Read everything below carefully — the data, the reverse-engineered data generator, what we built, every
   experiment and every submission score, and all the EDA (including negative results).
2. Tell me what you think the top teams are doing that we are not, and **why** (reason from the numbers).
3. Propose concrete, testable ideas ranked by expected gain × feasibility on a CPU-only laptop with ~1 day left,
   including how to validate each locally and/or with an exact-attribution submission (§8.4).
4. Challenge our assumptions (§12 lists the ones we are least sure about).

---

## 1. The competition

### 1.1 Task
Three sources of business records, each record = `entity_id`, `business_name`, `business_address`, `country`.
- **Source 1 (S1)** is a deduplicated reference list.
- **Sources 2 and 3 (S2, S3)** are noisy, NOT deduplicated.
- For **every S1 record**, output all S2/S3 records that are the same real-world business (0, 1 or many).

### 1.2 Files
- Train: `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
  (`source1_entity_id`, `matched_entity_ids` = comma list, empty for singletons).
- Test: `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (no labels).
- All TSV, UTF-8. IDs look like `S1-925783039`, `S2-166376419`, `S3-202863386` (random 9-digit numbers).

### 1.3 Output (two files, tab-separated, header row)
- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids` — **exactly one row per test S1**, ids
  comma-separated with no spaces, only S2/S3 ids that exist in test, no duplicates, empty field for no match.
  **This is the only file scored.**
- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids` — the exact candidate set the model scored
  (the last filtering stage). Matches must be a subset of candidates.
- **Organiser email (after day 1):** candidate_pairs.tsv is part of the final submission; *"the approach that
  generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the
  public/private leaderboard"*. Blocking must scale ("Amazon resolves entities across billions of records").

### 1.4 Metric
**F0.5 per S1 entity, macro-averaged over ALL S1 (singletons included):**
- Singleton (no true matches): predict empty → 1.0; predict anything → 0.0.
- Non-singleton: predict empty → 0.0; else F0.5 = 1.25·P·R / (0.25·P + R).
- Precision weighs ~2× recall. Examples with 3 true: 1 correct only → 0.71; 2 correct only → 0.91;
  3 correct + 1 wrong → 0.79; 1 correct + 1 wrong → 0.45. With 4 true: finding 3 of 4 (P=1, R=0.75) → 0.9375.
- Public LB = subset of test; private LB (remaining test) decides the final ranking. Scores shown with 5 decimals.

### 1.5 Rules / constraints
- **No external data** of any kind (no geocoding, registries, web, gazetteers). No network at runtime.
  Hand-written dictionaries from general knowledge are OK (must be documented).
- Any model used must be **MIT or Apache-2.0 licensed and ≤ 8B parameters** (e.g. e5-small/base MIT, bge-m3 MIT,
  multilingual MiniLM Apache, Qwen2.5-7B/1.5B/0.5B-Instruct Apache). Libraries: LightGBM MIT, sklearn BSD, etc.
- **Country is an open set**: train has US + India; **test also has France**, which never appears in train.
- **5 submissions per day.** Final package (code + outputs + methodology doc) must be reproducible.
- Deadline: leaderboard closes **27 Sep 2026, 23:59 IST**. Package ready by ~18:00 IST that day.
- We cannot hand-label test or tune on test labels; we use the public LB only through a few careful,
  hypothesis-driven submissions (§8.4).

### 1.6 Hardware
Apple M4 Pro laptop, 12 CPU cores, 24 GB RAM, **no NVIDIA GPU** (Apple MPS only). Python 3.12.
A full pipeline run takes ~100 min cold (blocking train ~40 min, test ~25 min, model ~20 min, scoring ~15 min),
~60 min when blocking caches exist. Only one heavy job fits in memory at a time (12-15 GB peak).

---

## 2. The dataset (facts we measured)

### 2.1 Sizes
| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,820 (US 1,323,633 · India 883,188) | 5,034,616 | 5,285,603 |
| Test | 1,732,543 (India 809,986 · US 663,106 · **France 259,452 = 14.98%**) | 4,887,273 | 5,082,316 |

Pool (S2+S3) per S1: train 4.68 (India) / 4.67 (US); test France 5.53, India 5.82, US 5.76 (**~24% more in test**).
Test pool by country: S2 France 703k, India 2.31M, US 1.87M; S3 France 732k, India 2.41M, US 1.95M.

### 2.2 Ground truth structure (train)
- **Singleton rate 0.056** (same for US and India) → predicting all-empty scores 0.056.
- Matches per S1: 0:123k · 1:119k · 2:375k · 3:531k · 4:484k · 5:322k · 6:165k · 7:64k · 8:19k · 9:4k · 10+:0.6k
  (mean ≈ 3.46; ≈3.66 per non-singleton). 1.69M S1 have >1 match from the same source.
- Matches come from: both S2+S3 for 1.78M S1, only S3 for 164k, only S2 for 143k.
- Copies per S1 by source: S2 {0:8%, 1:38%, 2:31%, 3:16%, 4+:7%}; S3 similar.
- **One-to-one is strict**: no S2/S3 id appears in two S1 lists.
- **All 7.64M true pairs share the country label** → blocking within country is safe.
- 26% of the train pool (2.68M of 10.32M) belongs to NO S1 (distractors).
- Exact normalised-name equality in only 22% of true pairs.
- Postcodes rare (~7% of addresses; equal in 90% of true pairs when both present). **US addresses contain
  essentially no ZIP codes**; 5-digit numbers in US addresses are house numbers (~10% of US addresses).
- S1 never has an empty address; ~3% of S2/S3 records do (14-17 empty-address pool records per 100 S1 in every
  country). Empty-address records belong to SOME S1 ~97.7% of the time (4.5% of copies vs 0.3% of distractors).
- India: 18% of partner names are native script (Devanagari, Tamil, Telugu, Bengali, Gujarati, Odia, Kannada…);
  23% of Indian partner addresses contain native-script parts (often just the state).
- 24% (India) / 8% (US) of true pairs share NO Latin name token (native script, domains, random names) and ~4-5%
  share no address token, but **name OR address always shares a token** (0.0000 have neither).
- 23% of S1 normalised names are duplicated among S1 (chains / generic names); ~40-53% of S1 share their core name
  with at least one other S1 (FR 52%, IN 53-54%, US 40-48%).
- **No leakage**: IDs and file row order are uncorrelated with matches; copies of the same S1 are NOT near each
  other in row order or ID number (medians equal to random pairs); S2-vs-S3 positions uncorrelated (corr −0.0015).
- **No overlap between train and test**: 0 identical S1 or pool records across splits (34% of test S1 names occur
  in train S1 names — generic names, different businesses).
- Exact-duplicate pool records (identical name+address) are 0.9% of the pool; 99.8% of duplicate groups share one owner.
- **Whether an S1 is a singleton is unpredictable from the S1 record** (classifier AUC 0.500).

### 2.3 Real examples — true matches (train)
```
S1-13345903 [India] OTE Food Pvt Ltd | 7 Manjusar Gidc Savli, Vadodara, Gujarat
  S2 OTE Ltd Pvt  Food | NO 5 MANJUSAR GIDC SAVLI, VADODARA, Gujarat          (house number 7→5 on a TRUE copy)
  S2 OTE Food Pvt | 5 MANJUSAR GIDC SAVLI, VADODARA, Gujarat
  S3 Pvt OTE Fdr Ltd | Door No 7 Manjusar Gidc Savli, Vadodara, ગુજરાત
S1 [US] Ruperta Templin, L.C.S.W., PC | 6605 Jockey Club Drive, Whitsett, NC
  S2 Ruperta Templin, L.C.S.W., | 6605 Jockey Club Drive, WHITSETT, NC
  S2 Ruperta Templin, | WHITSETT, 6605 JOCKEY CLUB DRIVE, NC
  S2 Ruperta Templiisgt, L.C.S.W., PC | 6605 JOCKEY CLUB DRIVE, WHITSETT, NC
  S3 Ruperta Ruperta Témplin, L.C.S.W., PC | 6605 Jockey Club Dr, Whitsett, North Carolina
S1 [US] Renae Foster Pegasus | NY, Brookhaven, 95 Church Street
  S2 Renae Foster  Pegasus Ltd | 95 CHURCH ST, BROOKHAVEN, NY
  S2 renaefosterpegasus.com | 95 Church St, BROOKHAVEN, NY
  S2 pegasusfoster.com | 95 CHURCH SAINT, LAKE RONKONKOMA, NY
  S2 Renae Foster Pegasus Inc |                                                 (empty address)
  S3 Renae Foster Pegasus-LP | ##95 Church St, NULL, Ronkonnkoma, New York
S1 [India] Shiva Energy | No 130 Ground Floor Seniamman Koil Street Tondiarpet, Chennai, Tamil Nadu
  S2 சிவா எனர்ஜி | MADRAS, CHENNAI, தமிழ்நாடு, H.NO 130 GROUSD FLOOR SENIAMMAN KOIL STREET TONDIARPET
  S3 SHIVA ENERGY | No 130 Ground Floor Seniamman Koil Street Tondiarpet, Chennai, TN
S1 [India] Pioneer Products Private Limited | 51/10G/1 West Arjun Nagar, Agra, Uttar Pradesh
  S2 पायोनियर प्रोडक्ट्स प्राइवेट लिमिटेड | DOOR NO 51/10G/1 WEST ARJUN NAGAR, AGRA, उत्तर प्रदेश
  S2 Pioneer-Products  Private Limited (ID: 63945) | NO 51/10G/1 WEST ARJUN NAGAR, AGRA, Uttar Pradesh
  S3 pioneerproducts.com | 51/10G/1 West Arjun Nagar, Agra, UP
S1 [US] Rev It Up Seafood | 1701 Hairston Avenue, Unit Apt 114, Conway, AR
  S2 revitup.com | 1701 HAIRSTON AVE, CONWAY, AR
  S2 Violyra | 1701 HAIRSTON AVENUE, CONWAY, AR                                   (fully random name, exact address)
  S3 Rev Up It Seafood | 2060-C Hairston Ave, Unit Apt 114, Conway, Arkansas
  S3 Rev It Up Seafood Ltd |
Random-name true copies: "Rizaumbradova", "Wexavi", "Zetadrexvera One", "Korzeta", "Jaxfaye Sys", "Brixbrix".
```

### 2.4 Real examples — hard negatives ("fake branches"), train
```
SINGLETON S1 Clear United Platinum LLC | 30 Franklin Street, Buckland, MA
   0.57 nobody  Clear United Platinum Riverside Llc | 34 Franklin Street, Buckland, Massachusetts
SINGLETON S1 Kessler Pegasus LLC | 8520 Reitz Lake Road, Waconia, MN
   0.80 nobody  Kessler Specialists | 8527 REITZ LAKE RD, WACONIA, MN
SINGLETON S1 Noify Roman, Inc | 306 Wilmington Court, Bloomington, IN
   0.83 nobody  Noify Róman, LLC | 309 WILMINGTON CT, BLOOMINGTON, IN
   0.71 nobody  Noify Roman, Partners | 309 Wilmington Ct, Bloomington, Indiana
SINGLETON S1 Lopes's Law Group | 2933 Willow Creek Drive, Sandy City, UT
   0.72 nobody  Lopes'S Law Group Llc | 2934 Willow Creek Dr, Sandy, Utah
   0.69 nobody  Lopes's Law Group East | 2934 Willow Creek Drive, Sandy, Utah
SINGLETON S1 Express Genomic Corporation | 4238 Bluebird Road, Northampton County, VA
   0.76 nobody  EXPRESS GENOMIC CORPORATION INC | 004249 BLUEBIRD ROAD, NORTHAMPTON COUNTY, VA
```
Branch words seen ONLY on negatives in train: group, holdings, ventures, public, exports, overseas, infratech,
industries, care, india, south, harbor, east, metro, highland, valley, summit, coastal, north, west, uptown,
downtown, southside, westgate, eastgate, lakeside, riverside, midtown, associates, solutions, greater, central,
northside, health, global, engineering, clinic, medicine, foods, traders. Words seen on POSITIVES (copy noise):
center, service(s), labs, inc, doing business as, etc.

### 2.5 France (test only — never in train)
```
Fédération de Vos | Hauts-de-France, 46 Rue Maximilien de Robespierre, Lille
   0.94 S2 fédération de vos | (46) RUE MAXIMILIEN DE ROBESPIERRE, LILLE, Hauts-de-France          (copy)
   0.84 S3 Fédération De Vos SNC Services | 53 R. Maximilien De Robespierre, Lille, Nord          (fake branch 46→53)
   0.77 S3 Fédération De Vos Développement | 53 Rue Maximilien De Robespierre, Lille, Nord
   0.64 S2 etsloisirsascom | 53 RUE MAXIMILIEN DE ROBESPIERRE, LILLE, Nord
   0.64 S3 Vantagehalopyra | N° 65 Rue Maximilien De Robespierre, Lille, Nord
Boxing Theatre SARL | 2 Stade du Souvenir, Calais, Hauts-de-France
   1.00 S2 Boxing Theatre | 2 STADE DU SOUVENIR, CALAIS, Pas-de-Calais
   0.74 S3 Boxing-Theatre Distribution SARL | 6 Stade Du Souvenir, Calais                           (fake branch 2→6)
Autonomie (France) Amicale SA | 41 Rue de Neuville, Tourcoing, Hauts-de-France
   0.87 S3 Korriza Co formerly Autonomie (France) Amicale SA | 41 Rue de Neuville, Tourcoing
   0.83 S3 Autonomie (France) Amicale Distribution Sa | 0054 Rue De Neuville, Tourcoing, Nord        (fake 41→54)
```
France properties: names = generic French words + legal form (SARL, SAS, SASU, EURL, SCI, SA, SNC, EI, Cie,
Et Fils, & Frères); few cities (Lille, Nantes, Bordeaux, Pessac, Mérignac, Dunkerque, Roubaix, Tourcoing, Calais,
Saint-Nazaire, Saint-Herblain, La Teste-de-Buch, Lège-Cap-Ferret, Pornic, La Baule-Escoublac); street words
Rue/R./BD/Av./Cours/Chemin/Quai/Impasse/Allée, "Bis", "N°", "(46)", zero-padding "0054"; region (Hauts-de-France,
Pays de la Loire, Nouvelle-Aquitaine) vs département (Nord, Gironde, Loire-Atlantique, Pas-de-Calais).
Density: 16 candidates with blocking sim ≥0.5 per S1 (US/IN ~5); **~9 branch-like candidates per S1**; 12% of
French S1 share an exact address with another S1 (US/IN 4-5.6%); acronym records 9.2 per 100 S1 (US/IN 0.5-1.1).
French branch words (from label-free statistics on test): Holding, International, Distribution, Participations,
Développement, Groupe. French copy-noise words: Associés, Services, Fils, Cie, SARL/SAS/EURL variants.

---

## 3. The data generator — what we reverse-engineered

**True copies** are generated INDEPENDENTLY from the S1 (siblings are no more similar to each other than to S1)
with noise operations: token shuffle / drop; added suffix words ([Inc], (Corp), Center, Services, Dr, Shri, The,
Group…); legal-form changes; OCR/typos (Muri1lo, Humme1, 0↔o, char swaps); injected accents; UPPERCASE (S2 style;
S2 addresses uppercase ~63%); domain/handle forms (`revitup.com`, `@handle`, `#name`, `vestsafeintelligencecom`);
acronyms (RC, PM); **fully random made-up names at the exact address**; Indian native-script transliteration; empty
address (~4% of true pairs); "NULL", "##", "(ID: 63945)" junk; address component reordering; St/Street/**Saint**;
state code ↔ full name ↔ native script; city variants (Brookhaven↔Ronkonkoma, Trappe Borough↔Collegeville);
unit/PMB additions. **House-number noise on US/Indian copies**: equal 70% of true pairs; digit drop/add
(3432→432, 302→30; P(true|this relation)=0.46); zero padding (0700); ranges (1976-1978); one digit changed
(P 0.40); small shifts (e.g. 537→536; P(true) of "same street, ±1..30" among strong candidates ≈0.70 US / 0.91 IN).
**French copies carry almost no house-number noise** (accepted digit-drop pairs 1.05 per 100 S1 in France vs ~20 in
US/India; house-diff share of accepted pairs 4.4% FR vs 14.4% US vs 19.7% IN).

**Fake branches (hard negatives)** = a different business: the S1 name (often + a branch word, sometimes a legal
form change) with the **house number shifted by ~1-30** on the same street; they come with their own noisy copies.
80% of singletons' top candidate is one. **Test has more of them per S1 than train** (branch-like candidates per S1:
US 1.02→1.58, IN 1.37→1.89, France ≈9.2). **"Shifted look-alikes"** (same street, number ±1..30, name token-set ≥90)
per 100 S1: train US 8.72 (13% carry an extra word; P(true)≈0.42) vs **test US 35.76 (4.1×; 22% extra word)**;
train India 5.13 (P(true)≈0.81) vs test India 10.25 (2×); test France 43.88 (54% extra word).

**Unresolvable ties**: an empty-address copy whose core name is shared by k S1 belongs to one of them with P≈1/k
(exact name shared by 1 S1 → P(true)=0.970; by 2 → 0.465; by 3 → 0.31; by 4+ → 0.022). Raw-name similarity
(legal form + punctuation kept) picks a unique winner in 58% of answerable ties and is right 78% of the time; the S1's
OTHER confidently matched copies pick the right owner only 31-38% (no help). 85% of S1 have no empty-address copy,
14% one, 1% two.

---

## 4. Our pipeline (Python; LightGBM; ~2,500 lines)

### 4.1 Normalisation (`normalize.py`, hand-written rules only)
- NFKD accent strip, lowercase, `&`→and; **Indic→Latin transliteration with ONE offset table** (all Indic Unicode
  blocks share the same 128-code-point layout; consonant + inherent 'a', matras, virama, schwa deletion at word end).
- **Learned transliteration dictionary** from TRAIN true pairs (Latin S1 vs native-script copy, aligned token by
  token when token counts match; majority vote): 567 mappings (pharst→first, payoniyar→pioneer, enarji→energy,
  mainejament→management, bildars→builders). Native-pair token overlap 0.21→0.94. India OOF +0.012.
- OCR digit fix inside words (humme1→hummel), merge single-letter runs (L.L.C.→llc), name abbreviation expansion
  (pvt→private, ltd→limited, corp→corporation…), legal-form removal lists (US/IN/FR: inc, llc, ltd, pvt, private,
  sarl, sas, sasu, eurl, sci, snc, ei, cie…), filler removal (the, and, of, shri, dr, de, du, la…), DBA/aka/fka split,
  domain/@handle → compact name, acronym. Fields: name_norm, name_core, name_compact, acronym, dba.
- Address: contraction (street/st/saint→st, road/rd, avenue/ave/av, boulevard/blvd/bd, rue/r, nagar/ngr…), noise
  removal (null, no, door, h.no…), postcode, house_numbers, landmark flag.
  **Bug found on day 2**: any standalone 5-6 digit number was treated as a postcode, so 5-digit US house numbers
  were dropped (8.9% of US S1 / 6.3% of US pool had no house number). Fixed (postcode only when a comma component on
  its own, a trailing "STATE 12345", or a 6-digit PIN that is not the first number). The fix changed test behaviour
  little (US shifted acceptances 9.79 → 9.89 per 100 S1).

### 4.2 Blocking (`blocking.py`)
Per country (country-hard-blocking), sparse IDF-weighted cosine top-K via chunked sparse matmul in processes +
per-row argpartition; features with df > max_df (20000) or df < 2 are dropped. Spaces:
- `combo`: name_core + address word unigrams/bigrams (prefixed n:/a:).
- `combo_c`: name char-4-grams (space-free core) + address word unigrams/bigrams.
- (tested) `name_c4`, `addr_tok`, `name_bi`, `name_tok`.
Current: `combo_c` K=60 + `combo` K=30 (~70 candidates/S1) → **stage-1 filter**.

Blocking recall table (30k train S1, full pool, with transliteration dict):
| config | pair recall | cand/S1 | oracle F0.5 |
|---|---|---|---|
| combo@20 + name_c4@10 | 0.9735 | 26.6 | 0.991 |
| combo_c@20 + combo@10 | 0.9770 | 22.5 | 0.9925 |
| combo_c@25 + combo@15 | 0.9813 | 30.0 | 0.9938 |
| combo_c@30 + combo@20 + c4@10 | 0.9846 | 42.4 | 0.9950 |
| union K60/30 (+c4, addr) | 0.9887 | 77 | 0.9962 |
| **K60/30 + stage-1 (p≥0.001)** | **0.9842** | **10.0** | 0.9952 |
| crowding +60% synthetic branches, combo_c25+combo15 | 0.9793 | 29.6 | 0.9931 |

### 4.3 Stage-1 filter (`stage1.py`)
Cheap LightGBM (250 rounds) on blocking cosines/ranks, 3 rapidfuzz scores (name token-set, address token-set,
compact-name ratio), house-number agreement, empty-address flag, per-S1 gaps; trained on 60k held-out train S1;
keep p ≥ 0.001 and ≤ 20 per S1 → **~12 candidates per S1** (test: France 16.7, India 11.6, US 10.4; 41.5% of
French S1 hit the 20 cap but the 20th candidate's final score is ~0). Recall after stage-1 on the full train
(model sample): 0.9819 (normal) / 0.9811 (crowded). This set is what candidate_pairs.tsv contains.

### 4.4 Pair features (`features.py`, ~95)
- Name: rapidfuzz ratio / partial / token_sort / token_set / Jaro-Winkler on name_norm and name_core; compact ratio &
  partial; exact core; acronym hit; first token equal; DBA best; **raw-name** ratio / token_sort / exact (legal form
  and punctuation kept).
- Address: ratio / partial / token_sort / token_set; empty flag; postcode state (equal / conflict / missing).
- House number: first-number equal/diff, |diff| (log), relative diff, S1 number present in candidate numbers,
  overlap/conflict, **small shift ≤30**, **digit substring (drop/add)**, digit Levenshtein, length diff, same length,
  **branch signature** (small shift + same street by number-free address token-set ≥90), cluster size (other
  candidates of the S1 with the same number), #candidates with the S1's number.
- Branch words: `xtok_*` = OOF target encoding of "extra tokens" (candidate name tokens not in S1 name) learned from
  TRAIN labels; `hmis_*` = **label-free** rate at which an extra token co-occurs with a changed house number, computed
  on each dataset's own candidates (works for French words: holding .99, distribution .99, participations .99,
  développement .83, groupe .82 vs associés .17, services .24, fils .22).
- Ambiguity: #S1 sharing the core name / exact address (S1 side and candidate side).
- **Context features**: within-S1 rank / gap to best / #close; **candidate-side** rank / gap / #S1 claiming the record
  — computed on the FULL candidate set for blocking cosines AND for name/compact/address/raw-name similarities.
  Candidate-side rank on the blocking cosine is the #1 feature ("does another S1 fit this record better?").
- Train-time augmentation ("crowd"): +50% synthetic fake branches in the train pool (nudged copies of distractors).

### 4.5 Model & decision
- LightGBM binary (lr 0.1, 127 leaves, min_data 50, feature/bagging 0.8, L2 1, deterministic), 5-fold GroupKFold by
  S1 on a sampled set of train S1 (800k S1 ≈ 9.6M pairs), OOF probabilities; final single model on all pairs.
- Decision (`decide.py`): greedy one-to-one (each pool record to its highest-scoring S1); keep top-1 if p ≥ t1;
  extra candidates if p ≥ t2 and p ≥ r·best; (t1, t2, r) tuned by coordinate descent on OOF macro-F0.5
  (typically t1≈0.72-0.76, t2≈0.70-0.74, r=0). Expected-F0.5 subset selection was worse (0.9763 vs 0.9771).
- Scores are well calibrated; only ~7% of candidate pairs have p in [0.1, 0.9].

### 4.6 Post-processing rules (`postprocess.py`) — applied to saved test scores
- **France house rule (data-driven)**: for any country whose accepted pairs show almost no house-number noise
  (accepted digit-drop pairs < 3 per 100 S1 — France 1.08 vs India 19.9 vs US 23.1, measured without labels), drop
  accepted pairs whose first house number differs from the S1's. France: 39,575 pairs dropped, 2,044 French S1
  become empty (French empty rate 5.0% → 5.8%, matching other countries).
- **US shift rule**: drop US accepted pairs with a house number shifted by 1-30 on the same street (64,911 pairs,
  2,562 US S1 become empty). Refined variant ("keep-only") never empties an S1 (only drops shifted pairs when the S1
  keeps a normal match), because train says "S1 whose ONLY matches are shifted" are 91.5% true.

---

## 5. How we evaluate (and what burned us)

- **The random validation split from the provided `evaluate.py make-split` is misleading**: it assigns distractors
  randomly, so ~80% of a val S1's fake branches land in the train side → val is easy. Rule baseline: val 0.773 vs
  public 0.679. We stopped using it.
- **Honest local metric = OOF macro-F0.5 on the FULL train** (all distractors present). Compare models on the same
  S1 sample (same seed) or on the S1 common to two OOF samples.
- **Every submission scored ~0.013-0.019 below its local OOF** (0.9736→0.955, 0.9769→0.963, 0.9773→0.964,
  ~0.9804→0.966).
- Simulators: country-held-out CV (train US→predict India and vice versa) drops 0.9752→0.9608 (all extra loss =
  false positives); crowded validation (synthetic branches) drops a normal model by 0.0048 (crowd-trained model
  recovers +0.0031; unseen-country +0.0017).
- **Adversarial validation** (classifier separating train pairs from test pairs using model features): France AUC
  0.998 (top shifts: address near-duplicates per S1 4.0→7.5; **learned branch-word score xtok_sum 2.14→0.36**;
  #candidates 13.9→17.5); US AUC 0.904 (name-duplicate counts 35→19 because US test has half the S1 of US train;
  #S1 claiming a record 46→38); India AUC 0.884.
- **Exact-attribution probes** (our most useful tool): F0.5 is per S1 and blocking/one-to-one never cross countries,
  so rows of different countries are independent.
  - France-empty probe P of file S: US+India F = (P − 0.1498·0.05)/0.8502; France F = (S − P)/0.1498 + 0.05.
  - To measure a change in one country exactly: keep the other countries' rows identical to an already-scored file.

---

## 6. Experiment log (local = OOF on full train; "crowded" = train pool with synthetic branches)

| tag | change | local OOF | India / US | public |
|---|---|---|---|---|
| exp01 | rule baseline: max(combo cosine, 0.8·name_c4), one-to-one | tune-OOF 0.680 (val 0.773) | – | **0.679** |
| exp02 | first LightGBM, 53 features | 0.962 | – | – |
| exp04 | + learned transliteration + name-dup, 250k S1 | 0.9736 | .9662/.9785 | **0.955** |
| exp05 | blocking combo_c25+combo15 + hmis, 300k S1 | 0.9769 | .9716/.9805 | **0.963** |
| exp06 | + branch-signature feats, − length feats | 0.9771 | .9719/.9805 | – |
| exp06+07 blend | exp07 = 600k S1 | ≈0.9773 | – | **0.964** |
| exp09 | wide blocking K60/30 + stage-1 (~12 cand/S1) | 0.9795 | .9763/.9817 | – |
| exp10-12 | + crowd 0.5 (300-400k S1) | 0.979 (crowded) | – | – |
| exp13-15 | crowd 0.5, 800k S1, seeds 3/4/5 | 0.9804/0.9800/0.9802 | .9776/.9822 | blend 13/14/15 **0.966** |
| exp16 | + cand-side name/address competition, address dup counts | 0.9807 | .9781/.9824 | – |
| exp17 | + raw-name features + digit-level house features | **0.98146** | .9788/.9832 | (base of best files) |
| exp18/19 | exp17 features, seeds 4/5 | 0.9812/0.9814 | – | blend 17/18/19 = 0.98145 on common S1 |
| exp20 | exp17 − learned xtok, counts clipped | 0.9813 | .9787/.9831 | not used (see §7.5) |
| exp21 | exp20 + 5-digit house-number fix | **0.9817** | .9785/.9838 | – |

Loss decomposition of our best local model (exp13/17, ≈0.019 total): recall-only 0.0115 (≈40% due to blocking
misses), non-singleton predicted empty 0.0039, a wrong match included 0.0027, singleton given a match 0.0014.
True pairs: blocking/stage-1 lose 1.87%, model rejects ~2.7% (64% of these are empty-address copies — mostly the
unresolvable ties; 25% "similar but rejected" house-number noise; 7% random names at the exact address; 2% native
script; 2% domains), false-positive pairs ≈0.35%. Model recall by record type: empty address 0.544 vs 0.99 for
every other type (junk address 0.992, native 0.991, domain 0.991, acronym 0.986, random name 0.969).

Blocking/stage-1 misses (share of ALL true pairs, exp17): both fields heavily noised 0.47%; empty address + different
name 0.33%; native-script name 0.31%; empty-address tie-type 0.23%; **"easy" (name ≥80 & address ≥80) 0.19%**
(e.g. "Maid Ice Cream | 7646 Blackstone Ave" vs "CREAM ICE MAID | 7646 BLACKSTOEN AVENUE"); random name at right
address 0.13%; name ok address different 0.13%; domains 0.10%.

---

## 7. Findings (chronological highlights, with evidence)

7.1 LightGBM ≫ rule (0.679 → 0.955).
7.2 Learned transliteration fixed Indian native-script names (India OOF 0.954 → 0.966).
7.3 Wide blocking + stage-1 beats narrow blocking on recall AND candidate size (0.9819 @ ~11/S1 vs 0.9756 @ 30).
7.4 Crowd training helps on test-like density (A/B +0.0031; unseen country +0.0017); more data (800k S1) helps.
7.5 **France**: our learned branch-word feature is blind to French branch words → exp17 accepted **23,364 French
pairs whose candidate ADDS a branch word (9 per 100 S1) vs 0 in US and 20 in India** (e.g. "HM Residence SAS |
30 Rue des Lilas" ← "HM RÉSIDENCE DÉVELOPPEMENT SAS | NO 32…"). Dropping xtok (exp20) fixed those (→839) but also
dropped 59,868 French pairs, 67% with the SAME house number (swapped generic word "4l Ecole SARL"→"4l Amicale Sarl"),
because the label-free detector is polluted on dense French streets ("club" .74, "nantes" .84) → not used.
7.6 **French copies carry no house-number noise → French house-diff acceptances are fake branches.** Rule applied
→ **public 0.966/0.967 → 0.968745** (France ≈ 0.925 → 0.938).
7.7 **US**: accepted "same street, number ±1..30" pairs per 100 S1: train 3.68 (precision 0.978) vs test 9.89;
India 5.87 vs 6.88. Split: "S1 whose ONLY matches are shifted" train 0.34/100 (precision 0.915, 13.6% singletons)
vs test 0.59; "shifted pairs of S1 that also have a normal match" train 3.34 (0.975) vs **test 9.20 (2.75×)**.
US shift rule → **public 0.970125** (+0.00138 overall = +0.0036 on US).
7.8 Things that did NOT explain the gap / did not help: French acronyms (95.5% accepted when = S1 initials at same
number), co-located S1 (normal behaviour), name sharing (FR ≈ IN), rejected copy-like records (≤ +0.002 even if all
right), blocking-score saturation (no cross-S1 ties), train/test similarity-distribution shift for US/IN (none),
test's extra pool (~1 record/S1, almost all LOW similarity: best sim ≥0.8 records per S1 US 1.66→1.78, IN 2.14→2.34),
record-format fingerprints (owned vs distractor AUC 0.78, already captured), duplicates, singleton predictability,
sibling tie-breaking, robust street-number parser (did not separate true vs fake shifted numbers in train),
"shared shifted number" (did not separate), expected-F0.5, per-country thresholds, unit/ID/row-order leakage.
7.9 France same-address records with a partly different name (similarity 50-90): France accepts 58.7% vs US/IN
86-88%. The rejected ones are mostly [same brand] + [different activity word] ("Veterans Musique EURL" vs "Veterans
Club EURL", "Chasse Comite SARL" vs "Chasse Culturelle SARL", "EHPAD Saint Julien" vs "élémentaire Saint Julien") —
look like different businesses; ~11.8k pairs, ±0.001 either way; not pursued.

---

## 8. Submissions (public LB)

| # | file | content | public |
|---|---|---|---|
| d1-1 | rule baseline | exp01 | 0.679 |
| d1-2 | exp04 | first good LightGBM | 0.955 |
| d1-3 | probe | exp04 with France emptied | 0.831 → US+IN 0.969, France 0.877 |
| d1-4 | exp05 | wider blocking + label-free branch words | 0.963 |
| d1-5 | exp06+07 blend | | 0.964 |
| d2-1 | blend exp13/14/15 | crowd training, 800k S1, stage-1 | 0.966 |
| d2-2 | probe | exp17 with France emptied | 0.836 → **US+IN 0.9745** |
| d2-3 | **Aprime** | exp17 + France house rule | **0.968745** → **France ≈ 0.938** |
| d2-4 | **C** | Aprime + US shift rule | **0.970125** (best) → US +0.0036 |
| d2-5 (planned) | blend exp17/18/19 + France rule + refined US rule (never empties an S1) | | expected ≈0.9705-0.971 |

Current decomposition: **US+India ≈ 0.976 (85% of S1), France ≈ 0.938 (15%)**. Leader 0.9907 ⇒ leader total loss
≈0.009; our US/IN loss alone ≈0.020 of the overall score, France ≈0.009.

---

## 9. Where the points are (our best estimate)

- US/India: local OOF ≈0.9815 (loss 0.0185) vs test ≈0.976 → a ~0.006 train→test gap (fake branches that differ only
  by a shifted number; partially fixed) + our modelling loss.
- Local modelling loss on US/IN ≈0.0185: unavoidable empty-address ties ≈0.004; blocking/stage-1 misses ≈0.004-0.005;
  non-singleton predicted empty ≈0.004; false positives ≈0.004; other recall ≈0.002.
- France ≈0.938: remaining ~0.04 deficit NOT identified by label-free EDA (France predicted matches per S1 3.29 vs
  US 3.34; empty 5.8% vs 5.5%).
- For a leader at 0.9907, US/IN must be ≈0.99+ and France ≈0.98+ — i.e. near-perfect except the unavoidable ties.

---

## 10. Ideas we have NOT done yet (our own list)

1. Loosen the stage-1 filter (min_p 0.0002-0.0005, cap 30) to recover blocking recall (~+0.001-0.002); add a
   character-level address space / phonetic keys for the "both fields noisy" misses.
2. Train with realistic S1-derived fake branches (S1 name ± branch word + number shifted 1-30 on the same street,
   coded as `--s1-branches`) so the model itself learns what the rules do.
3. India version of the shift rule (evidence says ~1 fake branch per 100 India S1 → +0.0005-class).
4. A transformer cross-encoder (e5-small/MiniLM, MIT/Apache) fine-tuned on hard pairs and applied only to the ~7%
   uncertain pairs (CPU/MPS only — feasibility unclear in 1 day).
5. Better candidate-side reasoning for France (co-located businesses; generic brand + activity names).

---

## 11. Code / files (for reference)

```
student_resource/
  CLAUDE.md, START_HERE.md, PROGRESS.md (full handover), evaluate.py (organiser-style scorer), utils/validate_submission.py
  code/business_entity_resolution/src/
    config.py  io_utils.py  normalize.py  blocking.py  stage1.py  features.py  model.py  decide.py
    postprocess.py  run_pipeline.py  sanity_check.py
  code/business_entity_resolution/dev/
    block_dev.py stage1_dev*.py country_cv.py country_params.py crowd_eval.py adversarial.py blend.py blend3_eval.py
    expf.py package.sh eda_types.py eda_empty.py eda_house.py eda_acronym.py eda_france.py eda_rejected.py
    eda_sibling.py eda_accepted.py eda_branch_cluster.py eda_street_number.py eda_fingerprint.py eda_blind2.py
    eda_shifted.py eda_shift_only.py eda_exact_addr.py eda_fr_exact.py eda_block_miss.py
  runs/<exp>_test/ (local): matching_results.tsv, candidate_pairs.tsv, test_scores.parquet, report/oof*.parquet
  submissions/<name>/: files + NOTE.md
```
Run: `run_pipeline.py --train-dir dataset/train --test-dir dataset/test --out-dir runs/X --crowd 0.5
--sample-seed 3 --model-s1 800000 [--profile exp17]`, then `postprocess.py --run runs/X --out out --house-rule
[--shift-rule US --shift-keep-only]`.

Key decision rule (pseudo-code):
```python
c = candidates sorted by score desc; c = c.drop_duplicates("pool_id")          # one-to-one
best = c.groupby("s1").score.transform("max")
keep = where(score >= best, score >= t1, (score >= t2) & (score >= r * best) & (best >= t1))
```

---

## 12. Assumptions we are least sure about (please challenge)

1. That the ~0.04 France deficit is not visible without labels — maybe there is a France-specific copy-noise
   pattern we have not modelled (e.g. activity-word substitution being COPY noise in France, or region vs
   département mismatches hurting address similarity).
2. That empty-address ties are unresolvable (we tested raw names, siblings, IDs, row order, source counts).
3. That the leaders use a stronger pair model (cross-encoder / LLM on GPU) rather than a data trick we missed.
4. That the train→test difference for US is only "more fake branches differing only by a shifted number" — maybe
   test copies ALSO have a different noise mix (we saw 4.1× more shifted look-alikes; only 22% carry an extra word).
5. That our decision thresholds (tuned on train OOF) are near-optimal for test; test's density is higher.
6. That per-S1 F0.5 with ~3.5 true matches makes adding a 50%-likely extra match a bad bet (it is under our
   calibration).

**Remaining budget:** ~1 day, ~5-6 submissions, CPU-only laptop, ~1 heavy run (≈1-2 h) at a time.
