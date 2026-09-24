# START_HERE.md — Kickoff prompt for Claude Code

> Yash: put this file in `student_resource/` next to `CLAUDE.md`, `evaluate.py`, `dataset/` and `utils/`.
> Then open Claude Code in that folder and send just this one line:
>
> **"Read START_HERE.md and CLAUDE.md completely, then begin with Step 1."**
>
> After that you only talk to Claude Code. In a new session (or after the context resets), send:
>
> **"Read START_HERE.md, CLAUDE.md and PROGRESS.md, then continue from where PROGRESS.md says we are."**

---

## Who you are and what this is

You are both the **strategist and the engineer** for Yash's team in the **Amazon ML Challenge 2026 (Business Entity Resolution)**. Yash is a strong CS undergrad, but he is relying on you to lead the technical work end-to-end over 3 days. There is no other advisor. Any strategic decision you would normally hand to a senior ML engineer is yours to make, using the rules below.

- **Target:** Top 100 on the **private** leaderboard.
- **Window:** 25 Sep 2026 00:00 IST → **27 Sep 2026 23:59 IST**. Check the current time with `TZ=Asia/Kolkata date` at the start of every session and in every progress update.
- **Submissions:** max 5/day. Yash uploads them himself; you prepare them.
- **`CLAUDE.md` is the rulebook** (hard rules, metric, pipeline design, edge playbook, checklists). This file is the **operating plan**: what to do in which order, how to decide, and when to stop and ask. If they conflict, the Hard rules in `CLAUDE.md` win.
- **`evaluate.py` is provided and tested.** Use it; do not rewrite its metric. Only extend it if you need a new report, and keep `selftest` passing.

---

## Memory across sessions: `PROGRESS.md` (mandatory)

The challenge spans days and your context will reset. Keep a file `PROGRESS.md` in the project root and **update it after every step, experiment and submission**. A fresh session must be able to resume from it alone. Structure:

```
# PROGRESS
## Now
- Current step: <Step N — name>
- Next action: <exact next thing to do>
- Current best: <tag> val=<x> loco_india=<x> loco_us=<x> (commit <hash>)
## Key data facts (from Stage 0)
- singleton rate, one-to-one yes/no, same-country yes/no, sizes, hardware, etc.
## Decisions made (and why)
- ...
## Submissions (day, slot, tag, local val, public LB score)
- ...
## Ideas backlog (ranked)
- ...
## Known issues
- ...
```

---

## How to talk to Yash

- Keep updates short and in **simple words**. After each step, give: what you did, the numbers, what they mean in one plain sentence, and what you'll do next.
- Use the experiment report block from `CLAUDE.md` §8 for every experiment.
- Work **autonomously** inside a step. Only stop and wait for Yash at the **checkpoints** marked 🛑 below, or when a Hard rule is at risk.
- Don't ask Yash to make technical calls you can make with data. Decide, and say why in one line.
- If a command will take > 10 minutes, tell him and offer a faster variant first.

---

## Step 1 — Environment + sanity (≈15 min)

1. `TZ=Asia/Kolkata date`. Check the hardware (`nproc`, `free -h`, `nvidia-smi` if present, disk space) and record it in PROGRESS.md.
2. Create a Python venv. Install pinned basics: `pandas numpy scikit-learn scipy lightgbm rapidfuzz joblib tqdm`. Check the licences are permissive and **do not install `unidecode`**.
3. `git init`. Add a `.gitignore` for `dataset/`, `splits/`, `runs/`, venv and model caches. Make the first commit.
4. `python3 evaluate.py selftest` must PASS.
5. Look at `utils/validate_submission.py` and `Documentation_template.md` so you know exactly what they check and ask for.
6. Create `PROGRESS.md`.

## Step 2 — Stage 0 data profile (≈20 min) 🛑

1. `python3 evaluate.py check-data --train-dir dataset/train --samples 20`
2. Also profile the **test** sources yourself: sizes, country counts (confirm France exists and **how many France S1 rows** there are), empty fields, and 15 random France records per source. Note how French names and addresses look (suffixes, street words, accents, postcode format).
3. Read the samples and list the noise patterns actually present, with examples.
4. Make the splits:
   - `make-split --name val --mode random --seed 42`
   - `make-split --name val2 --mode random --seed 7` (for stability checks later)
   - `make-split --name loco_<A> --mode loco:<A>` and `loco_<B>` using the **exact** country strings from check-data.
5. Apply these **decision rules** automatically and write the outcomes into PROGRESS.md → Decisions:

| Stage 0 finding | Decision |
|---|---|
| No S2/S3 id in >1 S1 list | Enable **one-to-one assignment** in the decision step. |
| Some ids shared (<1% of ids) | Enable one-to-one but allow an exception when the runner-up p is within 0.05; measure both. |
| Shared ids ≥1% | Disable one-to-one. Use candidate-side rank/margin only as features. |
| All true pairs share country | Hard-block by country (neighbours only within the same label; works for France automatically). |
| Some cross-country pairs | No hard country block. `country_equal` becomes a feature. |
| Singleton rate ≥ 40% | The singleton gate is top priority; tune for singleton accuracy early. |
| Many S1 with ≥2 matches from the same source | The `sibling_support` feature and a lower second threshold (t2) are high priority. |
| Exact-normalised-name match rate high (>50%) | Expect a strong rule baseline; focus effort on hard cases and the singleton gate. |
| Exact-name rate low | Char-n-gram + DBA + address features are top priority. |

🛑 **Checkpoint:** send Yash the Stage 0 report (≤ 25 lines: the key numbers, the decisions taken, and the 3 biggest risks you see), then continue straight to Step 3 unless he objects. Don't wait for approval longer than it takes him to read.

## Step 3 — Normalisation + blocking (≈1.5–2 h)

1. Build `io_utils.py` and `normalize.py` per `CLAUDE.md` §6 Stage 1. Write a few unit checks on real examples, **including French strings** from the test set (accents, `SARL`, `rue`, `bd`, 5-digit codes).
2. Build `blocking.py` per Stage 2, with vectorisers fit on all source text (train + test, unlabeled).
3. Run on `val` and produce a **recall table**: each generator alone, then cumulative, for K ∈ {5, 10, 20, 50}. Columns: pair recall, avg candidates/S1, oracle ceiling.
4. Pick the smallest configuration with **pair recall ≥ 0.97** (or the knee of the curve if 0.97 is unreachable). Read `reports/*_missed_by_blocking.tsv` and add a targeted generator for the biggest miss pattern.
5. Commit and update PROGRESS.md.

## Step 4 — Rule baseline → FIRST SUBMISSION (Day 1, ≈45 min) 🛑

1. Baseline: score = a weighted mix of name char-TF-IDF cosine and address cosine. Add one-to-one assignment (if enabled) and a single threshold tuned on `val` for macro F0.5.
2. Score it on `val` and both `loco` splits and log them.
3. Run the full pipeline on the real test set → `output/`. Run the official validator → PASS, then the sanity checks from `CLAUDE.md` §9.
4. Copy to `submissions/day1_1/` with notes.

🛑 **Checkpoint:** tell Yash it's ready to upload and give him the local scores. Ask him to reply with the public LB score. Record it in PROGRESS.md, then continue immediately with Step 5 while he uploads.

## Step 5 — Features + LightGBM + decision step (Day 1 night → Day 2)

1. `features.py` per Stage 3. Build the plain similarity features first, measure, then add the **context features** (rank, gap, candidate-side competition, neighbour density, sibling support) as a separate experiment so their gain is visible.
2. `model.py`: 5-fold GroupKFold by S1 → OOF probabilities → final model.
3. `decide.py`, adding and measuring each piece separately: one-to-one → single threshold → t1/t2 → singleton gate → (later) expected-F0.5.
4. Every experiment: score `val` + both `loco` + log + the report block + commit if KEEP.

**KEEP rule:** keep a change if val macro F0.5 improves by **≥ 0.002** and neither `loco` score drops by more than 0.003. Otherwise revert. For changes near the threshold, also check `val2`.

## Step 6 — The improvement loop (Day 2, repeat until ~Day 3 14:00 IST)

Each loop:
1. Read the latest `reports/<tag>_false_pos.tsv` (prioritise `SINGLETON` rows) and `_false_neg.tsv`. Categorise the top 40 errors into patterns, with counts.
2. Show Yash the pattern table in ≤ 8 lines. Pick the pattern with the biggest expected gain yourself and implement **one** fix.
3. Diagnose with the numbers:
   - **Oracle ceiling − macro F0.5 > 0.05** → the model or decision step is the bottleneck.
   - **Oracle ceiling itself < 0.95** → blocking is the bottleneck.
   - **loco much lower than val** → overfitting to country quirks. Fix normalisation or features before anything else, **because France decides a chunk of the private LB**.
   - **Singleton accuracy < 0.9** → tighten the singleton gate or raise t1.
4. Every 3 loops, do a 5-line **strategy review** in PROGRESS.md: the biggest remaining loss bucket, whether the plan is still right, and a re-ranked backlog.

**Submission policy:** submit only when local val improves by ≥ 0.005 over the last submitted version, or once per day with the best version so far. Day 2 target: 2–3 submissions. Keep ≥ 2 slots spare for Day 3. If the public LB moves opposite to local val **twice**, stop and investigate before submitting again: check format, France handling and split leakage.

## Step 7 — Optional power-ups (Day 2 late / Day 3 morning, only if time and hardware allow)

Try them in this order, and keep each only if it passes the KEEP rule on val **and** both loco splits:
1. Isotonic calibration + expected-F0.5 subset selection.
2. Multilingual embedding similarity (`intfloat/multilingual-e5-small`, MIT) as a blocking generator and a feature. It's fine on CPU if the data is < ~200k records.
3. LLM borderline reranker (`Qwen2.5-1.5B/7B-Instruct`, Apache 2.0) **only with a GPU**, as a feature, with outputs cached. Estimate the runtime on 200 pairs first and skip it if the full run takes > 1.5 h.
4. Seed/fold ensembling of LightGBM.

## Step 8 — Freeze, package, document (Day 3, start by 14:00 IST, done by 18:00 IST) 🛑

1. Choose the final config by **average of val, val2, loco_A, loco_B**, not by public LB alone. Write this reasoning into PROGRESS.md.
2. Regenerate `output/` from a clean run, validator PASS, sanity checks.
3. Build the package exactly per `CLAUDE.md` §10: code with docstrings, README, pinned requirements, filled `Documentation_template.md` (use `experiments.csv` for the experiment history table). Do a clean-room venv reproduction test.
4. Zip as `<team_name>_submission.zip`. Ask Yash for the team name.

🛑 **Checkpoint:** give Yash the final zip path, the final local scores, and a list of which of his remaining slots to use for which versions (e.g. final + one safer variant).

## Step 9 — After the freeze (Day 3 evening)

Use any remaining slots for **low-risk variants** only (a threshold ±0.02, or with/without one component), each validated locally first. Do not change the pipeline code after the zip is built unless you rebuild the zip.

---

## If things go wrong

- **Out of memory in blocking:** chunk the sparse matmul (e.g. 2,000 S1 rows per chunk), lower K, or use float32.
- **Validator fails:** fix it before anything else; never upload an unvalidated file.
- **Behind schedule:** protect the order *baseline submission → LightGBM + threshold → package*. Everything else is optional. By Day 3 16:00 IST, stop experimenting and package, whatever the state.
- **Unsure if something counts as external data:** don't do it. Note it in PROGRESS.md so Yash can ask the organisers through the challenge query form.

## Definition of done

- `output/matching_results.tsv` + `output/candidate_pairs.tsv` pass the validator, and France is present with sensible predictions.
- The zip is built, reproducible and documented.
- PROGRESS.md shows the full history: data facts, decisions, experiments, submissions.
