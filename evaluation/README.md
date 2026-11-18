# Evaluation harness (GAIE 2027)

Implements the evaluation protocol agreed with Dr. Wade, with the additions from Angelah's assessment
(`docs/research/gaie2027-assessment-and-plan.md` in her repository).

**Which grader is tested.** The paper evaluates Angelah's grader (`src/grading/` in
`angelah/igcse-math-tutor-research`). This folder runs unchanged on her repository and on this one (which
holds a stop-gap rebuild); `grader_adapter.py` detects which grader is in `src/` and is the only place
that talks to it. Nothing in `evaluation/` edits `src/`, so the frozen grader stays frozen.

Install the extra tools in a separate environment so the project's pinned SymPy is not touched:
`pip install -r requirements-eval.txt`.

## What covers what

| Protocol item | Where |
|---|---|
| Dev/test split (2024 papers dev, June 2025 papers test) | `split` column, `author.py` |
| Specs written from the mark scheme only, checked, then **frozen before any answer exists** | `author.py` self-check, `freeze.py specs` |
| Answers, grader, prompts **frozen by hash before any run**; checked again in every report | `freeze.py answers` / `verify` |
| Errors taken from Principal Examiner Reports, each cited | `source = examiner_report` + `source_note` (enforced) |
| Out-of-scope answer lines (graphs, constructions, …) listed with a reason | *Excluded* sheet → `excluded.csv` |
| Two independent raters, adjudication, 6 reason categories | `sheets.py`, `adjudicate.py`, `categories.py` |
| Rater agreement: raw %, Cohen's κ **and Gwet's AC1** (stable when labels are imbalanced) | `data.agreement` |
| Every judge gets the question and the mark-scheme answer | `judges.py` |
| LLM judges can abstain: self-consistency k = 5 | `judges.py` (`aggregate`) |
| Baselines: Math-Verify, CompassVerifier, Qwen3-4B, Qwen2.5-7B | `systems.py`, `notebooks/run_judges_kaggle.ipynb` |
| Cascade (ours first, LLM only on abstentions) as a headline row | `metrics.cascade`, `report.py` |
| Risk–coverage curves, AURC, comparison at matched coverage, risk at fixed coverages | `metrics.py`, `report.py` |
| False accepts and false rejects reported separately, with counts | `metrics.summarise`, `metrics.error_counts` |
| 95% bootstrap CIs clustered by question part; **paired risk difference as the primary test**, McNemar secondary | `metrics.py` |
| Rare events: exact (Clopper–Pearson) CIs and the rule of three when there are 0 false accepts | `metrics.error_counts` |
| **Two coverage figures**: over specifiable parts and over all answer lines | `report.py` (Scope) |
| **Value-only vs form-sensitive** results (fair to Math-Verify, which checks value only) | `grader_adapter.stratum` |
| Breakdowns by answer type, response source and syllabus topic | `report.py` |
| Every wrong decision of our grader listed for the error analysis | `results/our_errors.csv` |
| RQ2 reason agreement | `report.py` |
| Efficiency as a table, not an RQ | `report.py` |
| RQ4 light leakage study: naive tutor vs the app's controller vs controller + output guard | `leakage.py` |

## Steps

**1. Write the test set in Excel.** Copy `templates/test_set_authoring.xlsx` to `data/test/authoring.xlsx`.
*Items*: one row per answer line you can specify (copy a pattern from *Spec examples*).
*Excluded*: one row per answer line you cannot specify, with a reason. Then build and check:

```bash
python -m evaluation.author build evaluation/data/test/authoring.xlsx evaluation/data/test
```

Every `PROBLEM:` line must be fixed. The self-check only grades the **mark-scheme answer** against its spec,
never a candidate answer, so it cannot leak how the grader treats the test responses.

**2. Freeze the specs, then write the answers, then freeze the answers.**

```bash
python -m evaluation.freeze specs   evaluation/data/test     # before writing any candidate answer
# ... fill the Responses sheet, run step 1 again ...
python -m evaluation.freeze answers evaluation/data/test     # before running any system
```

Keep `FREEZE_specs.json` and `FREEZE.json`; they go in the paper's appendix. Do the same for `data/dev/` (2024).

**3. Raters.** `python -m evaluation.sheets evaluation/data/test --rater rater1` (and `rater2`) makes a shuffled
sheet with dropdowns; raters never see sources or system verdicts. Export and adjudicate:

```bash
python -c "from evaluation.sheets import sheet_to_csv; sheet_to_csv('evaluation/data/test/labelling_rater1.xlsx','evaluation/data/test/labels_rater1.csv')"
python -m evaluation.adjudicate evaluation/data/test            # κ, AC1 + disagreements.csv
python -m evaluation.adjudicate evaluation/data/test --decisions evaluation/data/test/disagreements.csv   # gold.csv
```

**4. Development set first.** Run everything on `data/dev/` to check prompt parsing and timing and to choose the
judge for the headline cascade. Do not change prompts after looking at test results (the freeze would fail).

**5. Run every system on the test set.** The judges need no labels, so they can run while the raters work.

* Easiest: open `notebooks/run_judges_kaggle.ipynb` on Kaggle (free GPU) – it runs all judges plus our grader
  and Math-Verify and gives you a zip of `runs/`.
* Or locally: `--system ours`, `--system math_verify`, and for each judge
  `python -m evaluation.systems evaluation/data/test --system qwen3_4b --backend ollama:qwen3:4b`
  (try `--limit 20` first; it prints the time left; runs resume where they stopped).
  CompassVerifier: `--system compass_3b --kind compass --backend hf:opencompass/CompassVerifier-3B`.
  Use one backend for all judges (all Kaggle or all Ollama) and say which in the paper.

**6. Leakage (RQ4).**

```bash
python -m evaluation.leakage generate evaluation/data/test --backend ollama:qwen3:4b
# fill manual_leak (0/1) for every row with check_by_hand = 1
python -m evaluation.leakage score evaluation/data/test
```

**7. Results.** `python -m evaluation.report evaluation/data/test --headline-judge <judge chosen on dev>` writes
`results/results.md` (all tables), `results/tables.tex` (paste into the paper), `results/risk_coverage.png` and
`results/our_errors.csv`. The first line of `results.md` says whether the freeze check passed.

## Notes

* `data/pilot_PIPELINE_TEST_ONLY/` (built by `python -m evaluation.pilot`) uses the developers' own regression
  cases with simulated raters and a mock judge. It only shows that every script runs; **never report its numbers**.
* CompassVerifier's prompt is copied verbatim from its repository (Apache-2.0). Its authors recommend greedy
  decoding; we sample k = 5 at temperature 0.7 like the other judges so every judge can abstain the same way.
  Its "C: invalid" votes count as abstentions.
* Math-Verify receives the mark-scheme answer and the student answer converted to `$…$` LaTeX; it abstains when
  either cannot be parsed.
* A spec mistake found after the freeze counts as a grader error. It is reported, never fixed.
