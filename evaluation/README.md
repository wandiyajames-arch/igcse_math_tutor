# Evaluation harness (GAIE 2027)

Implements the evaluation protocol agreed with Dr. Wade. The grader under test is the frozen release
[`gaie-eval-v1`](https://github.com/wandiyajames-arch/igcse_math_tutor/releases/tag/gaie-eval-v1); nothing in
`evaluation/` changes `src/grading/`.

| Protocol item | Where |
|---|---|
| Dev/test split (2024 papers dev, 2025 papers test, test untouched until the freeze) | `split` column, `author.py` |
| Errors taken from Principal Examiner Reports, each cited | `source = examiner_report` + `source_note` (enforced) |
| Two independent raters, κ, 6 reason categories, adjudication | `sheets.py`, `adjudicate.py`, `data.py` |
| Every judge gets the question and the mark-scheme answer | `judges.py` |
| LLM judges can abstain: self-consistency k = 5, abstain below τ | `judges.py` (`aggregate`, `decide`) |
| Baselines: Math-Verify, CompassVerifier, Qwen3-4B, Qwen2.5-7B | `systems.py` |
| Cascade (ours first, LLM only on abstentions) as a headline row | `metrics.cascade`, `report.py` |
| Risk–coverage curves, AURC, comparison at matched coverage | `metrics.py`, `report.py` |
| False accepts and false rejects reported separately | `metrics.summarise` |
| 95% bootstrap CIs clustered by question part; McNemar on paired decisions | `metrics.py` |
| RQ2 reason agreement; breakdown by answer type and source | `report.py` |
| Efficiency as a table, not an RQ | `report.py` |
| RQ3 light leakage study (50 items, grader detects leaks) | `leakage.py` |

## Steps

**1. Write the test set in Excel.** Open `templates/test_set_authoring.xlsx` (copy it into `data/test/`).
One row per question part on *Items* (copy a spec pattern from *Spec examples*), one row per candidate
answer on *Responses*. Then:

```bash
python -m evaluation.author build evaluation/data/test/authoring.xlsx evaluation/data/test
```

Fix every `PROBLEM:` line before going on. Do the same for `data/dev/` with the 2024 papers.

**2. Raters.** `python -m evaluation.sheets evaluation/data/test --rater rater1` (and `rater2`) makes a
shuffled sheet with dropdowns; raters never see sources or system verdicts. Export each finished sheet:

```bash
python -c "from evaluation.sheets import sheet_to_csv; sheet_to_csv('evaluation/data/test/labelling_rater1.xlsx','evaluation/data/test/labels_rater1.csv')"
python -m evaluation.adjudicate evaluation/data/test            # κ + disagreements.csv
python -m evaluation.adjudicate evaluation/data/test --decisions evaluation/data/test/disagreements.csv   # gold.csv
```

**3. Development set first.** Run the judges on `data/dev/` to check the prompt parsing and to pick the
judge used in the headline cascade. Do not change prompts after looking at test results.

**4. Run every system on the test set.** Rule systems run anywhere:

```bash
python -m evaluation.systems evaluation/data/test --system ours
python -m evaluation.systems evaluation/data/test --system math_verify
```

LLM judges through Ollama (`ollama pull qwen3:4b` and `ollama pull qwen2.5:7b` first). Time 20 responses
before the full run (`--limit 20`); runs resume where they stopped.

```bash
python -m evaluation.systems evaluation/data/test --system qwen3_4b  --backend ollama:qwen3:4b
python -m evaluation.systems evaluation/data/test --system qwen25_7b --backend ollama:qwen2.5:7b
```

CompassVerifier through transformers (a free Kaggle/Colab GPU is enough for the 3B model):

```bash
python -m evaluation.systems evaluation/data/test --system compass_3b --kind compass --backend hf:opencompass/CompassVerifier-3B
```

**5. Results.** `python -m evaluation.report evaluation/data/test --headline-judge <judge chosen on dev>`
writes `results/results.md`, `results/tables.tex` (paste into the paper) and `results/risk_coverage.png`.

**6. Leakage (RQ3).** `python -m evaluation.leakage generate evaluation/data/test --backend ollama:llama3.2:1b`,
fill the `manual_leak` column (0/1) in `leakage.csv`, then `python -m evaluation.leakage score evaluation/data/test`.

## Notes

* `data/pilot_PIPELINE_TEST_ONLY/` is built from the grader's own regression cases. It only checks that the
  scripts run; **never report its numbers**.
* CompassVerifier's prompt is copied verbatim from its repository (Apache-2.0). Its authors recommend greedy
  decoding; we sample k = 5 at temperature 0.7 like the other judges so that every judge can abstain the same
  way. Its "C: invalid" votes count as abstentions.
* Math-Verify receives the mark-scheme answer and the student answer converted to `$…$` LaTeX; it abstains when
  either cannot be parsed.
