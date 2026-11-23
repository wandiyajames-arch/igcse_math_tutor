# Evaluation harness (GAIE 2027)

Implements the evaluation protocol **v1.1**, agreed with Dr. Wade on 21 November 2026 and frozen on
23 November (fingerprints in the public `protocol-v1.1-freeze` release). Section numbers below refer to
that protocol.

**Which grader is tested.** The paper evaluates Angelah's grader (`src/grading/` in
`angelah/igcse-math-tutor-research`, commit `439ce32`, tag `gaie-eval-v1`). This folder runs unchanged on
her repository and on this one; `grader_adapter.py` detects which grader is in `src/` and is the only place
that talks to it. Nothing in `evaluation/` edits `src/`, so the frozen grader stays frozen.

Install the extra tools in a separate environment so the project's pinned SymPy is not touched:
`pip install -r requirements-eval.txt`.

## What covers what

| Protocol item | Where |
|---|---|
| §2 Systems: our grader, Math-Verify, CompassVerifier, Qwen3-4B, Qwen2.5-7B, string match, cascade | `systems.py`, `judges.py` |
| §2 k = 5 self-consistency; vote-level tie-break by mean token log-probability [Change 3] | `judges.aggregate` |
| §2 Model digests and seeds recorded for every judge run | `runs/<judge>.meta.json` |
| §3 Item rules: exclusions with reasons; parts the grader cannot express stay in (`"type": "unsupported"`) | *Excluded* sheet, `grader_adapter.is_unsupported` |
| §3 Size cap: at most 100 parts, stratified by paper, seed 2027 | `author.py sample` |
| §3 Value-only / form-sensitive tag set from the mark scheme | `tag` column |
| §4 Specs from the mark scheme only; self-check grades only the mark-scheme answer | `author.py build` |
| §4 5 answers per part: ms_form, alt_form, examiner_report (cited), generic (named error) | *Responses* sheet, `data.validate` |
| §4 Intended label for every answer [Addition] | `intended_label`, `intended_category` |
| §4 Blind typing: about 1 in 5 answers typed by classmates from descriptions [Change 1] | `blind_typing.py` |
| §4 Hash freezes of specs, answers, gold labels, grader and prompts; published publicly [Change 2] | `freeze.py` |
| §5 Two raters, blinded shuffled sheets; raw agreement, κ, AC1; adjudication | `sheets.py`, `adjudicate.py` |
| §6 Matched coverage: threshold closest to the grader's without falling below; coverage reached shown [Change 3] | `metrics.operating_point` |
| §6 Primary outcome: false-accept rate, clustered bootstrap CI of the difference; McNemar secondary | `report.py` |
| §6 Abstention on correct answers; exact CIs; rule of three | `metrics.py` |
| §6 Confirmatory breakdowns: value-only/form-sensitive, paper, classmate-typed subset, cascade on abstentions, source (if generic fallback) | `report.py` |
| §6 RQ2: reason agreement, 6 × 6 confusion matrix, κ | `metrics.reason_confusion` |
| §6 Intended vs gold label [Addition] | `metrics.intended_vs_gold` |
| §7 RQ3 table: median and p95 time, cold start, peak memory, model size, digest | `report.py` |
| §7 RQ4: controller vs instructed tutor prompt [Change 4]; leaks detected by the grader; human check | `leakage.py` |

## Steps

**1. Items (23–24 Nov).** Copy `templates/test_set_authoring.xlsx` to `data/test/authoring.xlsx`. On *Items*,
one row per included part with its `tag` and spec (patterns on *Spec examples*; use
`{"type":"unsupported","value":"..."}` for a part the grader cannot express). On *Excluded*, every excluded
answer line with its reason. If more than 100 parts are eligible:

```bash
python -m evaluation.author sample evaluation/data/test/authoring.xlsx evaluation/data/test
```

**2. Specs, then freeze them and publish the hash.**

```bash
python -m evaluation.author build evaluation/data/test/authoring.xlsx evaluation/data/test   # fix every PROBLEM
python -m evaluation.freeze specs evaluation/data/test
```

**3. Answers (24–25 Nov).** On *Responses*, 5 per part, each with a plain `description` and its
`intended_label`. Write `data/test/typing_guide.txt` **without looking at the grader** ("type it as a student
would on a phone or keyboard"). Then:

```bash
python -m evaluation.author build evaluation/data/test/authoring.xlsx evaluation/data/test
python -m evaluation.blind_typing assign evaluation/data/test --typists 4    # send typing_sheet_typist*.xlsx
python -m evaluation.blind_typing collect evaluation/data/test             # when the classmates send them back
python -m evaluation.freeze answers evaluation/data/test                    # refuses if anything is missing
```

Publish every `FREEZE*.json` the day it is made (a public GitHub release and OSF).

**4. Raters (25–30 Nov).** `python -m evaluation.sheets evaluation/data/test --rater rater1` (and `rater2`). Export:

```bash
python -c "from evaluation.sheets import sheet_to_csv; sheet_to_csv('evaluation/data/test/labelling_rater1.xlsx','evaluation/data/test/labels_rater1.csv')"
python -m evaluation.adjudicate evaluation/data/test     # κ, AC1 + disagreements.csv
```

The raters discuss `disagreements.csv`; only the cases they cannot settle go to Dr. Wade (both labels and
reasons, without rater names; more than about 40 means calibration failed). Then:

```bash
python -m evaluation.adjudicate evaluation/data/test --decisions evaluation/data/test/disagreements.csv
python -m evaluation.freeze gold evaluation/data/test
```

**5. Development set first.** Run every judge on `data/dev/` (2024 papers): timing, prompt parsing,
Qwen3's thinking mode, whether the tie-break log-probabilities come back, and the headline judge (lowest
AURC). Do not change prompts after that (the freeze would fail).

**6. Systems (from 25 Nov, need no labels).** On the laptop:

```bash
python -m evaluation.systems evaluation/data/test --system ours
python -m evaluation.systems evaluation/data/test --system math_verify
python -m evaluation.systems evaluation/data/test --system string_match
python -m evaluation.systems evaluation/data/test --system qwen3_4b   --backend ollama:qwen3:4b
python -m evaluation.systems evaluation/data/test --system qwen25_7b  --backend ollama:qwen2.5:7b
python -m evaluation.systems evaluation/data/test --system compass_3b --kind compass --backend hf:opencompass/CompassVerifier-3B
```

Try `--limit 20` first; runs print the time left and resume where they stopped. Fallback (agreed): if they
cannot finish by 1 December, run the judges with `notebooks/run_judges_kaggle.ipynb`; RQ3 timings stay on
the laptop.

**7. RQ4.** `python -m evaluation.leakage generate evaluation/data/test --backend ollama:qwen3:4b`, fill
`manual_leak` (0/1) where `check_by_hand = 1`, then `python -m evaluation.leakage score evaluation/data/test`.

**8. Results, once.** `python -m evaluation.report evaluation/data/test --headline-judge <judge chosen on dev>`
writes `results/results.md`, `results/tables.tex`, `results/risk_coverage.png` and `results/our_errors.csv`.
Its first line says whether the freeze check passed.

## Notes

* `data/pilot_PIPELINE_TEST_ONLY/` is built from the grader's own regression cases. It only checks that the
  scripts run; **never report its numbers**.
* CompassVerifier's prompt is copied verbatim from its repository (Apache-2.0); its "C: invalid" votes count
  as abstentions. Like the other judges it is sampled k = 5 at temperature 0.7.
* Math-Verify receives the mark-scheme answer and the student answer converted to `$…$` LaTeX; it abstains
  when either cannot be parsed. String match never abstains.
* A spec mistake found after the freeze counts as a grader error. It is reported, never fixed.
