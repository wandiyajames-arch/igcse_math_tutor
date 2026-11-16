# IGCSE Math Pro

A local-first tutoring system for Cambridge IGCSE Mathematics (0580). Its core is a **deterministic answer grader**: it decides whether a student's final answer is correct, says why it is wrong, or **abstains** when it cannot be certain. A local LLM (Ollama) is used only to explain, never to judge correctness.

## What is built

| Part | Status | Where |
|---|---|---|
| Deterministic grader (27 verdict reasons, 9 answer types) | Built and tested | `src/grading/` |
| Answer specifications (numeric, angle, bearing, exact, expression, solutions, coordinates, vector, ratio) | Built and tested | `src/models/answers.py` |
| Sandbox worker (separate process, 256 MiB memory limit, 2 s timeout, no file writes) | Built and tested | `src/grading/worker.py`, `sandbox.py` |
| RAG chatbot over 13 Cambridge PDFs (Gradio + Ollama + Chroma) | Original prototype | `src/agent/`, `src/rag/` |
| Tutoring controller (attempt → hints → reveal) | Not yet in this repository | — |

## How the grader works

1. **Safe parser** (`parser.py`) — a hand-written tokenizer and precedence parser. Student text is never passed to `eval`, `sympify` or `parse_expr`. Ambiguous notation abstains instead of being guessed (`1/2x`, `x^2y`, `1,5`, `2 1/3`, `√2x`). Limits: 200 characters, 100 tokens, bracket depth 10, 15-digit numbers, exponent literals up to 20.
2. **Numeric answers** (`numeric.py`) — exact rational arithmetic, no floats. π and square roots use certified enclosures (integer arithmetic), refined at 10⁻⁴⁰, 10⁻⁸⁰ and 10⁻¹⁶⁰; if a decision is still uncertain the grader abstains. Handles s.f./d.p., half-up rounding, ranges, units, three-figure bearings.
3. **Exact answers** (`exact.py`) — π, surd and simplest-fraction forms checked on the student's own expression tree, then compared as exact canonical forms.
4. **Algebra** (`expression.py`) — equivalence via `expand`/`cancel`/`simplify` = 0; `WRONG_VALUE` only with an exact counterexample at fixed probe points (or a numeric one plus a failed symbolic check); otherwise abstain. Form checks for simplified, expanded and factorised-over-ℚ answers.
5. **Structured answers** (`structured.py`) — solution sets (maximum bipartite matching), ordered pairs, vectors, ratios in simplest form.

Check order is fixed and the first failure wins (spec self-check → parse → structure → unit → supported structure → form → simplest form → value), so e.g. `NOT_SIMPLIFIED` never implies the value was right.

## Usage

```python
from src.grading.grader import grade, grade_safe

spec = {"type": "expression", "value": "7a-6b", "variables": ["a", "b"]}
grade_safe(spec, "2a+3b+5a-9b")   # incorrect / not_simplified (runs in the sandbox)
grade({"type": "ratio", "parts": [2, 3, 4]}, "20:30:40")   # incorrect / not_simplified
```

Use `grade_safe` for untrusted input; `grade` runs in the current process.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

The suite has 178 tests: parser ambiguity and limits, 25 hostile inputs, 3,000 fuzzed inputs, certified π/√ enclosures, 87 labelled cases with their marking basis (`tests/grading_cases/cases.json`), property tests, and sandbox tests (timeout, memory limit, file-write block, concurrency). The labelled cases are **developer-written regression tests, not an independent evaluation**.

## Marking rules

Rules come from the 0580 syllabus (2025–2027) and mark-scheme conventions (3 s.f. default, 1 d.p. angles, three-figure bearings, π from the calculator or 3.142, simplest form, oe/cao/awrt), plus seven grading-policy rules listed in `src/models/answers.py` (`POLICY_RULES`). These are our grading decisions, not official Cambridge marking rules.

## Running the chatbot prototype

```bash
ollama pull llama3.2:1b
python -m src.agent.ui
```

## Limitations

Final answers only (no method marks); inequalities and text answers are not supported; forms like `(1+√5)/2` abstain; the sandbox needs Linux (`resource` limits). Cambridge past-paper material is used for research only.
