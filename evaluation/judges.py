"""LLM judges: prompts, output parsing and model backends.

Every judge gets the same three inputs: the question, the mark-scheme answer
(with its notes) and the candidate answer. Each judge is sampled k = 5 times
(temperature 0.7, seeds 0..k-1); its confidence is the share of samples that
agree with the majority, so it can abstain below a threshold τ.

Protocol Change 3: five samples give only a few vote levels, so ties within a
level are broken by the mean token log-probability of the majority samples'
outputs: confidence = vote share + 0.01 · exp(mean log-prob). The bonus is below
0.01, so it never moves an answer across a vote level (levels are 0.2 apart).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

# CompassVerifier's official prompt (open-compass/CompassVerifier, src/prompts.py, CV_PROMPT),
# copied verbatim; Apache-2.0.
CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

# Prompt for the general LLM judges (fixed on the development set, then frozen).
EXAMINER_PROMPT = """You are a Cambridge IGCSE Mathematics (0580) examiner marking a student's final answer.
Use the mark scheme. Award the mark only if the final answer would earn the final-answer mark:
the value must be right, and any accuracy (significant figures, decimal places) or form
(simplified, expanded, factorised, exact, units) required by the question or the mark scheme must be met.

Question:
{question}

Mark scheme answer:
{gold_answer}
{notes}
Student's final answer:
{llm_response}

Reply with exactly one word: CORRECT or INCORRECT."""


def build_prompt(kind: str, question: str, gold: str, notes: str, answer: str) -> str:
    if kind == "compass":
        g = gold if not notes else f"{gold} ({notes})"
        return CV_PROMPT.format(question=question, gold_answer=g, llm_response=answer)
    return EXAMINER_PROMPT.format(question=question, gold_answer=gold,
                                  notes=f"Mark scheme notes: {notes}\n" if notes else "", llm_response=answer)


_WORD = re.compile(r"\b(INCORRECT|CORRECT|INVALID)\b")


def parse_vote(kind: str, text: str) -> str:
    """Map raw model output to correct | incorrect | invalid."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    up = t.upper()
    if kind == "compass":
        m = re.match(r"^[\s\"'*(\[]*([ABC])\b", up)
        if m:
            return {"A": "correct", "B": "incorrect", "C": "invalid"}[m.group(1)]
    m = _WORD.search(up)
    if m:
        return m.group(1).lower()
    return "invalid"


TIE_BREAK_WEIGHT = 0.01


def aggregate(votes: list[str], logprobs: list | None = None) -> tuple[str, float]:
    """Majority decision among correct/incorrect votes and its confidence: the share of all k samples,
    plus the within-level tie-break when per-sample mean log-probabilities are available."""
    c, i = votes.count("correct"), votes.count("incorrect")
    if c == i == 0:
        return "abstain", 0.0
    if c == i:
        return "abstain", c / len(votes)
    label = "correct" if c > i else "incorrect"
    conf = max(c, i) / len(votes)
    if logprobs:
        lp = [x for v, x in zip(votes, logprobs) if v == label and x is not None]
        if lp:
            import math
            conf += TIE_BREAK_WEIGHT * math.exp(sum(lp) / len(lp))
    return label, conf


def decide(votes: list[str], tau: float) -> str:
    label, conf = aggregate(votes)
    return label if label != "abstain" and conf >= tau else "abstain"


# ------------------------------------------------------------- backends
@dataclass
class OllamaBackend:
    model: str
    host: str = "http://127.0.0.1:11434"
    max_tokens: int = 16
    timeout: float = 300.0

    think: bool = False  # Qwen3 thinking mode: fixed on the development set (protocol §2)

    def generate(self, prompt: str, temperature: float, seed: int, system: str | None = None,
                 max_tokens: int | None = None) -> str:
        return self.generate_scored(prompt, temperature, seed, system, max_tokens)[0]

    def generate_scored(self, prompt, temperature, seed, system=None, max_tokens=None):
        """(text, mean token log-probability or None if this Ollama version does not return them)."""
        body = {"model": self.model, "prompt": prompt, "stream": False, "think": self.think, "logprobs": True,
                "options": {"temperature": temperature, "seed": seed,
                            "num_predict": max_tokens or self.max_tokens}}
        if system:
            body["system"] = system
        d = self._post("/api/generate", body)
        lps = [t.get("logprob") for t in (d.get("logprobs") or []) if isinstance(t, dict)]
        lps = [x for x in lps if isinstance(x, (int, float))]
        return d["response"], (sum(lps) / len(lps) if lps else None)

    def _post(self, path, body):
        req = urllib.request.Request(f"{self.host}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def info(self) -> dict:
        """Exact model digest, so laptop and Kaggle runs provably use the same weights (protocol §2)."""
        out = {"runtime": "ollama", "model": self.model, "think": self.think}
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=30) as r:
                for m in json.loads(r.read()).get("models", []):
                    if m.get("name") == self.model or m.get("model") == self.model:
                        out["digest"] = m.get("digest")
                        out["size_bytes"] = m.get("size")
                        out["details"] = m.get("details")
            with urllib.request.urlopen(f"{self.host}/api/version", timeout=30) as r:
                out["ollama_version"] = json.loads(r.read()).get("version")
        except Exception as e:
            out["digest_error"] = str(e)[:120]
        return out


@dataclass
class TransformersBackend:
    """Hugging Face transformers (e.g. opencompass/CompassVerifier-3B on a GPU)."""
    model: str
    max_tokens: int = 16
    _tok: object = None
    _lm: object = None

    def _load(self):
        if self._lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model)
            self._lm = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype="auto", device_map="auto")

    def generate(self, prompt: str, temperature: float, seed: int, system: str | None = None,
                 max_tokens: int | None = None) -> str:
        import torch
        self._load()
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ids = self._tok([text], return_tensors="pt").to(self._lm.device)
        torch.manual_seed(seed)
        out = self._lm.generate(**ids, max_new_tokens=max_tokens or self.max_tokens, do_sample=temperature > 0,
                                temperature=temperature if temperature > 0 else None,
                                output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[0][ids.input_ids.shape[1]:]
        self._last_lp = None
        try:
            sc = self._lm.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            self._last_lp = float(sc.float().mean()) if len(sc) else None
        except Exception:
            pass
        return self._tok.decode(gen, skip_special_tokens=True)

    def generate_scored(self, prompt, temperature, seed, system=None, max_tokens=None):
        text = self.generate(prompt, temperature, seed, system, max_tokens)
        return text, self._last_lp

    def info(self) -> dict:
        out = {"runtime": "transformers", "model": self.model}
        try:
            from huggingface_hub import model_info
            out["revision_sha"] = model_info(self.model).sha
            import transformers
            out["transformers_version"] = transformers.__version__
        except Exception as e:
            out["digest_error"] = str(e)[:120]
        return out


@dataclass
class MockBackend:
    """Deterministic stand-in used only to test the pipeline (never for results)."""
    model: str = "mock"
    accuracy: float = 0.8

    def generate(self, prompt: str, temperature: float, seed: int, system: str | None = None,
                 max_tokens: int | None = None) -> str:
        import hashlib
        import re as _re
        h = int(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest(), 16)
        m = _re.search(r"Final answer \(for your reference only; do NOT reveal it\): (.+)", prompt)
        if m:  # mock tutor: an instructed tutor that sometimes still gives the answer away
            return f"Nearly there! The answer is {m.group(1).strip()}." if h % 2 else "Start by writing down what you know."
        if "Student's message" in prompt:
            return "Think about which formula links the quantities in the question."
        truth = "CORRECT" if "[[gold=correct]]" in prompt else "INCORRECT"
        other = "INCORRECT" if truth == "CORRECT" else "CORRECT"
        return truth if (h % 1000) / 1000 < self.accuracy else other

    def generate_scored(self, prompt, temperature, seed, system=None, max_tokens=None):
        import hashlib
        h = int(hashlib.sha256(f"lp|{prompt}|{seed}".encode()).hexdigest(), 16)
        return self.generate(prompt, temperature, seed, system, max_tokens), -((h % 1000) / 1000) * 2

    def info(self) -> dict:
        return {"runtime": "mock", "model": "mock", "digest": "mock"}


def make_backend(spec: str):
    """'ollama:qwen3:4b', 'hf:opencompass/CompassVerifier-3B' or 'mock'."""
    kind, _, name = spec.partition(":")
    if kind == "ollama":
        return OllamaBackend(name)
    if kind == "hf":
        return TransformersBackend(name)
    if kind == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend {spec!r}")


def sample(backend, prompt: str, kind: str, k: int = 5, temperature: float = 0.7, base_seed: int = 0):
    """k samples with seeds base_seed..base_seed+k-1: votes, raw outputs, mean ms per sample, log-probs."""
    votes, raw, lps, t0 = [], [], [], time.perf_counter()
    for i in range(k):
        txt, lp = backend.generate_scored(prompt, temperature, base_seed + i)
        raw.append(txt)
        lps.append(lp)
        votes.append(parse_vote(kind, txt))
    return votes, raw, (time.perf_counter() - t0) * 1000 / k, lps
