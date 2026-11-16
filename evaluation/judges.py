"""LLM judges: prompts, output parsing and model backends.

Every judge gets the same three inputs: the question, the mark-scheme answer
(with its notes) and the candidate answer. Each judge is sampled k = 5 times
(temperature 0.7, fixed seeds); its confidence is the share of samples that
agree with the majority, so it can abstain below a threshold τ.
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


def aggregate(votes: list[str]) -> tuple[str, float]:
    """Majority decision among correct/incorrect votes, and its share of all k samples."""
    c, i = votes.count("correct"), votes.count("incorrect")
    if c == i == 0:
        return "abstain", 0.0
    if c == i:
        return "abstain", c / len(votes)
    return ("correct", c / len(votes)) if c > i else ("incorrect", i / len(votes))


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

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False, "think": False,
                "options": {"temperature": temperature, "seed": seed, "num_predict": self.max_tokens}}
        req = urllib.request.Request(f"{self.host}/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["response"]


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

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        import torch
        self._load()
        text = self._tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ids = self._tok([text], return_tensors="pt").to(self._lm.device)
        torch.manual_seed(seed)
        out = self._lm.generate(**ids, max_new_tokens=self.max_tokens, do_sample=temperature > 0,
                                temperature=temperature if temperature > 0 else None)
        return self._tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


@dataclass
class MockBackend:
    """Deterministic stand-in used only to test the pipeline (never for results)."""
    model: str = "mock"
    accuracy: float = 0.8

    def generate(self, prompt: str, temperature: float, seed: int) -> str:
        import hashlib
        h = int(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest(), 16)
        truth = "CORRECT" if "[[gold=correct]]" in prompt else "INCORRECT"
        other = "INCORRECT" if truth == "CORRECT" else "CORRECT"
        return truth if (h % 1000) / 1000 < self.accuracy else other


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
    votes, raw, t0 = [], [], time.perf_counter()
    for i in range(k):
        txt = backend.generate(prompt, temperature, base_seed + i)
        raw.append(txt)
        votes.append(parse_vote(kind, txt))
    return votes, raw, (time.perf_counter() - t0) * 1000 / k
