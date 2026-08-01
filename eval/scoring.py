"""Answer scoring.

The headline accuracy metric is **deterministic key-fact recall**, not an LLM
judge. Two reasons:

1. *Reproducibility.* A judge introduces variance into the one number the whole
   project is measured on. Key-fact recall gives the same answer every run, and
   a judge can read the rule in ten seconds.
2. *Self-grading.* Without an OpenAI key the only available judge is the same
   3B model that produced the answers. A model grading its own output is not
   evidence.

The LLM judge still runs when one is configured, and is reported alongside - it
catches correct answers phrased in a way the literal check misses. But it is a
second opinion, not the score.
"""

from __future__ import annotations

import json
import logging
import os
import re

from engine.config import EvaluationConfig

from .testset import TestItem, fact_present

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading a question-answering system.

QUESTION: {question}
REFERENCE ANSWER: {expected}
CANDIDATE ANSWER: {candidate}

Does the candidate state the same facts as the reference? Ignore wording,
formatting and extra detail. It is correct only if every fact in the reference
is present and nothing contradicts it.

Reply with JSON only: {{"correct": true|false, "reason": "<10 words"}}"""


def key_fact_recall(answer: str, item: TestItem) -> tuple[float, list[str]]:
    """Fraction of the item's key facts stated in `answer`, plus what is missing."""
    if not item.key_facts:
        return 0.0, []
    missing = [
        fact if isinstance(fact, str) else "/".join(fact)
        for fact in item.key_facts
        if not fact_present(fact, answer)
    ]
    return 1.0 - len(missing) / len(item.key_facts), missing


class Judge:
    """LLM-as-judge with OpenAI preferred, local fallback, or disabled."""

    def __init__(self, cfg: EvaluationConfig) -> None:
        self.cfg = cfg
        self.provider = self._resolve()

    def _resolve(self) -> str:
        requested = self.cfg.judge_provider
        if requested == "exact":
            return "none"
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        if requested == "openai":
            return "openai" if has_key else "none"
        if requested == "ollama":
            return "ollama"
        # auto
        return "openai" if has_key else "none"

    @property
    def describe(self) -> str:
        if self.provider == "openai":
            return f"openai:{self.cfg.judge_model_openai}"
        if self.provider == "ollama":
            return f"ollama:{self.cfg.judge_model_local}"
        return "none (key-fact recall only)"

    def score(self, item: TestItem, answer: str) -> bool | None:
        """True/False, or None when no judge is available."""
        if self.provider == "none" or not answer.strip():
            return None
        prompt = JUDGE_PROMPT.format(
            question=item.question,
            expected=item.expected_answer,
            candidate=answer[:2000],
        )
        raw = (
            self._ask_openai(prompt)
            if self.provider == "openai"
            else self._ask_ollama(prompt)
        )
        if raw is None:
            return None
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        try:
            return bool(json.loads(match.group())["correct"])
        except Exception:
            return None

    def _ask_openai(self, prompt: str) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.cfg.judge_model_openai,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=60,
            )
            return response.choices[0].message.content
        except Exception as exc:
            log.warning("openai judge failed: %s", exc)
            return None

    def _ask_ollama(self, prompt: str) -> str | None:
        try:
            import requests

            response = requests.post(
                f"{self.cfg.host}/api/generate",
                json={
                    "model": self.cfg.judge_model_local,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 60},
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response")
        except Exception as exc:
            log.warning("local judge failed: %s", exc)
            return None
