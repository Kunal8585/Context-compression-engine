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
import re
from typing import TYPE_CHECKING

from engine.config import EvaluationConfig
from engine.providers import keys
from engine.providers.generation import OllamaProvider, OpenAIGenerationProvider

from .testset import TestItem, fact_present

if TYPE_CHECKING:  # pragma: no cover
    from engine.config import ProvidersConfig

log = logging.getLogger(__name__)

DEFAULT_LOCAL_HOST = "http://localhost:11434"

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
    """LLM-as-judge with OpenAI preferred, local fallback, or disabled.

    Goes through :mod:`engine.providers` like every other model call in the
    project - the judge used to reach for the OpenAI SDK and an Ollama URL
    directly, which was exactly the kind of second implementation the provider
    migration exists to remove.

    It stays deliberately *unchained*: a judge that silently fell back to the
    same small model that produced the answers would be self-grading, which is
    the failure this class's docstring warns about. If the requested judge is
    unavailable there is no judge, and the report says so.
    """

    #: Consecutive failures after which the judge gives up for the rest of the
    #: run. A judge is a second opinion, not the score, so an unreachable one
    #: must not cost a failed round-trip on all 30 calls of a benchmark - which
    #: is exactly what an expired key used to do.
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self, cfg: EvaluationConfig, providers_cfg: "ProvidersConfig | None" = None
    ) -> None:
        self.cfg = cfg
        self.providers_cfg = providers_cfg
        self.provider = self._resolve()
        self._failures = 0
        self._disabled_reason: str | None = None

    def _resolve(self) -> str:
        requested = self.cfg.judge_provider
        if requested == "exact":
            return "none"
        has_key = keys.has("OPENAI_API_KEY")
        if requested == "openai":
            return "openai" if has_key else "none"
        if requested == "ollama":
            return "ollama"
        # auto
        return "openai" if has_key else "none"

    @property
    def describe(self) -> str:
        if self._disabled_reason:
            return f"none ({self._disabled_reason})"
        if self.provider == "openai":
            return f"openai:{self.cfg.judge_model_openai}"
        if self.provider == "ollama":
            return f"ollama:{self.cfg.judge_model_local}"
        return "none (key-fact recall only)"

    def score(self, item: TestItem, answer: str) -> bool | None:
        """True/False, or None when no judge is available."""
        if self.provider == "none" or self._disabled_reason or not answer.strip():
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
            self._failures += 1
            if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._disabled_reason = (
                    f"{self.provider} judge failed "
                    f"{self._failures} times in a row; disabled for this run"
                )
                log.warning("%s", self._disabled_reason)
            return None
        self._failures = 0
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        try:
            return bool(json.loads(match.group())["correct"])
        except Exception:
            return None

    def _local_host(self) -> str:
        return self.providers_cfg.local_host if self.providers_cfg else DEFAULT_LOCAL_HOST

    def _ask_openai(self, prompt: str) -> str | None:
        try:
            provider = OpenAIGenerationProvider(self.cfg.judge_model_openai, timeout_s=60)
            return provider.generate(prompt, max_tokens=60, timeout_s=60)
        except Exception as exc:
            # Redacted by the provider layer before it gets here.
            log.warning("openai judge failed: %s", exc)
            return None

    def _ask_ollama(self, prompt: str) -> str | None:
        try:
            provider = OllamaProvider(
                self.cfg.judge_model_local, self._local_host(), timeout_s=60
            )
            return provider.generate(prompt, max_tokens=60, timeout_s=60)
        except Exception as exc:
            log.warning("local judge failed: %s", exc)
            return None
