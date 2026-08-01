"""Stage 6 - optional abstractive compression via a local model.

Every other stage in this pipeline is *extractive*: it decides what to keep, and
what it keeps is byte-identical to the input. This stage is the only one that
rewrites text, which makes it the only one that can invent something that was
never there. It is therefore built defensively, and it is optional.

Three guarantees:

**It cannot hang the demo.** Every call has a per-request timeout, the stage has
an overall wall-clock ceiling, and a chunk that times out keeps its original
text. If Ollama is not running, the stage reports ``skipped`` and the pipeline
continues unchanged.

**It cannot silently lose a fact.** Every paraphrase is checked against the
original for critical-token retention - numbers, identifiers, error codes,
named entities. Numbers are non-negotiable: a paraphrase that turns a 64
character limit into 128, or drops ``8000ms``, is discarded and the original
kept. This is the check that makes rewriting safe enough to ship.

**It cannot make things worse.** A paraphrase that is not actually shorter is
rejected, so the stage is monotonic: output tokens never exceed input tokens.

``fast_mode`` skips the stage entirely - a guaranteed-fast demo path.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .config import AbstractiveConfig, Config, get_config
from .entities import EntityScorer
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, StageMetrics, StageStatus

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You compress technical text. Rewrite the passage using fewer words while "
    "preserving every fact.\n"
    "RULES:\n"
    "1. Keep every number, identifier, function name, error code, version, "
    "path, and proper noun EXACTLY as written.\n"
    "2. Remove only filler: hedging, redundant restatement, verbose connectives.\n"
    "3. Do not summarise, editorialise, or add anything not in the passage.\n"
    "4. Preserve the original format (code stays code, log lines stay log lines).\n"
    "5. Output ONLY the rewritten passage. No preamble, no explanation, no "
    "markdown fences."
)

# Tokens that must survive a rewrite. Deliberately broader than spaCy's NER,
# because the facts that matter in code and logs - `PAYMENT_POOL_SIZE`,
# `ORD-427039`, `8000ms` - are not entities any prose model was trained on.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Order matters: `re` alternation is first-match-wins at each position, so the
# dotted form must come first or the snake_case branch eats `checkout_service`
# and never sees `checkout_service.handlers`.
_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z_]\w*(?:\.\w+)+"                      # dotted.path
    r"|[A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+)+"       # snake_case
    r"|[a-z]+[A-Z]\w*"                                  # camelCase
    r"|[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*)\b"             # CONSTANT_CASE
)


def critical_tokens(text: str) -> set[str]:
    """Facts a rewrite must not lose."""
    tokens = set(_IDENTIFIER.findall(text))
    tokens |= set(_NUMBER.findall(text))
    return tokens


def numbers_in(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


@dataclass
class Rejection:
    chunk_id: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "reason": self.reason, "detail": self.detail}


@dataclass
class AbstractiveResult:
    chunks: list[Chunk] = field(default_factory=list)
    metrics: StageMetrics = field(default_factory=lambda: StageMetrics("abstractive"))
    accepted: int = 0
    rejected: list[Rejection] = field(default_factory=list)
    tokens_saved: int = 0

    @property
    def rejection_rate(self) -> float:
        attempted = self.accepted + len(self.rejected)
        return len(self.rejected) / attempted if attempted else 0.0


class OllamaClient:
    """Minimal Ollama HTTP client that never raises into the pipeline."""

    def __init__(self, cfg: AbstractiveConfig) -> None:
        self.cfg = cfg
        self._available: bool | None = None
        self._error: str | None = None

    def available(self, refresh: bool = False) -> bool:
        if self._available is not None and not refresh:
            return self._available
        try:
            import requests

            response = requests.get(f"{self.cfg.host}/api/tags", timeout=2.0)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            wanted = self.cfg.model
            self._available = any(
                name == wanted or name.split(":")[0] == wanted.split(":")[0]
                for name in models
            )
            if not self._available:
                self._error = (
                    f"model {wanted!r} not pulled (available: {models or 'none'})"
                )
        except Exception as exc:
            self._available = False
            self._error = f"ollama unreachable at {self.cfg.host}: {exc}"
        return self._available

    @property
    def error(self) -> str | None:
        return self._error

    def generate(self, prompt: str, timeout: float) -> str | None:
        """Return the model's completion, or None on any failure."""
        try:
            import requests

            response = requests.post(
                f"{self.cfg.host}/api/generate",
                json={
                    "model": self.cfg.model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": self.cfg.temperature},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return (response.json().get("response") or "").strip()
        except Exception as exc:
            log.debug("ollama generate failed: %s", exc)
            return None

    def warmup(self) -> float:
        """Load the model into memory (~15 s cold) so the first real call is fast."""
        started = time.perf_counter()
        self.generate("ok", timeout=60.0)
        return (time.perf_counter() - started) * 1000


class AbstractiveCompressor:
    def __init__(
        self,
        cfg: Config | None = None,
        tokenizer: Tokenizer | None = None,
        client: OllamaClient | None = None,
        entity_scorer: EntityScorer | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.settings = self.cfg.abstractive
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)
        self.client = client or OllamaClient(self.settings)
        self.entity_scorer = entity_scorer or EntityScorer(self.cfg.density.spacy_model)

    def run(self, chunks: list[Chunk], fast_mode: bool = False) -> AbstractiveResult:
        started = time.perf_counter()
        tokens_in = sum(c.token_count for c in chunks)
        metrics = StageMetrics(
            name="abstractive",
            chunks_in=len(chunks),
            chunks_out=len(chunks),
            tokens_in=tokens_in,
            tokens_out=tokens_in,
        )
        result = AbstractiveResult(chunks=chunks, metrics=metrics)

        if fast_mode:
            return self._skip(result, "fast_mode requested; stage skipped")
        if not self.settings.enabled:
            return self._skip(result, "abstractive.enabled is false")
        if not chunks:
            return self._skip(result, "no chunks to compress")
        if not self.client.available():
            return self._skip(result, f"{self.client.error}; chunks left unmodified")

        protected = set(self.cfg.selection.protected_kinds)
        candidates = [
            c
            for c in chunks
            if c.kind not in protected
            and c.token_count >= self.settings.min_tokens_to_compress
        ]
        # Biggest first: the wall-clock ceiling may cut us off, so spend the
        # time we have on the chunks with the most to give back.
        candidates.sort(key=lambda c: -c.token_count)
        candidates = candidates[: self.settings.max_chunks]

        if not candidates:
            return self._skip(
                result,
                f"no chunk reached the {self.settings.min_tokens_to_compress} "
                f"token threshold",
            )

        deadline = started + self.settings.total_timeout_s
        saved = 0
        timed_out = 0

        for chunk in candidates:
            if time.perf_counter() >= deadline:
                timed_out += 1
                continue
            remaining = min(self.settings.timeout_s, deadline - time.perf_counter())
            if remaining <= 0.5:
                timed_out += 1
                continue

            outcome = self._compress_one(chunk, remaining)
            if isinstance(outcome, Rejection):
                result.rejected.append(outcome)
                continue
            saved += chunk.token_count - self.tokenizer.count(outcome)
            chunk.compressed_text = outcome
            result.accepted += 1

        tokens_out = sum(
            self.tokenizer.count(c.output_text) if c.compressed_text else c.token_count
            for c in chunks
        )
        result.tokens_saved = max(0, tokens_in - tokens_out)
        metrics.tokens_out = tokens_out
        metrics.duration_ms = (time.perf_counter() - started) * 1000
        metrics.status = StageStatus.OK
        if timed_out:
            metrics.note = (
                f"{timed_out} chunk(s) skipped: {self.settings.total_timeout_s}s "
                f"stage budget exhausted"
            )
        metrics.details = {
            "model": self.settings.model,
            "attempted": len(candidates),
            "accepted": result.accepted,
            "rejected": len(result.rejected),
            "rejection_rate": round(result.rejection_rate, 4),
            "rejections_by_reason": _count_reasons(result.rejected),
            "skipped_no_time": timed_out,
            "tokens_saved": result.tokens_saved,
            "eligible_chunks": len(candidates),
        }
        return result

    # -- one chunk ---------------------------------------------------------
    def _compress_one(self, chunk: Chunk, timeout: float) -> str | Rejection:
        completion = self.client.generate(chunk.text, timeout)
        if not completion:
            return Rejection(chunk.id, "no_response", "timeout or model error")

        candidate = _strip_fences(completion)
        if not candidate.strip():
            return Rejection(chunk.id, "empty_response")

        new_tokens = self.tokenizer.count(candidate)
        if new_tokens >= chunk.token_count:
            return Rejection(
                chunk.id,
                "no_gain",
                f"{chunk.token_count} -> {new_tokens} tokens",
            )

        verdict = self._safety_check(chunk.text, candidate)
        if verdict is not None:
            return Rejection(chunk.id, *verdict)
        return candidate

    def _safety_check(self, original: str, candidate: str) -> tuple[str, str] | None:
        """Reject a paraphrase that dropped facts. Returns (reason, detail)."""
        if self.settings.require_numbers_preserved:
            lost_numbers = numbers_in(original) - numbers_in(candidate)
            if lost_numbers:
                return (
                    "numbers_lost",
                    f"dropped {sorted(lost_numbers)[:5]}",
                )

        original_tokens = critical_tokens(original)
        if not original_tokens:
            return None
        retained = original_tokens & critical_tokens(candidate)
        overlap = len(retained) / len(original_tokens)
        if overlap < self.settings.entity_overlap_threshold:
            missing = sorted(original_tokens - retained)[:5]
            return (
                "entity_overlap",
                f"{overlap:.0%} retained (< {self.settings.entity_overlap_threshold:.0%}), "
                f"lost {missing}",
            )
        return None

    @staticmethod
    def _skip(result: AbstractiveResult, note: str) -> AbstractiveResult:
        result.metrics.status = StageStatus.SKIPPED
        result.metrics.note = note
        result.metrics.details = {"accepted": 0, "rejected": 0, "tokens_saved": 0}
        return result


_FENCE = re.compile(r"^\s*```[\w-]*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Small models wrap output in markdown fences despite being told not to."""
    match = _FENCE.match(text)
    return match.group(1) if match else text


def _count_reasons(rejections: list[Rejection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rejection in rejections:
        counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
    return counts
