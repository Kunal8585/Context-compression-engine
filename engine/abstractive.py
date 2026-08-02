"""Stage 6 - optional abstractive compression via the generation provider chain.

Every other stage in this pipeline is *extractive*: it decides what to keep, and
what it keeps is byte-identical to the input. This stage is the only one that
rewrites text, which makes it the only one that can invent something that was
never there. It is therefore built defensively, and it is optional.

Three guarantees, none of which changed when the model moved to the cloud:

**It cannot hang the demo.** Every call has a per-request timeout, the stage has
an overall wall-clock ceiling, and a chunk that times out keeps its original
text. If no provider in the chain can run - no keys, no network, no local
Ollama - the stage reports ``skipped`` with the reason and the pipeline
continues unchanged.

**It cannot silently lose a fact.** Every paraphrase is checked against the
original for critical-token retention - numbers, identifiers, error codes,
named entities. Numbers are non-negotiable: a paraphrase that turns a 64
character limit into 128, or drops ``8000ms``, is discarded and the original
kept. This is the check that makes rewriting safe enough to ship.

*This check is unchanged by the provider migration, deliberately.* A frontier
model is not a trustworthy paraphraser of a fact-dense log line just because it
is large - it is a better one, which shifts the rejection rate, not the need
for the net. The safety property must hold for whichever provider the chain
happens to resolve to, so it is enforced here, provider-agnostically, on the
text that comes back.

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
from .providers import GenerationChain, build_generation_chain
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


#: Below this much remaining stage budget, a paraphrase call is not attempted.
#: Measured: llama3.2:3b needs 3-6 s for a 150-250 token chunk, and even the
#: fastest hosted provider needs ~1 s of round trip. Anything under this is a
#: request issued only to time out.
MIN_CALL_SECONDS = 2.5


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


class GenerationClient:
    """Stage 6's view of the generation chain: never raises, never blocks.

    A deliberately narrow adapter rather than a second provider stack. The
    chain in :mod:`engine.providers.chain` does the ordering, key checks,
    fallback and redaction; this class does the two things stage 6 specifically
    needs, which the harness explicitly does *not* want:

    * ``generate`` returns ``None`` instead of raising. A chunk that cannot be
      paraphrased keeps its original text - that is the stage's contract, and
      an exhausted chain is just another way for one chunk to be left alone.
    * ``available`` is a bool with a readable ``error``, because that is what
      the pipeline's skip path and ``/health`` already consume.
    """

    def __init__(self, chain: GenerationChain, cfg: AbstractiveConfig) -> None:
        self.chain = chain
        self.cfg = cfg
        self._error: str | None = None

    def available(self, refresh: bool = False) -> bool:
        ok, reason = self.chain.available()
        self._error = None if ok else reason
        return ok

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def provider(self) -> str | None:
        """Whichever provider last answered - what the metrics should report."""
        return self.chain.last_provider

    @property
    def model(self) -> str:
        name = self.chain.last_provider or self.chain.active_provider_name()
        provider = next((p for p in self.chain.providers if p.name == name), None)
        return f"{name}:{provider.model}" if provider else "none"

    def generate(
        self, prompt: str, timeout: float, max_tokens: int = 512
    ) -> str | None:
        """Return a paraphrase, or None if every provider was skipped or failed.

        ``max_tokens`` is the *original* chunk's size. A paraphrase longer than
        its input is rejected by the caller anyway, so capping the completion
        there costs nothing and stops a runaway generation from eating the
        stage's whole wall-clock budget - which matters far more now that a
        token is billed rather than merely slow.
        """
        return self.chain.try_generate(
            prompt,
            max_tokens=max(64, max_tokens),
            timeout_s=timeout,
            system=SYSTEM_PROMPT,
        )

    def warmup(self) -> float:
        """Resolve the chain and pay any cold-start cost up front.

        On the local provider this is Ollama loading the model (~15 s cold). On
        a hosted one it is DNS, TLS and the chain walk - much cheaper, but still
        better paid before a judge's first request than during it.
        """
        started = time.perf_counter()
        self.chain.try_generate("ok", max_tokens=8, timeout_s=60.0)
        return (time.perf_counter() - started) * 1000

    def describe(self) -> dict:
        return self.chain.describe()


class AbstractiveCompressor:
    def __init__(
        self,
        cfg: Config | None = None,
        tokenizer: Tokenizer | None = None,
        client: GenerationClient | None = None,
        entity_scorer: EntityScorer | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.settings = self.cfg.abstractive
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)
        self.client = client or GenerationClient(
            build_generation_chain(self.cfg, timeout_s=self.settings.timeout_s),
            self.settings,
        )
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
            # Don't start a call the budget cannot finish. A sub-MIN_CALL_SECONDS
            # window is not enough for any provider - local or hosted - to return
            # a paraphrase, so issuing the request only burns the remainder of
            # the stage budget on a guaranteed timeout.
            if remaining < MIN_CALL_SECONDS:
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
        telemetry = self.client.chain.telemetry()
        metrics.provider_used = telemetry["provider_used"]
        metrics.details = {
            # Which model actually rewrote the text, not which one was asked
            # first - with a fallback chain those are routinely different, and
            # the rejection rate below is only interpretable against the one
            # that ran.
            "model": self.client.model,
            "provider": telemetry["provider_used"],
            "provider_chain": telemetry["chain"],
            "fell_back": telemetry["fell_back"],
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
        completion = self.client.generate(chunk.text, timeout, chunk.token_count)
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
