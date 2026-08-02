"""The two provider contracts every model call in this project goes through.

There are exactly two things the engine asks a model to do: turn text into
vectors (stage 3) and turn a prompt into text (stage 6 and the eval harness).
So there are exactly two interfaces here, and one implementation set behind
each. The eval harness deliberately shares :class:`GenerationProvider` with the
pipeline rather than owning a parallel stack - the whole point of the migration
was to stop having two places that know how to call a model.

Design notes worth knowing before adding a provider:

**Batching lives in the base class.** ``embed`` is a template method: it slices
the input to the provider's real batch limit, retries a failed batch once, and
records call count and latency. A subclass implements ``_embed_batch`` for one
already-correctly-sized batch and nothing else. Sending one HTTP request per
chunk would be the single easiest way to burn a free tier.

**Failures are typed, not booleans.** A missing key is not the same as a 429 is
not the same as a 500. :class:`ProviderUnavailable` means "skip me, this is not
my turn"; :class:`ProviderFailed` means "I tried and could not". The chain in
:mod:`engine.providers.chain` treats them differently in what it reports.

**No error leaves here unredacted.** Every exception string is passed through
:func:`engine.providers.keys.redact` on the way out.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from . import keys

log = logging.getLogger(__name__)

#: Providers only ever return text this long by accident; used to bound a
#: pathological response before it reaches the tokenizer.
MAX_RESPONSE_CHARS = 200_000


class ProviderError(RuntimeError):
    """Base for anything a provider can go wrong with. Always redacted."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.reason = keys.redact(str(message))
        super().__init__(f"{provider}: {self.reason}")


class ProviderUnavailable(ProviderError):
    """Not configured, or not reachable. Skipping it is normal, not an error."""


class ProviderFailed(ProviderError):
    """Configured and tried, but the call did not succeed (timeout, 429, 5xx).

    ``kind`` separates the two failures that deserve different treatment from a
    chain. A ``"timeout"`` is usually a statement about *this request* - a long
    prompt, or a caller that imposed a deliberately tight deadline. Anything
    else (429, 401, 5xx, a malformed body) is a statement about the provider.
    Only the latter earns a cooldown; see :mod:`engine.providers.chain`.
    """

    def __init__(self, provider: str, message: str, *, kind: str = "other") -> None:
        self.kind = kind
        super().__init__(provider, message)

    @property
    def retryable(self) -> bool:
        """Whether trying the *same* provider again could plausibly work.

        A 429 quota, a 401 bad key and a 404 retired model are settled facts:
        repeating the call 0.4 s later burns a round trip to learn what we
        already know, and there is a whole other provider in the chain waiting
        to be asked. Only transient shapes - a 5xx, a dropped connection, a
        timeout - earn a second attempt.

        Measured: with two exhausted providers ahead of a working one, dropping
        these retries cut a cold compression from ~3.1 s to ~1.6 s.
        """
        settled = ("429", "401", "403", "404", "quota", "invalid api key",
                   "credit", "billing", "not found")
        return not any(marker in self.reason.lower() for marker in settled)


class NoProviderAvailable(RuntimeError):
    """Every provider in a configured chain was skipped or failed.

    Carries the per-provider reasons so the message a user sees says *why*
    nothing ran, rather than just that nothing did.
    """

    def __init__(self, role: str, attempts: list["Attempt"]) -> None:
        self.role = role
        self.attempts = attempts
        if attempts:
            detail = "; ".join(f"{a.provider} ({a.reason})" for a in attempts)
        else:
            detail = "no providers configured in the chain"
        super().__init__(
            f"no {role} provider available: {detail}. "
            f"Configure a key in .env (see .env.example) or add a reachable "
            f"local provider to the chain in config.yaml."
        )


@dataclass
class Attempt:
    """One provider's outcome inside a chain, for telemetry and error text."""

    provider: str
    ok: bool
    reason: str = ""
    duration_ms: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "ok": self.ok,
            "skipped": self.skipped,
            "reason": keys.redact(self.reason),
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class CallStats:
    """Counters a cloud call makes material that a local one did not.

    Embedding used to be a local matrix multiply whose only cost was wall-clock
    on this machine. Over an API it is N HTTP requests against a rate limit, so
    the count is now a first-class metric alongside the latency.
    """

    calls: int = 0
    latency_ms: float = 0.0
    items: int = 0
    retries: int = 0

    def record(self, duration_ms: float, items: int = 0) -> None:
        self.calls += 1
        self.latency_ms += duration_ms
        self.items += items

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "latency_ms": round(self.latency_ms, 2),
            "items": self.items,
            "retries": self.retries,
        }


class Provider(ABC):
    """Shared identity and configuration checks."""

    #: Stable name used in config.yaml, /health and every report.
    name: str = "provider"
    #: Environment variable holding this provider's key. None for local ones.
    env_key: str | None = None
    #: Model identifier, surfaced so a report can say what actually answered.
    model: str = ""

    def configured(self) -> tuple[bool, str]:
        """(usable, reason). A False here means "skip", never "fail"."""
        if self.env_key and not keys.has(self.env_key):
            return False, f"{self.env_key} is not configured"
        return True, ""

    def describe(self) -> dict:
        ok, reason = self.configured()
        return {
            "provider": self.name,
            "model": self.model,
            "configured": ok,
            "requires_key": self.env_key,
            "reason": reason or None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}:{self.model}>"


class EmbeddingProvider(Provider):
    """``embed(texts) -> list[list[float]]``, batched to the provider's limit.

    Vectors are returned as the provider produced them. Normalisation is the
    caller's job (:class:`engine.embeddings.EmbeddingModel` does it), because
    stage 3's clustering treats cosine as a dot product and must be able to
    guarantee that invariant regardless of which vendor answered.
    """

    #: Real per-request input limit. Enforced here so a caller never has to.
    batch_limit: int = 64
    #: Declared output dimension where the provider fixes one; 0 = discover.
    dimension: int = 0

    def __init__(self) -> None:
        self.stats = CallStats()

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch already sized within ``batch_limit``."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed every text, in input order, in as few requests as allowed."""
        if not texts:
            return []
        ok, reason = self.configured()
        if not ok:
            raise ProviderUnavailable(self.name, reason)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_limit):
            batch = texts[start : start + self.batch_limit]
            vectors.extend(self._embed_batch_with_retry(batch))
        if len(vectors) != len(texts):
            raise ProviderFailed(
                self.name,
                f"expected {len(texts)} vectors, received {len(vectors)}",
            )
        return vectors

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """One retry, because a free tier's 429 is usually over in a moment.

        Deliberately *one*. An exhausted quota does not recover inside a demo,
        and the chain below has a whole other provider to move on to - spending
        the user's wall-clock on a third attempt helps nobody.
        """
        last: Exception | None = None
        for attempt in range(2):
            started = time.perf_counter()
            try:
                result = self._embed_batch(batch)
                self.stats.record((time.perf_counter() - started) * 1000, len(batch))
                return result
            except ProviderError as exc:
                self.stats.record((time.perf_counter() - started) * 1000)
                last = exc
                # A settled failure (quota, bad key, retired model) will not
                # resolve in 0.4 s. Fall through to the next provider now
                # instead of paying a second round trip to confirm it.
                if not getattr(exc, "retryable", True):
                    break
            except Exception as exc:  # noqa: BLE001 - normalised below
                self.stats.record((time.perf_counter() - started) * 1000)
                last = ProviderFailed(self.name, exc)
            if attempt == 0:
                self.stats.retries += 1
                time.sleep(0.4)
        raise last if isinstance(last, ProviderError) else ProviderFailed(self.name, last)


class GenerationProvider(Provider):
    """``generate(prompt, max_tokens, timeout_s) -> str``.

    One interface, two callers: stage 6 asks it to paraphrase a chunk, and the
    eval harness asks it to answer a question about a context. They differ only
    in ``system`` and in the prompt - which is exactly why they should not have
    two separate provider stacks.
    """

    #: Default per-call ceiling; a caller may always pass a tighter one.
    default_timeout_s: float = 30.0

    def __init__(self) -> None:
        self.stats = CallStats()

    @abstractmethod
    def _generate(
        self, prompt: str, max_tokens: int, timeout_s: float, system: str | None
    ) -> str:
        """Issue one completion request. Raise ProviderFailed on any problem."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        timeout_s: float | None = None,
        system: str | None = None,
    ) -> str:
        """Return a completion, or raise. Never returns None, never blocks past
        ``timeout_s``."""
        ok, reason = self.configured()
        if not ok:
            raise ProviderUnavailable(self.name, reason)
        budget = float(timeout_s if timeout_s is not None else self.default_timeout_s)
        started = time.perf_counter()
        try:
            text = self._generate(prompt, max_tokens, budget, system)
        except ProviderError:
            self.stats.record((time.perf_counter() - started) * 1000)
            raise
        except Exception as exc:  # noqa: BLE001 - normalised for the chain
            self.stats.record((time.perf_counter() - started) * 1000)
            raise ProviderFailed(self.name, exc) from None
        self.stats.record((time.perf_counter() - started) * 1000, 1)
        return (text or "")[:MAX_RESPONSE_CHARS]
