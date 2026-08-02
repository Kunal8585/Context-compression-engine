"""Ordered fallback chains - the part that makes cloud providers safe to depend on.

A single hosted provider is a single point of failure: free tiers rate-limit
without warning, models get retired, and conference wifi drops. A chain turns
that into a degradation instead of an outage. The rules are deliberately small
enough to explain in one breath:

1. Walk the configured providers in order.
2. A provider with no key configured is **skipped**, not failed. Not having a
   Cohere key is a normal state, not an error.
3. A provider that is tried and fails (timeout, 429, 5xx, bad response) is
   recorded, put in a short cooldown, and the chain moves to the next one.
4. If every provider is skipped or fails, raise :class:`NoProviderAvailable`
   with the per-provider reason. That is the one hard error this layer can
   produce, and it says exactly what to configure.

**Why a cooldown rather than a permanent demotion.** The chain object outlives
one request - the API server holds a pipeline for its whole lifetime. Demoting
a rate-limited provider forever means one bad minute at 10am silently costs you
your fastest provider for the rest of the day. Retrying it on every single call
is the opposite mistake: a 12-chunk stage 6 pays the timeout twelve times over.
A 60-second cooldown is the cheap middle: at most one wasted attempt per minute,
and the provider comes back on its own.
"""

from __future__ import annotations

import logging
import time

from .base import (
    Attempt,
    EmbeddingProvider,
    GenerationProvider,
    NoProviderAvailable,
    ProviderError,
    ProviderUnavailable,
    Provider,
)

log = logging.getLogger(__name__)

#: How long a failed provider sits out before the chain tries it again.
DEFAULT_COOLDOWN_S = 60.0


class _Chain:
    """Shared ordering, cooldown and telemetry for both provider roles."""

    role = "provider"

    def __init__(
        self, providers: list[Provider], cooldown_s: float = DEFAULT_COOLDOWN_S
    ) -> None:
        self.providers = providers
        self.cooldown_s = cooldown_s
        self._cooldown_until: dict[str, float] = {}
        #: Attempts from the most recent call, for stage metrics and /health.
        self.last_attempts: list[Attempt] = []
        #: Which provider actually answered last. None until something has.
        self.last_provider: str | None = None

    # -- introspection -----------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [p.name for p in self.providers]

    def describe(self) -> dict:
        return {
            "role": self.role,
            "chain": self.names,
            "providers": [p.describe() for p in self.providers],
            "active": self.active_provider_name(),
            "last_used": self.last_provider,
        }

    def active_provider_name(self) -> str | None:
        """First provider that would be tried right now, or None if the chain
        has nothing usable. Cheap enough for /health; makes no model calls."""
        for provider in self.providers:
            ok, _ = provider.configured()
            if ok and not self._cooling(provider.name):
                return provider.name
        return None

    def available(self) -> tuple[bool, str]:
        """(usable, reason) for the chain as a whole."""
        reasons = []
        for provider in self.providers:
            ok, reason = provider.configured()
            if ok and not self._cooling(provider.name):
                return True, ""
            reasons.append(f"{provider.name} ({reason or 'cooling down'})")
        if not self.providers:
            return False, f"no {self.role} providers configured"
        return False, "; ".join(reasons)

    # -- cooldown ----------------------------------------------------------
    def _cooling(self, name: str) -> bool:
        until = self._cooldown_until.get(name)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._cooldown_until[name]
            return False
        return True

    def _penalise(self, name: str) -> None:
        self._cooldown_until[name] = time.monotonic() + self.cooldown_s

    # -- the walk ----------------------------------------------------------
    def _run(self, call):
        """Try each provider in turn; return the first success.

        ``call`` takes a provider and returns its result, raising ProviderError
        on failure. Everything about ordering, skipping and reporting is here so
        that neither role has to reimplement it.
        """
        attempts: list[Attempt] = []
        self.last_attempts = attempts

        for provider in self.providers:
            ok, reason = provider.configured()
            if not ok:
                attempts.append(
                    Attempt(provider.name, ok=False, reason=reason, skipped=True)
                )
                continue
            if self._cooling(provider.name):
                remaining = self._cooldown_until[provider.name] - time.monotonic()
                attempts.append(
                    Attempt(
                        provider.name,
                        ok=False,
                        reason=f"in cooldown for another {remaining:.0f}s",
                        skipped=True,
                    )
                )
                continue

            started = time.perf_counter()
            try:
                result = call(provider)
            except ProviderUnavailable as exc:
                # "Not my turn" - no cooldown, it was never really tried.
                attempts.append(
                    Attempt(
                        provider.name,
                        ok=False,
                        reason=exc.reason,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        skipped=True,
                    )
                )
                continue
            except ProviderError as exc:
                elapsed = (time.perf_counter() - started) * 1000
                attempts.append(
                    Attempt(provider.name, ok=False, reason=exc.reason, duration_ms=elapsed)
                )
                # A timeout does not earn a cooldown. Callers impose deadlines
                # far tighter than any provider's default - stage 6 shrinks its
                # per-call budget as its 12 s stage ceiling runs down, and once
                # that budget is under a couple of seconds *every* provider
                # "fails". Cooling one off for that would let stage 6's own
                # impatience mark a perfectly healthy local model as unhealthy
                # for the next minute of unrelated requests. Callers that
                # genuinely cannot afford repeated timeouts bound themselves
                # with a wall-clock ceiling instead, which stage 6 does.
                if getattr(exc, "kind", "other") != "timeout":
                    self._penalise(provider.name)
                log.warning(
                    "%s provider %s failed (%s); falling back",
                    self.role,
                    provider.name,
                    exc.reason,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - a bug in a provider is
                # still not allowed to take the chain down.
                elapsed = (time.perf_counter() - started) * 1000
                attempts.append(
                    Attempt(provider.name, ok=False, reason=repr(exc), duration_ms=elapsed)
                )
                self._penalise(provider.name)
                log.exception("provider %s raised unexpectedly", provider.name)
                continue

            attempts.append(
                Attempt(
                    provider.name,
                    ok=True,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
            self.last_provider = provider.name
            return result

        raise NoProviderAvailable(self.role, attempts)

    def telemetry(self) -> dict:
        """What the last call did, for a stage's ``details`` block."""
        return {
            "chain": self.names,
            "provider_used": self.last_provider,
            "attempts": [a.to_dict() for a in self.last_attempts],
            "fell_back": bool(
                self.last_provider and self.last_attempts
                and self.last_attempts[0].provider != self.last_provider
            ),
        }


class EmbeddingChain(_Chain):
    """Fallback chain over :class:`EmbeddingProvider`.

    One caveat that does not apply to generation: **vectors from different
    providers are not comparable.** Dimensions differ (1536 / 768 / 1024 / 384)
    and so do the spaces. Falling back mid-document would produce a matrix that
    silently mixes two geometries, so the chain resolves one provider for the
    whole ``embed`` call and re-walks only on the next call.
    """

    role = "embedding"

    def __init__(self, providers: list[EmbeddingProvider], **kwargs) -> None:
        super().__init__(providers, **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._run(lambda provider: provider.embed(texts))

    def stats(self) -> dict:
        """Per-provider call counts and latency, summed across the run."""
        return {
            provider.name: provider.stats.to_dict()
            for provider in self.providers
            if provider.stats.calls
        }


class GenerationChain(_Chain):
    """Fallback chain over :class:`GenerationProvider`.

    Serves both callers of a text model: stage 6's paraphrase and the eval
    harness's answer. They pass different prompts and a different ``system``;
    everything else - ordering, timeouts, fallback, redaction - is identical and
    lives here.
    """

    role = "generation"

    def __init__(self, providers: list[GenerationProvider], **kwargs) -> None:
        super().__init__(providers, **kwargs)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        timeout_s: float | None = None,
        system: str | None = None,
    ) -> str:
        return self._run(
            lambda provider: provider.generate(prompt, max_tokens, timeout_s, system)
        )

    def try_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        timeout_s: float | None = None,
        system: str | None = None,
    ) -> str | None:
        """``generate`` for callers that must not raise - stage 6's contract.

        Stage 6 is optional by design: a chunk it cannot paraphrase keeps its
        original text. Turning an exhausted chain into None here keeps that
        guarantee at the one call site that needs it, without weakening the
        harness's requirement that an exhausted chain be a loud, clear error.
        """
        try:
            return self.generate(prompt, max_tokens, timeout_s, system)
        except NoProviderAvailable as exc:
            log.info("generation unavailable: %s", exc)
            return None

    def stats(self) -> dict:
        return {
            provider.name: provider.stats.to_dict()
            for provider in self.providers
            if provider.stats.calls
        }
