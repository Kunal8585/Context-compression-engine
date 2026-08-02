"""Stage 3's embedding backend: a thin adapter over the provider chain.

This module used to *be* the embedding implementation - a direct
sentence-transformers wrapper. It is now the seam between the pipeline and
:mod:`engine.providers`: the MiniLM code moved to
:class:`~engine.providers.embedding.LocalEmbeddingProvider` and sits alongside
OpenAI, Gemini and Cohere as one option among four.

The interface stage 3 sees is unchanged on purpose - ``encode`` still returns
an L2-normalised float32 matrix or ``None``, and ``available``/``load_error``
still mean what they meant. The clustering in :mod:`engine.redundancy` did not
have to change a line, which is the whole point of doing this as a swap behind
an interface rather than a rewrite.

Two guarantees this layer owns:

**It never raises into the pipeline.** An exhausted chain - no key, no network,
no local model - returns ``None`` and the stage degrades to exact-hash dedup,
exactly as it did when the only failure mode was "MiniLM did not load".

**Vectors are always L2-normalised.** Stage 3 treats cosine similarity as a dot
product. OpenAI already returns unit vectors, Gemini and Cohere do not
consistently, and no vendor promises to keep doing whatever they do today. So
normalisation happens here, once, no matter who answered.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config, RedundancyConfig, get_config
from .providers import (
    EmbeddingChain,
    NoProviderAvailable,
    build_embedding_chain,
    resolve_device,
)

log = logging.getLogger(__name__)

__all__ = ["EmbeddingModel", "EncodeStats", "resolve_device"]


@dataclass
class EncodeStats:
    """Per-run embedding telemetry.

    ``api_calls`` and ``provider_latency_ms`` are new since the migration and
    are not decoration: embedding used to be a local matrix multiply whose only
    cost was wall-clock on this machine, and is now N HTTP requests against a
    rate limit. The call count is the number that tells you whether batching is
    actually working, and it is the one that decides whether a free tier
    survives a demo.
    """

    count: int = 0
    duration_ms: float = 0.0
    device: str = "cpu"
    model: str = ""
    dimension: int = 0
    provider: str | None = None
    api_calls: int = 0
    provider_latency_ms: float = 0.0
    retries: int = 0
    fell_back: bool = False
    chain: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "encoded": self.count,
            "encode_ms": round(self.duration_ms, 2),
            "device": self.device,
            "model": self.model,
            "dimension": self.dimension,
            "provider": self.provider,
            "chain": self.chain,
            "api_calls": self.api_calls,
            "provider_latency_ms": round(self.provider_latency_ms, 2),
            "retries": self.retries,
            "fell_back": self.fell_back,
            "attempts": self.attempts,
            **self.extra,
        }


class EmbeddingModel:
    """Pipeline-facing embedding backend that degrades instead of raising."""

    def __init__(
        self,
        cfg: Config | RedundancyConfig | None = None,
        chain: EmbeddingChain | None = None,
    ) -> None:
        # Accepts a full Config (what it needs, to see `providers`) or a bare
        # RedundancyConfig (what callers passed before the migration), so
        # existing construction sites and tests keep working unchanged.
        if isinstance(cfg, RedundancyConfig):
            self.cfg = get_config().model_copy(update={"redundancy": cfg})
        else:
            self.cfg = cfg or get_config()
        self.redundancy = self.cfg.redundancy
        self.chain = chain or build_embedding_chain(self.cfg)
        self._error: str | None = None
        self._probed = False
        #: Cumulative provider counters already attributed to earlier calls, so
        #: each call reports its own cost rather than the process's running total.
        self._seen = {"calls": 0, "latency": 0.0, "retries": 0}
        self.stats = EncodeStats(
            device=resolve_device(self.redundancy.device),
            model=self.redundancy.model,
            chain=self.chain.names,
        )

    # -- readiness ---------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when some provider in the chain could run.

        Deliberately does not make a model call: a hosted provider with a valid
        key is "available" until it actually refuses, and finding that out costs
        real quota. A provider that fails at encode time is handled there.
        """
        ok, reason = self.chain.available()
        if not ok:
            self._error = reason
        self._probed = True
        return ok

    @property
    def load_error(self) -> str | None:
        if not self._probed:
            self.available
        return self._error

    # -- encoding ----------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Return L2-normalised embeddings, or None if the whole chain is down."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        started = time.perf_counter()
        try:
            vectors = self.chain.embed(texts)
        except NoProviderAvailable as exc:
            # Every provider skipped or failed. Stage 3 falls back to exact-hash
            # dedup and says so in its note; the pipeline still produces output.
            self._error = str(exc)
            self._probed = True
            log.warning("embedding unavailable (%s); exact-hash dedup only", exc)
            self._record(started)
            return None

        matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        self.stats.count += len(texts)
        self.stats.duration_ms += (time.perf_counter() - started) * 1000
        self.stats.dimension = int(matrix.shape[1]) if matrix.size else 0
        self._record(started, record_duration=False)
        self._error = None
        self._probed = True
        return matrix

    def _record(self, started: float, record_duration: bool = True) -> None:
        """Fold the chain's per-provider counters into this run's stats.

        The provider counters are cumulative over the process - the chain lives
        as long as the pipeline does - so they are *differenced* against the
        previous call rather than summed. Reporting the running total as
        ``embedding_latency_ms`` made a 500 ms request read as 24 s after the
        server had been up a while, which is a metric that gets less true the
        longer you leave it.
        """
        if record_duration:
            self.stats.duration_ms += (time.perf_counter() - started) * 1000
        telemetry = self.chain.telemetry()
        self.stats.provider = self.chain.last_provider
        self.stats.chain = self.chain.names
        self.stats.fell_back = bool(telemetry["fell_back"])
        self.stats.attempts = telemetry["attempts"]

        totals = {
            "calls": sum(p.stats.calls for p in self.chain.providers),
            "latency": sum(p.stats.latency_ms for p in self.chain.providers),
            "retries": sum(p.stats.retries for p in self.chain.providers),
        }
        self.stats.api_calls = totals["calls"] - self._seen["calls"]
        self.stats.provider_latency_ms = totals["latency"] - self._seen["latency"]
        self.stats.retries = totals["retries"] - self._seen["retries"]
        self._seen = totals
        active = next(
            (p for p in self.chain.providers if p.name == self.chain.last_provider),
            None,
        )
        if active is not None:
            self.stats.model = active.model
            self.stats.device = getattr(active, "device", "api")

    def warmup(self) -> float:
        """Resolve the chain and run one encode, returning the cost in ms.

        Worth more since the migration, not less: on the local provider this
        pays MiniLM's ~9 s load up front, and on a hosted one it resolves DNS,
        the TLS handshake and the chain order before a judge's first request.
        """
        started = time.perf_counter()
        self.encode(["warmup"])
        return (time.perf_counter() - started) * 1000

    def describe(self) -> dict:
        """Provider-aware readiness, surfaced by ``GET /health``."""
        ok, reason = self.chain.available()
        active = self.chain.active_provider_name()
        provider = next((p for p in self.chain.providers if p.name == active), None)
        return {
            "model": provider.model if provider else self.redundancy.model,
            "device": getattr(provider, "device", "api") if provider else "none",
            "available": ok,
            "error": None if ok else reason,
            "provider": active,
            "chain": self.chain.names,
            "last_used": self.chain.last_provider,
            "dimension": self.stats.dimension,
        }


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so stage 3's dot product really is cosine similarity.

    Cheap insurance rather than a no-op: OpenAI returns unit vectors today,
    Gemini and Cohere do not consistently, and none of them promise to keep
    doing whatever they currently do. A silently un-normalised row would not
    error - it would just quietly shift every similarity away from the
    configured 0.88 threshold and change what gets collapsed.
    """
    if matrix.ndim != 2 or matrix.size == 0:
        return np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
