"""Scripted provider doubles - the whole suite's substitute for a real API.

These implement the same ``EmbeddingProvider`` / ``GenerationProvider``
contracts the real providers do, so tests exercise the *actual* chain, batching
and fallback code rather than a mock of it. What they replace is only the HTTP
call at the very bottom.

That distinction matters for the fallback tests: a chain assembled from these
doubles walks exactly the code path a chain of real providers walks, so a test
that says "the chain falls through a timing-out primary to the next provider"
is testing the shipped logic, not a rehearsal of it.
"""

from __future__ import annotations

from engine.abstractive import GenerationClient
from engine.config import get_config
from engine.providers.base import (
    EmbeddingProvider,
    GenerationProvider,
    ProviderFailed,
    ProviderUnavailable,
)
from engine.providers.chain import EmbeddingChain, GenerationChain


class ScriptedGeneration(GenerationProvider):
    """A generation provider whose behaviour the test states outright.

    ``responses`` maps a substring of the prompt to the completion to return.
    ``fail_with`` makes every call raise instead - the way a rate-limited or
    timing-out provider behaves, so a chain can be watched falling past it.
    """

    def __init__(
        self,
        name: str = "scripted",
        responses: dict[str, str] | None = None,
        *,
        model: str = "scripted-model",
        fail_with: Exception | None = None,
        configured: bool = True,
        unavailable_reason: str = "scripted: not configured",
        default: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.model = model
        self.env_key = None
        self._responses = responses or {}
        self._fail_with = fail_with
        self._configured = configured
        self._reason = unavailable_reason
        self._default = default
        #: Every prompt this provider was asked for, in order.
        self.calls: list[str] = []

    def configured(self) -> tuple[bool, str]:
        return (True, "") if self._configured else (False, self._reason)

    def _generate(self, prompt, max_tokens, timeout_s, system):
        self.calls.append(prompt)
        if self._fail_with is not None:
            raise self._fail_with
        for needle, response in self._responses.items():
            if needle in prompt:
                return response
        if self._default is not None:
            return self._default
        raise ProviderFailed(self.name, "no scripted response for this prompt")


class ScriptedEmbedding(EmbeddingProvider):
    """Deterministic vectors, with a recorded batch history.

    The vectors are a cheap bag-of-characters hash rather than anything
    meaningful - the tests that use this care about batching, ordering and
    fallback, not about similarity quality.
    """

    def __init__(
        self,
        name: str = "scripted",
        *,
        dimension: int = 8,
        batch_limit: int = 4,
        fail_with: Exception | None = None,
        fail_times: int = 0,
        configured: bool = True,
        unavailable_reason: str = "scripted: not configured",
    ) -> None:
        super().__init__()
        self.name = name
        self.model = "scripted-embedding"
        self.env_key = None
        self.dimension = dimension
        self.batch_limit = batch_limit
        self._fail_with = fail_with
        self._fail_times = fail_times
        self._configured = configured
        self._reason = unavailable_reason
        #: One entry per ``_embed_batch`` call, holding that batch's texts.
        self.batches: list[list[str]] = []

    def configured(self) -> tuple[bool, str]:
        return (True, "") if self._configured else (False, self._reason)

    def _embed_batch(self, texts):
        self.batches.append(list(texts))
        if self._fail_with is not None and (
            self._fail_times == 0 or len(self.batches) <= self._fail_times
        ):
            raise self._fail_with
        return [
            [float((sum(map(ord, text)) + axis) % 17) for axis in range(self.dimension)]
            for text in texts
        ]


def generation_chain(*providers: GenerationProvider, cooldown_s: float = 0.0) -> GenerationChain:
    """A chain over doubles. Cooldown defaults to 0 so tests stay order-only."""
    return GenerationChain(list(providers), cooldown_s=cooldown_s)


def embedding_chain(*providers: EmbeddingProvider, cooldown_s: float = 0.0) -> EmbeddingChain:
    return EmbeddingChain(list(providers), cooldown_s=cooldown_s)


class ScriptedClient(GenerationClient):
    """Stage 6's client, backed by one scripted provider.

    Keeps the shape stage 6's existing tests were written against - construct
    with a dict of prompt-substring -> completion, read ``.calls`` afterwards -
    while running through the real ``GenerationClient`` and ``GenerationChain``
    underneath.
    """

    def __init__(self, responses=None, available=True, delay=0.0):
        self._delay = delay
        # Named with a leading underscore: `provider` is a read-only property on
        # GenerationClient (it reports which chain entry last answered).
        self._scripted = ScriptedGeneration(
            "scripted",
            responses or {},
            configured=available,
            unavailable_reason="scripted: unavailable",
        )
        if delay:
            self._scripted._generate = _delayed(self._scripted._generate, delay)
        super().__init__(generation_chain(self._scripted), get_config().abstractive)

    @property
    def calls(self) -> list[str]:
        return self._scripted.calls


def _delayed(fn, delay: float):
    import time

    def wrapper(*args, **kwargs):
        time.sleep(delay)
        return fn(*args, **kwargs)

    return wrapper
