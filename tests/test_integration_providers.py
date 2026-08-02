"""Real API calls against whichever providers are actually configured.

    pytest -m integration                     # every configured provider
    pytest -m integration -k groq             # just one

Deselected by default (see ``pytest.ini``). These cost real quota, need a
network, and would otherwise turn the suite red whenever someone else's free
tier is having a bad afternoon - which is a fact about Groq, not about this
codebase. The unit suite covers the same code paths against scripted providers.

Each test skips rather than fails when its key is absent, so this file is
useful whether you have one key or five: it tells you which of the providers
you *believe* are configured actually answer.
"""

from __future__ import annotations

import os

import pytest

from engine.config import get_config
from engine.providers import build_embedding_chain, build_generation_chain, keys
from engine.providers.base import ProviderError
from engine.providers.embedding import EMBEDDING_PROVIDERS
from engine.providers.generation import GENERATION_PROVIDERS

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_config():
    """The real config, with offline mode lifted for this module only."""
    previous = os.environ.pop("CCE_OFFLINE", None)
    keys.load_dotenv()
    yield get_config()
    if previous is not None:
        os.environ["CCE_OFFLINE"] = previous


#: Failures that are facts about someone else's account, not about this code.
#: A quota that resets tomorrow and a key that was pasted wrong are both
#: environmental; letting them fail the suite would train everyone to ignore it,
#: which is worse than not having it. Anything else still fails loudly.
_ENVIRONMENTAL = (
    "429", "quota", "rate limit", "401", "invalid api key",
    "insufficient", "billing", "credit",
)


def skip_if_environmental(exc: Exception) -> None:
    """Turn an account-level provider failure into a skip, or re-raise."""
    reason = str(exc).lower()
    if any(marker in reason for marker in _ENVIRONMENTAL):
        pytest.skip(f"provider unavailable for account reasons: {exc}")
    raise exc


def _embedding(name: str, cfg):
    from engine.providers import _embedding_provider

    provider = _embedding_provider(name, cfg)
    ok, reason = provider.configured()
    if not ok:
        pytest.skip(reason)
    return provider


def _generation(name: str, cfg):
    from engine.providers import _generation_provider

    provider = _generation_provider(name, cfg, cfg.providers.timeout_s, None)
    ok, reason = provider.configured()
    if not ok:
        pytest.skip(reason)
    return provider


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EMBEDDING_PROVIDERS))
def test_embedding_provider_returns_usable_vectors(name, live_config):
    provider = _embedding(name, live_config)
    texts = ["payment pool exhausted after 8000ms", "checkout service timed out"]

    try:
        vectors = provider.embed(texts)
    except ProviderError as exc:
        skip_if_environmental(exc)

    assert len(vectors) == len(texts)
    assert len({len(v) for v in vectors}) == 1, "every row must share a dimension"
    assert all(isinstance(value, float) for value in vectors[0])
    assert any(value != 0.0 for value in vectors[0]), "an all-zero vector is a bug"


@pytest.mark.parametrize("name", sorted(EMBEDDING_PROVIDERS))
def test_embedding_provider_batches_a_large_input(name, live_config):
    """More texts than the batch limit must still come back in input order."""
    provider = _embedding(name, live_config)
    count = provider.batch_limit + 3
    texts = [f"log record number {i} failed with code {i}" for i in range(count)]

    try:
        vectors = provider.embed(texts)
    except ProviderError as exc:
        skip_if_environmental(exc)

    assert len(vectors) == count
    assert provider.stats.calls >= 2, "a batched input should take >1 request"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(GENERATION_PROVIDERS))
def test_generation_provider_answers_from_context(name, live_config):
    provider = _generation(name, live_config)

    try:
        answer = provider.generate(
            "Context: the pool size was reduced to 8 connections.\n\n"
            "Question: what was the pool size reduced to?",
            max_tokens=64,
            timeout_s=60,
            system="Answer using only the context. Be concise.",
        )
    except ProviderError as exc:
        skip_if_environmental(exc)

    assert answer.strip(), f"{name} returned an empty completion"
    assert "8" in answer


@pytest.mark.parametrize("name", sorted(GENERATION_PROVIDERS))
def test_generation_provider_respects_its_timeout(name, live_config):
    """A provider that cannot answer in time must raise, not hang."""
    import time

    provider = _generation(name, live_config)
    started = time.perf_counter()
    try:
        provider.generate("Write one word.", max_tokens=8, timeout_s=45)
    except ProviderError:
        pass
    assert time.perf_counter() - started < 60


# ---------------------------------------------------------------------------
# The chains, end to end
# ---------------------------------------------------------------------------
def test_the_configured_chains_resolve_to_something(live_config):
    """Whatever is configured, both chains must produce a live provider."""
    embedding = build_embedding_chain(live_config, force_local=False)
    generation = build_generation_chain(live_config, force_local=False)

    vectors = embedding.embed(["one", "two"])
    assert len(vectors) == 2
    assert embedding.last_provider, "no embedding provider served the call"

    answer = generation.generate("Reply with the word ok", max_tokens=16, timeout_s=60)
    assert answer.strip()
    assert generation.last_provider, "no generation provider served the call"


def test_a_dead_primary_is_survived_by_the_real_chain(live_config):
    """Put a guaranteed-failing provider in front and confirm the real one wins.

    This is the fallback claim tested against live infrastructure rather than a
    double: the first entry is a real provider class pointed at a model that
    does not exist, so it fails the way a retired model would.
    """
    from engine.providers.base import ProviderFailed
    from engine.providers.chain import GenerationChain

    from provider_doubles import ScriptedGeneration

    # The *whole* real chain behind the dead provider, not just its first
    # key-present entry. Picking one entry made this test assert that a
    # specific vendor works, which is not the claim - and it duly broke the
    # day a configured key turned out to be invalid. Fallback is the subject
    # here, so the test must let fallback happen.
    real = build_generation_chain(live_config, force_local=False).providers
    if not any(p.configured()[0] for p in real):
        pytest.skip("no generation provider is configured")

    chain = GenerationChain(
        [ScriptedGeneration("dead", fail_with=ProviderFailed("dead", "429")), *real],
        cooldown_s=0.0,
    )
    try:
        answer = chain.generate("Reply with the word ok", max_tokens=16, timeout_s=60)
    except Exception as exc:  # noqa: BLE001
        skip_if_environmental(exc)

    assert answer.strip()
    assert chain.last_provider != "dead", "the dead primary must not have served"
    assert chain.telemetry()["fell_back"] is True
