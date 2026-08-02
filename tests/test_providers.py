"""Provider layer tests: keys, batching, and above all the fallback chain.

The fallback chain is the claim the whole architecture rests on - "if one API
is rate-limited or down, the system automatically falls back to the next" - so
it is tested directly rather than inferred from the pipeline still working.

Everything here runs against scripted providers (``tests/provider_doubles.py``)
that implement the real contracts, so these exercise the shipped chain code and
not a rehearsal of it. Nothing here touches a network or needs a key. Tests
that hit real APIs live in ``tests/test_integration_providers.py`` and are
opt-in.
"""

from __future__ import annotations

import time

import pytest

from engine.providers import build_embedding_chain, build_generation_chain, provider_status
from engine.providers.base import (
    NoProviderAvailable,
    ProviderFailed,
    ProviderUnavailable,
)
from engine.providers.chain import EmbeddingChain, GenerationChain
from engine.providers import keys
from provider_doubles import (
    ScriptedEmbedding,
    ScriptedGeneration,
    embedding_chain,
    generation_chain,
)


# ---------------------------------------------------------------------------
# Key handling - never logged, never returned, placeholders are not keys
# ---------------------------------------------------------------------------
def test_missing_key_makes_a_provider_skippable_not_broken(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from engine.providers.generation import GroqProvider

    ok, reason = GroqProvider().configured()
    assert not ok
    assert "GROQ_API_KEY" in reason


def test_placeholder_key_is_treated_as_absent(monkeypatch):
    """A copied .env.example must read as 'no key', not produce a confusing 401."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-your-key-here")
    assert not keys.has("GROQ_API_KEY")


def test_redact_strips_key_shapes_and_configured_values(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "abcdefghijklmnopqrstuvwxyz012345")
    dirty = (
        "failed with sk-proj-AAAAAAAAAAAAAAAAAAAA and AIzaSyAAAAAAAAAAAAAAAAAAAAAAAA "
        "and abcdefghijklmnopqrstuvwxyz012345"
    )
    clean = keys.redact(dirty)
    assert "sk-proj-" not in clean
    assert "AIzaSy" not in clean
    assert "abcdefghijklmnopqrstuvwxyz012345" not in clean
    assert clean.count("[redacted]") == 3


def test_provider_errors_are_redacted_on_the_way_out(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-SUPERSECRETVALUE123456")
    error = ProviderFailed("openai", "401 for key sk-proj-SUPERSECRETVALUE123456")
    assert "SUPERSECRET" not in str(error)
    assert "[redacted]" in str(error)


# ---------------------------------------------------------------------------
# Batching - the reason a free tier survives a demo
# ---------------------------------------------------------------------------
def test_embeddings_are_batched_to_the_provider_limit():
    provider = ScriptedEmbedding(batch_limit=4)
    vectors = provider.embed([f"text {i}" for i in range(10)])

    assert len(vectors) == 10
    assert [len(batch) for batch in provider.batches] == [4, 4, 2]
    assert provider.stats.calls == 3, "one HTTP call per batch, not per text"


def test_batched_embeddings_stay_in_input_order():
    provider = ScriptedEmbedding(batch_limit=3)
    texts = [f"text {i}" for i in range(7)]
    vectors = provider.embed(texts)

    expected = [provider._embed_batch([text])[0] for text in texts]
    assert vectors == expected


def test_a_transient_batch_failure_is_retried_exactly_once():
    provider = ScriptedEmbedding(
        batch_limit=8,
        fail_with=ProviderFailed("scripted", "503 upstream unavailable"),
        fail_times=1,
    )
    vectors = provider.embed(["a", "b"])

    assert len(vectors) == 2
    assert provider.stats.retries == 1
    assert len(provider.batches) == 2, "one failure plus one retry"


@pytest.mark.parametrize(
    "reason",
    ["429 quota exceeded", "rejected the key (401): Invalid API Key",
     "model or endpoint not found (404)", "no credits remaining"],
)
def test_a_settled_failure_is_not_retried(reason):
    """Quota, bad key and retired model will not resolve in 0.4 s.

    Retrying them burns a round trip to confirm what the first call already
    established, while a working provider waits further down the chain.
    Measured: skipping these cut a cold compression from ~3.1 s to ~1.6 s.
    """
    provider = ScriptedEmbedding(fail_with=ProviderFailed("scripted", reason))

    with pytest.raises(ProviderFailed):
        provider.embed(["a"])

    assert len(provider.batches) == 1, f"{reason!r} must not be retried"
    assert provider.stats.retries == 0


def test_a_batch_that_keeps_failing_gives_up_rather_than_looping():
    provider = ScriptedEmbedding(fail_with=ProviderFailed("scripted", "500"))
    with pytest.raises(ProviderFailed):
        provider.embed(["a"])
    assert len(provider.batches) == 2, "one retry only, then surrender to the chain"


def test_empty_input_never_reaches_the_provider():
    provider = ScriptedEmbedding()
    assert provider.embed([]) == []
    assert provider.stats.calls == 0


# ---------------------------------------------------------------------------
# The fallback chain
# ---------------------------------------------------------------------------
def test_generation_falls_through_a_failing_primary_to_the_next():
    """The headline resilience claim, tested directly."""
    primary = ScriptedGeneration("primary", fail_with=ProviderFailed("primary", "429"))
    secondary = ScriptedGeneration("secondary", default="answer from secondary")
    chain = generation_chain(primary, secondary)

    assert chain.generate("question") == "answer from secondary"
    assert chain.last_provider == "secondary"
    assert primary.calls, "the primary must actually have been tried"
    assert chain.telemetry()["fell_back"] is True


def test_generation_falls_through_a_timeout():
    slow = ScriptedGeneration("slow", fail_with=ProviderFailed("slow", "timed out after 10s"))
    fast = ScriptedGeneration("fast", default="ok")
    chain = generation_chain(slow, fast)

    assert chain.generate("q") == "ok"
    attempts = {a.provider: a for a in chain.last_attempts}
    assert "timed out" in attempts["slow"].reason
    assert attempts["fast"].ok


def test_a_provider_without_a_key_is_skipped_not_failed():
    unkeyed = ScriptedGeneration(
        "unkeyed", configured=False, unavailable_reason="NOPE_API_KEY is not configured"
    )
    live = ScriptedGeneration("live", default="ok")
    chain = generation_chain(unkeyed, live)

    assert chain.generate("q") == "ok"
    attempts = {a.provider: a for a in chain.last_attempts}
    assert attempts["unkeyed"].skipped is True
    assert not unkeyed.calls, "an unconfigured provider must not be called at all"


def test_the_first_working_provider_wins_and_the_rest_are_untouched():
    first = ScriptedGeneration("first", default="from first")
    second = ScriptedGeneration("second", default="from second")
    chain = generation_chain(first, second)

    assert chain.generate("q") == "from first"
    assert not second.calls
    assert chain.telemetry()["fell_back"] is False


def test_an_exhausted_chain_raises_a_clear_error_naming_every_provider():
    """The one hard failure this layer produces must say what to configure."""
    chain = generation_chain(
        ScriptedGeneration("groq", configured=False, unavailable_reason="GROQ_API_KEY is not configured"),
        ScriptedGeneration("gemini", fail_with=ProviderFailed("gemini", "429 quota")),
        ScriptedGeneration("local", fail_with=ProviderUnavailable("local", "ollama unreachable")),
    )

    with pytest.raises(NoProviderAvailable) as excinfo:
        chain.generate("q")

    message = str(excinfo.value)
    assert "no generation provider available" in message
    for expected in ("groq", "GROQ_API_KEY", "gemini", "429 quota", "local", "unreachable"):
        assert expected in message
    assert ".env.example" in message, "the error should say how to fix it"


def test_an_exhausted_chain_raises_rather_than_returning_empty_text():
    chain = generation_chain(ScriptedGeneration("only", fail_with=ProviderFailed("only", "boom")))
    with pytest.raises(NoProviderAvailable):
        chain.generate("q")


def test_embedding_chain_falls_back_and_reports_which_provider_served():
    primary = ScriptedEmbedding("primary", fail_with=ProviderFailed("primary", "429"))
    secondary = ScriptedEmbedding("secondary", dimension=5)
    chain = embedding_chain(primary, secondary)

    vectors = chain.embed(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 5
    assert chain.last_provider == "secondary"


def test_one_embedding_provider_serves_a_whole_document():
    """Mixing vector spaces mid-document would silently corrupt clustering."""
    primary = ScriptedEmbedding("primary", dimension=8, batch_limit=2)
    secondary = ScriptedEmbedding("secondary", dimension=3, batch_limit=2)
    chain = embedding_chain(primary, secondary)

    vectors = chain.embed([f"t{i}" for i in range(6)])
    assert {len(v) for v in vectors} == {8}, "all rows must come from one provider"
    assert not secondary.batches


# ---------------------------------------------------------------------------
# Cooldown - do not pay the same timeout on every chunk of a stage
# ---------------------------------------------------------------------------
def test_a_failed_provider_is_skipped_for_the_cooldown_window():
    flaky = ScriptedGeneration("flaky", fail_with=ProviderFailed("flaky", "429"))
    backup = ScriptedGeneration("backup", default="ok")
    chain = generation_chain(flaky, backup, cooldown_s=30.0)

    chain.generate("first")
    assert len(flaky.calls) == 1

    chain.generate("second")
    assert len(flaky.calls) == 1, "a cooling provider must not be retried"
    attempts = {a.provider: a for a in chain.last_attempts}
    assert attempts["flaky"].skipped and "cooldown" in attempts["flaky"].reason


def test_a_provider_returns_after_its_cooldown_expires():
    flaky = ScriptedGeneration("flaky", fail_with=ProviderFailed("flaky", "429"))
    backup = ScriptedGeneration("backup", default="ok")
    chain = generation_chain(flaky, backup, cooldown_s=0.05)

    chain.generate("first")
    time.sleep(0.06)
    chain.generate("second")
    assert len(flaky.calls) == 2, "the cooldown must expire, not demote permanently"


def test_an_unconfigured_provider_is_not_put_in_cooldown():
    """Missing a key is a steady state, not a failure to back off from."""
    unkeyed = ScriptedGeneration("unkeyed", configured=False)
    live = ScriptedGeneration("live", default="ok")
    chain = generation_chain(unkeyed, live, cooldown_s=30.0)

    chain.generate("q")
    assert "unkeyed" not in chain._cooldown_until


def test_a_timeout_does_not_put_a_provider_in_cooldown():
    """Regression: stage 6's own impatience must not mark a model unhealthy.

    Stage 6 shrinks its per-call timeout as its wall-clock ceiling runs down.
    Once that budget drops to a second or two every provider "times out" - and
    cooling one off for that made a healthy local Ollama unavailable to the
    next minute of unrelated requests.
    """
    slow = ScriptedGeneration(
        "slow", fail_with=ProviderFailed("slow", "timed out after 2s", kind="timeout")
    )
    backup = ScriptedGeneration("backup", default="ok")
    chain = generation_chain(slow, backup, cooldown_s=30.0)

    chain.generate("q")
    assert "slow" not in chain._cooldown_until
    chain.generate("q")
    assert len(slow.calls) == 2, "a timed-out provider must still be tried next time"


def test_a_rate_limit_still_puts_a_provider_in_cooldown():
    """The other half of the rule: a 429 is about the provider, so back off."""
    limited = ScriptedGeneration(
        "limited", fail_with=ProviderFailed("limited", "429 quota", kind="other")
    )
    backup = ScriptedGeneration("backup", default="ok")
    chain = generation_chain(limited, backup, cooldown_s=30.0)

    chain.generate("q")
    assert "limited" in chain._cooldown_until


# ---------------------------------------------------------------------------
# Chain construction from config
# ---------------------------------------------------------------------------
def test_config_drives_the_chain_order(config):
    cfg = config.with_overrides(
        {"providers": {"generation_providers": ["gemini", "groq", "local"]}}
    )
    chain = build_generation_chain(cfg, force_local=False)
    assert chain.names == ["gemini", "groq", "local"]


def test_embedding_primary_leads_its_fallbacks(config):
    cfg = config.with_overrides(
        {"providers": {"embedding_provider": "cohere", "embedding_fallback": ["gemini", "local"]}}
    )
    chain = build_embedding_chain(cfg, force_local=False)
    assert chain.names == ["cohere", "gemini", "local"]


def test_a_duplicated_primary_is_not_tried_twice(config):
    cfg = config.with_overrides(
        {"providers": {"embedding_provider": "local", "embedding_fallback": ["local"]}}
    )
    assert build_embedding_chain(cfg, force_local=False).names == ["local"]


def test_offline_mode_collapses_both_chains_to_local(config):
    assert build_generation_chain(config, force_local=True).names == ["local"]
    assert build_embedding_chain(config, force_local=True).names == ["local"]


def test_an_unknown_provider_name_is_rejected_at_config_load(config):
    with pytest.raises(Exception) as excinfo:
        config.with_overrides({"providers": {"generation_providers": ["not-a-provider"]}})
    assert "not-a-provider" in str(excinfo.value)


def test_an_empty_generation_chain_is_rejected(config):
    with pytest.raises(Exception):
        config.with_overrides({"providers": {"generation_providers": []}})


# ---------------------------------------------------------------------------
# /health reporting
# ---------------------------------------------------------------------------
def test_provider_status_reports_presence_only_never_a_key(monkeypatch, config):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_averyrealsecretvalue123456")
    status = provider_status(config)

    assert status["keys_configured"]["GROQ_API_KEY"] is True
    assert "gsk_averyrealsecretvalue123456" not in repr(status)
    assert all(isinstance(v, bool) for v in status["keys_configured"].values())


def test_provider_status_lists_both_chains(config):
    status = provider_status(config)
    assert status["embedding"]["chain"]
    assert status["generation"]["chain"]
    assert "active" in status["embedding"] and "active" in status["generation"]
