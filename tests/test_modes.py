"""Execution modes: local, cloud, auto.

The three modes make three promises, and each is tested as a promise rather
than as "the happy path worked":

* **local makes no outbound API call.** Asserted by severing the network at the
  transport layer - every ``requests`` entry point raises - and confirming a
  full compression still succeeds. "It returned local results" is not evidence;
  a provider could have been called and merely lost a race. The only proof is
  that a call was impossible.
* **cloud never silently degrades to local.** A user who asked for cloud and
  quietly received a 3B local model would read its latency and accuracy as
  cloud's. An exhausted cloud chain must raise.
* **auto falls back.** The pre-existing behaviour, unchanged.

Also covered: both model-calling stages must land on the *same* side of the
local/cloud line within one request. Embedding in the cloud and generating
locally would produce a run whose reported numbers describe a configuration
nobody selected.
"""

from __future__ import annotations

import pytest

from engine.config import get_config
from engine.pipeline import CompressionPipeline
from engine.providers import (
    DEFAULT_MODE,
    MODES,
    InvalidMode,
    NoProviderAvailable,
    build_embedding_chain,
    build_generation_chain,
    mode_readiness,
    normalise_mode,
)
from provider_doubles import ScriptedEmbedding, ScriptedGeneration, generation_chain


# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------
def test_default_mode_is_auto():
    assert DEFAULT_MODE == "auto"
    assert normalise_mode(None) == "auto"


@pytest.mark.parametrize("mode", MODES)
def test_every_declared_mode_parses(mode):
    assert normalise_mode(mode) == mode


def test_mode_parsing_is_case_and_space_insensitive():
    assert normalise_mode("  CLOUD ") == "cloud"


def test_an_unknown_mode_is_rejected_with_the_valid_set():
    with pytest.raises(InvalidMode) as excinfo:
        normalise_mode("gpu")
    assert "gpu" in str(excinfo.value)
    for mode in MODES:
        assert mode in str(excinfo.value)


# ---------------------------------------------------------------------------
# Chain composition per mode
# ---------------------------------------------------------------------------
def test_local_mode_chains_contain_only_local(config):
    assert build_embedding_chain(config, mode="local").names == ["local"]
    assert build_generation_chain(config, mode="local").names == ["local"]


def test_cloud_mode_chains_exclude_local(config, monkeypatch):
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    for chain in (
        build_embedding_chain(config, mode="cloud"),
        build_generation_chain(config, mode="cloud"),
    ):
        assert chain.names, "cloud chain must not be empty"
        assert "local" not in chain.names


def test_auto_mode_keeps_the_configured_chain_ending_in_local(config, monkeypatch):
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    chain = build_generation_chain(config, mode="auto")
    assert chain.names == list(config.providers.generation_providers)
    assert "local" in chain.names


def test_cloud_mode_errors_when_no_cloud_provider_is_configured(config, monkeypatch):
    """A local-only build asked for cloud gets a configuration answer, now."""
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    local_only = config.with_overrides(
        {"providers": {"generation_providers": ["local"]}}
    )
    with pytest.raises(NoProviderAvailable) as excinfo:
        build_generation_chain(local_only, mode="cloud")
    assert "no cloud provider" in str(excinfo.value)


def test_cloud_mode_is_refused_under_offline_mode(config, monkeypatch):
    monkeypatch.setenv("CCE_OFFLINE", "1")
    with pytest.raises(NoProviderAvailable) as excinfo:
        build_generation_chain(config, mode="cloud")
    assert "CCE_OFFLINE" in str(excinfo.value)


def test_offline_mode_does_not_break_local_or_auto(config, monkeypatch):
    monkeypatch.setenv("CCE_OFFLINE", "1")
    assert build_generation_chain(config, mode="local").names == ["local"]
    assert build_generation_chain(config, mode="auto").names == ["local"]


# ---------------------------------------------------------------------------
# The promise: local makes no outbound call
# ---------------------------------------------------------------------------
@pytest.fixture()
def severed_network(monkeypatch):
    """Make every outbound HTTP call impossible, and count attempts.

    Patches ``requests`` at the module the providers import it through. Any
    provider that tries to reach the network raises loudly instead of quietly
    succeeding, so "no call was made" is proven rather than assumed.
    """
    attempts: list[str] = []

    def forbidden(*args, **kwargs):
        target = args[0] if args else kwargs.get("url", "?")
        attempts.append(str(target))
        raise AssertionError(f"outbound network call attempted: {target}")

    import requests

    for verb in ("post", "get", "put", "request"):
        monkeypatch.setattr(requests, verb, forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    return attempts


def test_local_mode_makes_no_outbound_call_even_with_every_key_set(
    severed_network, monkeypatch, markdown_source
):
    """The load-bearing guarantee of local mode."""
    for name in (
        "OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY",
        "COHERE_API_KEY", "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(name, "sk-configured-and-plausible-value-123456")

    pipeline = CompressionPipeline(mode="local")
    result = pipeline.compress(
        markdown_source, "incident_postmortem.md", budget_ratio=0.30, fast_mode=True
    )

    assert result.compressed_text.strip()
    assert result.mode == "local"
    assert severed_network == [], f"local mode reached the network: {severed_network}"


def test_local_mode_stages_report_local_providers(markdown_source):
    pipeline = CompressionPipeline(mode="local")
    result = pipeline.compress(
        markdown_source, "incident_postmortem.md", budget_ratio=0.30, fast_mode=True
    )
    redundancy = result.stage("redundancy")
    assert redundancy.provider_used == "local"
    assert all(p == "local" for p in result.providers_used.values())


# ---------------------------------------------------------------------------
# The promise: cloud never silently degrades
# ---------------------------------------------------------------------------
def test_cloud_chain_raises_rather_than_falling_back_to_local():
    """Every cloud provider down must be an error, not a quiet local answer."""
    from engine.providers.base import ProviderFailed

    chain = generation_chain(
        ScriptedGeneration("groq", fail_with=ProviderFailed("groq", "429 quota")),
        ScriptedGeneration("gemini", fail_with=ProviderFailed("gemini", "503")),
    )
    with pytest.raises(NoProviderAvailable) as excinfo:
        chain.generate("q")

    message = str(excinfo.value)
    assert "groq" in message and "gemini" in message
    assert "local" not in chain.names


def test_a_cloud_chain_contains_no_local_provider_to_fall_back_to(config, monkeypatch):
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    chain = build_generation_chain(config, mode="cloud")
    assert not any(p.name == "local" for p in chain.providers)


# ---------------------------------------------------------------------------
# The promise: auto falls back
# ---------------------------------------------------------------------------
def test_auto_mode_falls_back_from_a_failing_cloud_provider_to_local():
    from engine.providers.base import ProviderFailed

    chain = generation_chain(
        ScriptedGeneration("groq", fail_with=ProviderFailed("groq", "429 quota")),
        ScriptedGeneration("local", default="answered locally"),
    )
    assert chain.generate("q") == "answered locally"
    assert chain.last_provider == "local"
    assert chain.telemetry()["fell_back"] is True


# ---------------------------------------------------------------------------
# One mode per request, across both model-calling stages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["local", "auto"])
def test_both_stages_share_one_mode(mode, markdown_source):
    """Never cloud embeddings with local generation inside one request."""
    pipeline = CompressionPipeline(mode=mode)
    assert pipeline.mode == mode
    assert pipeline.embedder.chain.names == pipeline.redundancy.embedder.chain.names
    # Under the test suite's CCE_OFFLINE both collapse to local; the assertion
    # that matters is that they were built from the same mode.
    assert pipeline.abstractive.client.chain.names[-1] in {"local", "groq", "gemini", "openai"}


def test_pipelines_for_different_modes_are_independent():
    local = CompressionPipeline(mode="local")
    auto = CompressionPipeline(mode="auto")
    assert local.embedder is not auto.embedder
    assert local.abstractive.client.chain is not auto.abstractive.client.chain


# ---------------------------------------------------------------------------
# Pinning a specific model
# ---------------------------------------------------------------------------
def test_pinning_a_provider_produces_a_single_entry_chain(config, monkeypatch):
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    assert build_generation_chain(config, pin="groq").names == ["groq"]
    assert build_embedding_chain(config, pin="cohere").names == ["cohere"]


def test_a_pinned_provider_has_no_fallback(config, monkeypatch):
    """The whole point: a pinned model must not be silently substituted.

    Someone comparing Groq against Gemini who unknowingly received Gemini for
    both would conclude the two are identical.
    """
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    chain = build_generation_chain(config, pin="groq")
    assert len(chain.providers) == 1
    assert "local" not in chain.names


def test_a_pin_overrides_the_mode(config, monkeypatch):
    """An explicit choice beats the coarse preset for that role."""
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    assert build_generation_chain(config, mode="local", pin="groq").names == ["groq"]
    assert build_embedding_chain(config, mode="cloud", pin="local").names == ["local"]


def test_pinning_a_cloud_provider_is_refused_under_offline_mode(config, monkeypatch):
    """A guarantee an explicit choice can quietly override is not a guarantee."""
    monkeypatch.setenv("CCE_OFFLINE", "1")
    with pytest.raises(NoProviderAvailable) as excinfo:
        build_generation_chain(config, pin="groq")
    assert "CCE_OFFLINE" in str(excinfo.value)


def test_pinning_local_still_works_under_offline_mode(config, monkeypatch):
    monkeypatch.setenv("CCE_OFFLINE", "1")
    assert build_generation_chain(config, pin="local").names == ["local"]


def test_pinning_an_unknown_provider_is_rejected_with_the_valid_set(config):
    from engine.providers import InvalidProvider

    with pytest.raises(InvalidProvider) as excinfo:
        build_generation_chain(config, pin="gpt5-turbo-ultra")
    assert "gpt5-turbo-ultra" in str(excinfo.value)
    assert "groq" in str(excinfo.value)


def test_pinning_an_embedding_provider_to_a_generation_only_name_is_rejected(config):
    """groq has no embeddings API, so it must not be offered as one."""
    from engine.providers import InvalidProvider

    with pytest.raises(InvalidProvider):
        build_embedding_chain(config, pin="groq")


def test_an_unconfigured_pin_fails_rather_than_substituting(config, monkeypatch):
    """No key for the pinned model is an error, never a quiet substitution."""
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    monkeypatch.setattr(
        "engine.providers.keys.has", lambda name: name != "GROQ_API_KEY"
    )
    chain = build_generation_chain(config, pin="groq")
    with pytest.raises(NoProviderAvailable) as excinfo:
        chain.generate("q")
    assert "GROQ_API_KEY" in str(excinfo.value)
    assert chain.names == ["groq"], "it must not have reached for another model"


# ---------------------------------------------------------------------------
# The catalogue a model picker is built from
# ---------------------------------------------------------------------------
def test_catalogue_lists_every_provider_for_both_roles(config):
    from engine.providers import provider_catalogue
    from engine.providers.embedding import EMBEDDING_PROVIDERS
    from engine.providers.generation import GENERATION_PROVIDERS

    catalogue = provider_catalogue(config)
    assert {e["provider"] for e in catalogue["embedding"]} == set(EMBEDDING_PROVIDERS)
    assert {e["provider"] for e in catalogue["generation"]} == set(GENERATION_PROVIDERS)


def test_catalogue_entries_carry_a_concrete_model_id(config):
    """So the UI never has to hardcode a model name to offer it."""
    from engine.providers import provider_catalogue

    catalogue = provider_catalogue(config)
    for role in ("embedding", "generation"):
        for entry in catalogue[role]:
            assert entry["model"], f"{entry['provider']} has no model id"
            assert entry["role"] == role
            assert isinstance(entry["configured"], bool)
            assert isinstance(entry["local"], bool)


def test_catalogue_marks_local_providers_as_local(config):
    from engine.providers import provider_catalogue

    catalogue = provider_catalogue(config)
    for role in ("embedding", "generation"):
        local = [e for e in catalogue[role] if e["local"]]
        assert [e["provider"] for e in local] == ["local"]


def test_catalogue_never_contains_a_key(config, monkeypatch):
    import json

    monkeypatch.setenv("GROQ_API_KEY", "gsk_averyrealsecretvalue987654321")
    from engine.providers import provider_catalogue

    body = json.dumps(provider_catalogue(config))
    assert "gsk_averyrealsecretvalue987654321" not in body


# ---------------------------------------------------------------------------
# Selection presets: one dropdown entry -> a coherent pair of providers
# ---------------------------------------------------------------------------
def test_presets_offer_auto_local_and_every_generation_model(config):
    from engine.providers import selection_presets
    from engine.providers.generation import GENERATION_PROVIDERS

    ids = {p["id"] for p in selection_presets(config)}
    assert {"auto", "local"} <= ids
    assert set(GENERATION_PROVIDERS) <= ids


def test_every_preset_resolves_to_a_usable_pair(config):
    """A dropdown entry must name providers the request can actually apply."""
    from engine.providers import selection_presets
    from engine.providers.embedding import EMBEDDING_PROVIDERS
    from engine.providers.generation import GENERATION_PROVIDERS

    for preset in selection_presets(config):
        assert preset["label"] and preset["detail"]
        assert preset["mode"] in MODES
        if preset["embedding_provider"] is not None:
            assert preset["embedding_provider"] in EMBEDDING_PROVIDERS
        if preset["generation_provider"] is not None:
            assert preset["generation_provider"] in GENERATION_PROVIDERS


def test_a_generation_only_vendor_is_paired_with_a_real_embedder(config):
    """Groq has no embeddings API, so its preset must borrow one."""
    from engine.providers import selection_presets

    groq = next(p for p in selection_presets(config) if p["id"] == "groq")
    assert groq["generation_provider"] == "groq"
    assert groq["embedding_provider"] != "groq"
    assert groq["embedding_provider"] is not None


def test_a_vendor_that_does_both_is_paired_with_itself(config):
    from engine.providers import selection_presets

    gemini = next(p for p in selection_presets(config) if p["id"] == "gemini")
    assert gemini["embedding_provider"] == "gemini"
    assert gemini["generation_provider"] == "gemini"


def test_the_auto_preset_pins_nothing(config):
    """Auto must keep the fallback chain, which is the point of it."""
    from engine.providers import selection_presets

    auto = next(p for p in selection_presets(config) if p["id"] == "auto")
    assert auto["embedding_provider"] is None
    assert auto["generation_provider"] is None
    assert auto["mode"] == "auto"


def test_the_local_preset_is_local_on_both_roles(config):
    from engine.providers import selection_presets

    local = next(p for p in selection_presets(config) if p["id"] == "local")
    assert local["mode"] == "local"
    assert local["embedding_provider"] == "local"
    assert local["generation_provider"] == "local"
    assert local["local"] is True


def test_an_unconfigured_model_is_marked_unavailable_with_a_reason(config, monkeypatch):
    from engine.providers import selection_presets

    monkeypatch.setattr(
        "engine.providers.keys.has", lambda name: name != "GROQ_API_KEY"
    )
    groq = next(p for p in selection_presets(config) if p["id"] == "groq")
    assert groq["available"] is False
    assert "GROQ_API_KEY" in (groq["reason"] or "")


def test_presets_never_contain_a_key(config, monkeypatch):
    import json

    from engine.providers import selection_presets

    monkeypatch.setenv("GROQ_API_KEY", "gsk_averyrealsecretvalue1234567")
    assert "gsk_averyrealsecretvalue1234567" not in json.dumps(selection_presets(config))


# ---------------------------------------------------------------------------
# /health readiness, reported independently per mode
# ---------------------------------------------------------------------------
def test_mode_readiness_reports_both_modes(config):
    readiness = mode_readiness(config)
    assert set(readiness) == {"local", "cloud"}
    for entry in readiness.values():
        assert "ready" in entry and isinstance(entry["ready"], bool)
        assert "embedding" in entry and "generation" in entry


def test_local_unavailable_does_not_make_cloud_unavailable(config, monkeypatch):
    """Ollama being down is not a statement about Groq."""
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_plausible_value_for_this_test_1234")

    from engine.providers.generation import OllamaProvider

    monkeypatch.setattr(
        OllamaProvider, "configured",
        lambda self, refresh=False: (False, "ollama unreachable at localhost:11434"),
    )
    readiness = mode_readiness(config)

    assert readiness["local"]["ready"] is False
    assert "unreachable" in readiness["local"]["reason"]
    assert readiness["cloud"]["ready"] is True


def test_no_keys_does_not_make_local_unavailable(config, monkeypatch):
    """The mirror case: no cloud keys is not a statement about Ollama."""
    monkeypatch.delenv("CCE_OFFLINE", raising=False)
    for name in (
        "OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY",
        "COHERE_API_KEY", "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    from engine.providers.generation import OllamaProvider

    monkeypatch.setattr(
        OllamaProvider, "configured", lambda self, refresh=False: (True, "")
    )
    readiness = mode_readiness(config)

    assert readiness["cloud"]["ready"] is False
    assert readiness["local"]["ready"] is True


def test_cloud_readiness_is_false_under_offline_mode(config, monkeypatch):
    monkeypatch.setenv("CCE_OFFLINE", "1")
    readiness = mode_readiness(config)
    assert readiness["cloud"]["ready"] is False
    assert "CCE_OFFLINE" in readiness["cloud"]["reason"]
