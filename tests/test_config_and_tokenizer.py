"""Stage 1 tests: the config layer and the tokenizer that every metric depends on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    DensityWeights,
    load_config,
)
from engine.tokenizer import Tokenizer, get_tokenizer
from engine.config import TokenizerConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_shipped_config_file_is_valid():
    assert DEFAULT_CONFIG_PATH.exists(), "config.yaml is missing"
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, Config)


def test_default_budget_targets_seventy_percent_compression():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.selection.budget_ratio == pytest.approx(0.30), (
        "the headline target is a 70% reduction; budget_ratio must default to 0.30"
    )


def test_unknown_config_key_is_rejected():
    """A typo in config.yaml must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        Config.model_validate({"chunking": {"target_chunk_tokns": 100}})


def test_density_weights_normalise():
    weights = DensityWeights(entropy=1, tfidf=1, entities=1, novelty=1, structure=0)
    normalised = weights.normalised()
    assert sum(normalised.values()) == pytest.approx(1.0)
    assert normalised["entropy"] == pytest.approx(0.25)


def test_density_weights_reject_all_zero():
    with pytest.raises(ValueError):
        DensityWeights(entropy=0, tfidf=0, entities=0, novelty=0, structure=0).normalised()


def test_pricing_is_configured_not_hardcoded():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    price = cfg.pricing.price_for("gpt-4o-mini")
    assert price.input_per_1m > 0 and price.output_per_1m > 0
    assert cfg.pricing.default_model in cfg.pricing.known_models()


def test_pricing_for_unknown_model_raises():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    with pytest.raises(KeyError):
        cfg.pricing.price_for("not-a-real-model")


def test_overrides_merge_without_mutating_the_original():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    patched = cfg.with_overrides({"selection": {"budget_ratio": 0.15}})
    assert patched.selection.budget_ratio == pytest.approx(0.15)
    assert cfg.selection.budget_ratio == pytest.approx(0.30)
    # Untouched sibling fields survive the merge.
    assert patched.selection.protected_kinds == cfg.selection.protected_kinds


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg.selection.budget_ratio == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def test_tokenizer_uses_tiktoken_and_reports_exactness():
    tokenizer = get_tokenizer()
    assert tokenizer.is_exact, "tiktoken should be available in the dev environment"
    assert tokenizer.backend.startswith("tiktoken:")
    assert tokenizer.describe()["exact"] is True


def test_counts_are_consistent_between_single_and_batch():
    tokenizer = get_tokenizer()
    texts = ["hello world", "def f(x):\n    return x + 1", "", "a" * 300]
    assert tokenizer.count_many(texts) == [tokenizer.count(t) for t in texts]


def test_empty_text_costs_nothing():
    assert get_tokenizer().count("") == 0


def test_truncate_respects_the_budget():
    tokenizer = get_tokenizer()
    text = "The quick brown fox jumps over the lazy dog. " * 40
    truncated = tokenizer.truncate(text, 25)
    assert tokenizer.count(truncated) <= 25
    assert text.startswith(truncated[: len(truncated) - 1])


def test_truncate_is_a_noop_when_already_within_budget():
    tokenizer = get_tokenizer()
    assert tokenizer.truncate("short text", 100) == "short text"
    assert tokenizer.truncate("anything", 0) == ""


def test_estimate_fallback_is_flagged_as_inexact():
    """If tiktoken cannot load, counts must be marked approximate, not faked."""
    tokenizer = Tokenizer(TokenizerConfig(encoding="definitely-not-an-encoding"))
    assert tokenizer.is_exact is False
    assert tokenizer.backend == "estimate"
    assert tokenizer.count("the quick brown fox jumps") > 0
    assert tokenizer.count("") == 0


def test_estimate_fallback_is_in_the_right_ballpark():
    """The fallback must not silently distort the headline compression ratio."""
    exact = get_tokenizer()
    estimate = Tokenizer(TokenizerConfig(encoding="nope"))
    text = (
        "The checkout service opens one payment connection per in-flight charge. "
        "At peak the service sustains roughly 45 concurrent charges per pod."
    )
    ratio = estimate.count(text) / exact.count(text)
    assert 0.7 < ratio < 1.4, f"fallback estimate is off by {ratio:.2f}x"
