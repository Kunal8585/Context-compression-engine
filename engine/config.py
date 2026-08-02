"""Typed, validated access to ``config.yaml``.

Nothing in the pipeline is allowed to hardcode a threshold, weight or budget.
Every stage takes its settings from the objects in this module so that a judge
can point at one file and see exactly what the engine is doing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class _Base(BaseModel):
    """Strict base: an unknown key is a typo, not a feature."""

    model_config = ConfigDict(extra="forbid")


class TokenizerConfig(_Base):
    encoding: str = "cl100k_base"
    fallback_tokens_per_word: float = 1.3


class ChunkingConfig(_Base):
    target_chunk_tokens: int = 120
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 12
    merge_small_chunks: bool = True
    mergeable_kinds: list[str] = Field(
        default_factory=lambda: ["module_level", "paragraph", "sentence", "other"]
    )
    attach_leading_comments: bool = True
    max_parse_error_ratio: float = 0.30


class StructuralDedupConfig(_Base):
    enabled: bool = True
    min_tokens: int = 30
    kinds: list[str] = Field(
        default_factory=lambda: ["function", "method", "class"]
    )


class RedundancyConfig(_Base):
    enabled: bool = True
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.88
    #: Per-provider override. Cosine is not a shared scale across embedding
    #: models: on the same 26 near-duplicate tickets MiniLM's highest pair
    #: scores 0.908 while Cohere's tops out at 0.877, so one threshold applied
    #: to both silently disables fuzzy dedup for the second. Providers absent
    #: from this map fall back to `similarity_threshold` and the stage says so.
    similarity_threshold_by_provider: dict[str, float] = Field(
        default_factory=lambda: {"local": 0.88, "cohere": 0.84}
    )
    exact_hash_dedup: bool = True
    structural: StructuralDedupConfig = Field(default_factory=StructuralDedupConfig)
    batch_size: int = 64
    device: str = "auto"


class DensityWeights(_Base):
    entropy: float = 0.25
    tfidf: float = 0.25
    entities: float = 0.25
    novelty: float = 0.15
    structure: float = 0.10
    #: Cosine similarity to the user's question. Unavailable - and its weight
    #: redistributed over the others - whenever no query was supplied, so a
    #: query-less compression scores exactly as it did before this existed.
    query_relevance: float = 0.0

    def normalised(self) -> dict[str, float]:
        raw = self.model_dump()
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("density.weights must sum to a positive number")
        return {k: v / total for k, v in raw.items()}


class DensityConfig(_Base):
    weights: DensityWeights = Field(default_factory=DensityWeights)
    spacy_model: str = "en_core_web_sm"
    spacy_max_chars: int = 400_000
    frequency_boost: float = 0.15
    #: Per-kind structural prior in [0, 1]. Missing kinds fall back to `default`.
    structure_priors: dict[str, float] = Field(
        default_factory=lambda: {
            "query": 1.0,
            "instruction": 1.0,
            "system": 1.0,
            "function": 0.70,
            "method": 0.70,
            "class": 0.70,
            "class_header": 0.60,
            "code_block": 0.60,
            "paragraph": 0.50,
            "list": 0.50,
            "heading": 0.50,
            "log_record": 0.40,
            "module_level": 0.40,
            "sentence": 0.40,
            "other": 0.30,
            "default": 0.40,
        }
    )
    #: Log severity overrides the per-kind prior when a level was detected.
    level_priors: dict[str, float] = Field(
        default_factory=lambda: {
            "CRITICAL": 1.0,
            "FATAL": 1.0,
            "SEVERE": 1.0,
            "ERROR": 0.95,
            "WARNING": 0.80,
            "WARN": 0.80,
            "NOTICE": 0.50,
            "INFO": 0.35,
            "DEBUG": 0.20,
            "TRACE": 0.15,
        }
    )
    #: Chunk length at which the entropy estimate is considered fully reliable.
    entropy_reference_tokens: int = 64
    #: Added to the structural prior when failure vocabulary is present.
    keyword_boost: float = 0.25
    #: Kinds where failure vocabulary is type names, not events.
    keyword_boost_excluded_kinds: list[str] = Field(
        default_factory=lambda: [
            "function", "method", "class", "class_header", "module_level",
        ]
    )
    keywords: list[str] = Field(
        default_factory=lambda: [
            "traceback", "exception", "stacktrace", "caused by", "panic",
            "timeout", "timed out", "failed", "failure", "refused", "denied",
            "exhausted", "corrupt", "deadlock", "root cause", "regression",
            "rollback", "outage", "incident", "breach", "data loss",
        ]
    )


class SelectionConfig(_Base):
    budget_ratio: float = 0.30
    protected_kinds: list[str] = Field(
        default_factory=lambda: ["query", "instruction", "system"]
    )
    min_chunks_kept: int = 1
    #: Spend leftover budget re-admitting definitions that kept code references.
    preserve_code_dependencies: bool = True
    #: Fraction of the budget held back for stage 7's drop markers.
    marker_reserve: float = 0.03


class AbstractiveConfig(_Base):
    """Stage 6's own thresholds. *Which* model runs it lives in `providers`."""

    enabled: bool = True
    min_tokens_to_compress: int = 150
    target_ratio: float = 0.55
    #: Per-call timeout. A slow paraphrase is abandoned, not waited on.
    timeout_s: float = 10.0
    #: Whole-stage wall-clock ceiling; the demo must stay responsive.
    total_timeout_s: float = 30.0
    max_chunks: int = 12
    temperature: float = 0.0
    #: Fraction of the original's critical tokens a paraphrase must retain.
    entity_overlap_threshold: float = 0.90
    #: Numbers are never negotiable - 64 must not become 128.
    require_numbers_preserved: bool = True


class ProviderModelsConfig(_Base):
    """Model identifier per provider and role. Names only - never keys."""

    openai_embedding: str = "text-embedding-3-small"
    #: text-embedding-004 was retired and 404s on :embedContent.
    gemini_embedding: str = "gemini-embedding-001"
    cohere_embedding: str = "embed-english-v3.0"
    local_embedding: str = "sentence-transformers/all-MiniLM-L6-v2"

    groq_generation: str = "llama-3.3-70b-versatile"
    gemini_generation: str = "gemini-2.0-flash"
    openai_generation: str = "gpt-4o-mini"
    #: Which OpenRouter models carry the `:free` tag rotates month to month, so
    #: this is config, and $OPENROUTER_MODEL overrides it at runtime.
    openrouter_generation: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    local_generation: str = "llama3.2:3b"


class ProvidersConfig(_Base):
    """Which model backs each stage, and what happens when it cannot.

    This is the block to point at when asked "what is actually running this
    right now": the chains below are walked in the order written, top to
    bottom, and ``GET /health`` reports which entry is live.
    """

    #: Stage 3. openai | gemini | cohere | local
    embedding_provider: str = "openai"
    #: Tried in order after ``embedding_provider`` is skipped or fails.
    embedding_fallback: list[str] = Field(default_factory=lambda: ["gemini", "local"])
    #: Stage 6 and the eval harness, tried in order.
    generation_providers: list[str] = Field(
        default_factory=lambda: ["groq", "gemini", "openai", "local"]
    )
    #: Default per-request HTTP timeout; callers may pass a tighter one.
    timeout_s: float = 30.0
    #: How long a failed provider sits out before the chain retries it.
    cooldown_s: float = 60.0
    local_host: str = "http://localhost:11434"

    @field_validator("embedding_provider")
    @classmethod
    def _known_embedding(cls, value: str) -> str:
        from .providers.embedding import EMBEDDING_PROVIDERS

        if value not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"unknown embedding provider {value!r}; "
                f"choose from {sorted(EMBEDDING_PROVIDERS)}"
            )
        return value

    @field_validator("embedding_fallback")
    @classmethod
    def _known_embedding_chain(cls, value: list[str]) -> list[str]:
        from .providers.embedding import EMBEDDING_PROVIDERS

        unknown = [name for name in value if name not in EMBEDDING_PROVIDERS]
        if unknown:
            raise ValueError(
                f"unknown embedding provider(s) {unknown}; "
                f"choose from {sorted(EMBEDDING_PROVIDERS)}"
            )
        return value

    @field_validator("generation_providers")
    @classmethod
    def _known_generation(cls, value: list[str]) -> list[str]:
        from .providers.generation import GENERATION_PROVIDERS

        unknown = [name for name in value if name not in GENERATION_PROVIDERS]
        if unknown:
            raise ValueError(
                f"unknown generation provider(s) {unknown}; "
                f"choose from {sorted(GENERATION_PROVIDERS)}"
            )
        if not value:
            raise ValueError("generation_providers must name at least one provider")
        return value

    models: ProviderModelsConfig = Field(default_factory=ProviderModelsConfig)


class ReconstructionConfig(_Base):
    drop_markers: bool = True
    cluster_markers: bool = True


class EvaluationConfig(_Base):
    """How the harness measures. *Which* model answers lives in `providers`.

    The harness asks its questions through the same generation chain stage 6
    uses, so there is one answer to "what model produced these numbers" and the
    report records whichever provider actually served the run.
    """

    #: Ollama defaults to a 2048-token window and SILENTLY TRUNCATES beyond it.
    #: Every "original context" measurement would be a lie without this. Only
    #: applies to the local provider; hosted models have their own windows.
    num_ctx: int = 16384
    max_answer_tokens: int = 120
    timeout_s: float = 300.0
    judge_provider: str = "auto"
    judge_model_openai: str = "gpt-4o-mini"
    judge_model_local: str = "qwen2.5:7b-instruct"
    repeats: int = 1
    #: Model whose published pricing is used for the cost columns.
    pricing_model: str = "gpt-4o-mini"


class ModelPrice(_Base):
    input_per_1m: float
    output_per_1m: float


class PricingConfig(_Base):
    model_config = ConfigDict(extra="allow")

    default_model: str = "gpt-4o-mini"

    def price_for(self, model: str) -> ModelPrice:
        raw = getattr(self, "__pydantic_extra__", {}) or {}
        entry = raw.get(model)
        if entry is None:
            raise KeyError(
                f"No published pricing configured for {model!r}. "
                f"Add it under `pricing:` in config.yaml."
            )
        return ModelPrice.model_validate(entry)

    def known_models(self) -> list[str]:
        raw = getattr(self, "__pydantic_extra__", {}) or {}
        return sorted(raw.keys())


class Config(_Base):
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    redundancy: RedundancyConfig = Field(default_factory=RedundancyConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    abstractive: AbstractiveConfig = Field(default_factory=AbstractiveConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with a nested dict merged in (used by the API layer)."""
        merged = _deep_merge(self.model_dump(), overrides)
        return Config.model_validate(merged)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a config file. Falls back to built-in defaults."""
    resolved = Path(path or os.environ.get("CCE_CONFIG") or DEFAULT_CONFIG_PATH)
    if not resolved.exists():
        return Config()
    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


@lru_cache(maxsize=8)
def _cached(path: str | None) -> Config:
    return load_config(path)


def get_config(path: str | Path | None = None) -> Config:
    """Process-wide cached config accessor."""
    return _cached(str(path) if path else None)
