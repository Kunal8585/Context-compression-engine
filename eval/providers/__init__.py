"""Cross-provider comparison helpers for the eval harness.

This package used to hold a second, parallel implementation of "how to call a
model" - its own base class, its own OpenAI/Groq/Gemini clients, its own retry
loop. The provider migration deleted all of it. What remains is a thin adapter
over :mod:`engine.providers`, so there is exactly one place in the project that
knows how to talk to a vendor.

The one thing this layer still adds is the ``--providers`` report's framing: it
wants each provider *individually and unchained*, deliberately, because the
question that report answers is "how does accuracy retention differ between
models" - and a silent fallback would make two rows secretly the same model.
"""

from __future__ import annotations

from engine.config import Config
from engine.providers import GenerationProvider, ProviderError
from engine.providers.generation import GENERATION_PROVIDERS

__all__ = ["GenerationProvider", "ProviderError", "available_providers"]


def available_providers(cfg: Config) -> dict[str, GenerationProvider]:
    """Every known generation provider, standalone and unchained.

    Keyed by the name used in ``--providers groq,gemini,local``. Whether each
    one is usable is the caller's question to ask via ``configured()``; a
    provider with no key is reported as skipped in the comparison rather than
    quietly replaced by a different model.
    """
    from engine.providers import _generation_provider

    return {
        name: _generation_provider(
            name, cfg, cfg.providers.timeout_s, cfg.evaluation.num_ctx
        )
        for name in GENERATION_PROVIDERS
    }
