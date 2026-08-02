"""Model provider layer: one abstraction for every model call in the project.

Public surface:

    from engine.providers import build_embedding_chain, build_generation_chain

    embeddings = build_embedding_chain(cfg)       # stage 3
    generation = build_generation_chain(cfg)      # stage 6 + eval harness

Both return a chain that walks the providers named in ``config.yaml``, skips
any whose key is not configured, falls through on failure, and raises
:class:`NoProviderAvailable` only when nothing in the chain can run.

``CCE_OFFLINE=1`` collapses every chain to its local providers. That is what
the test suite runs under - a unit test must never depend on a network, a key,
or someone's free-tier quota - and it is also the honest "no keys, no wifi"
path for anyone running this from a clean checkout.
"""

from __future__ import annotations

import logging
import os

from .base import (
    Attempt,
    CallStats,
    EmbeddingProvider,
    GenerationProvider,
    NoProviderAvailable,
    Provider,
    ProviderError,
    ProviderFailed,
    ProviderUnavailable,
)  # noqa: F401  (re-exported)
from .chain import EmbeddingChain, GenerationChain
from .embedding import EMBEDDING_PROVIDERS, resolve_device
from .generation import GENERATION_PROVIDERS
from .keys import KEY_REGISTRY, MissingKey, configured_keys, has, load_dotenv, redact

log = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProvider",
    "GenerationProvider",
    "EmbeddingChain",
    "GenerationChain",
    "NoProviderAvailable",
    "ProviderError",
    "ProviderFailed",
    "ProviderUnavailable",
    "MissingKey",
    "Attempt",
    "CallStats",
    "Provider",
    "build_embedding_chain",
    "build_generation_chain",
    "provider_status",
    "provider_catalogue",
    "selection_presets",
    "mode_readiness",
    "offline_mode",
    "normalise_mode",
    "InvalidMode",
    "InvalidProvider",
    "MODES",
    "DEFAULT_MODE",
    "LOCAL_PROVIDERS",
    "configured_keys",
    "resolve_device",
    "redact",
    "KEY_REGISTRY",
]

EMBEDDING_NAMES = tuple(EMBEDDING_PROVIDERS)
GENERATION_NAMES = tuple(GENERATION_PROVIDERS)

#: Providers that run on this machine. Everything else needs a network and a
#: key. This one set defines what "local" and "cloud" mean everywhere.
LOCAL_PROVIDERS = frozenset({"local"})

#: Per-request execution mode.
#:
#: ``local``  - local providers only. No outbound API call is made even when
#:              every key is configured. Asserted by a test that fails the
#:              build if any HTTP call escapes.
#: ``cloud``  - configured cloud providers only, in their configured order.
#:              **Local is deliberately not a fallback here.** A user who asked
#:              for cloud and silently got a 3B local model would draw exactly
#:              the wrong conclusion from the latency and accuracy numbers, so
#:              an exhausted cloud chain is a loud error instead.
#: ``auto``   - the full configured chain, cloud first, local last. The
#:              pre-existing behaviour, and still the default.
MODES = ("local", "cloud", "auto")
DEFAULT_MODE = "auto"


class InvalidMode(ValueError):
    """Raised for a mode outside :data:`MODES`."""


def normalise_mode(mode: str | None) -> str:
    resolved = (mode or DEFAULT_MODE).strip().lower()
    if resolved not in MODES:
        raise InvalidMode(f"unknown mode {mode!r}; choose from {list(MODES)}")
    return resolved


def offline_mode() -> bool:
    """True when CCE_OFFLINE forces local-only providers."""
    return os.environ.get("CCE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def _filter_for_mode(names: list[str], mode: str, role: str) -> list[str]:
    """Reduce a configured chain to the providers the mode permits.

    Raises rather than returning an empty chain: "cloud requested but this
    build has no cloud provider configured" is a configuration answer the
    caller needs immediately, not a mystery failure three stages later.
    """
    if mode == "local":
        return [n for n in names if n in LOCAL_PROVIDERS] or ["local"]

    if mode == "cloud":
        if offline_mode():
            raise NoProviderAvailable(
                role,
                [
                    Attempt(
                        "cloud",
                        ok=False,
                        skipped=True,
                        reason="CCE_OFFLINE is set, which forbids outbound calls",
                    )
                ],
            )
        cloud = [n for n in names if n not in LOCAL_PROVIDERS]
        if not cloud:
            raise NoProviderAvailable(
                role,
                [
                    Attempt(
                        "cloud",
                        ok=False,
                        skipped=True,
                        reason=(
                            f"mode='cloud' but the configured {role} chain "
                            f"({', '.join(names) or 'empty'}) contains no cloud provider"
                        ),
                    )
                ],
            )
        return cloud

    # auto: the configured chain, unchanged.
    return list(names)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def _embedding_provider(name: str, cfg) -> EmbeddingProvider:
    settings = cfg.providers
    models = settings.models
    if name == "openai":
        return EMBEDDING_PROVIDERS[name](models.openai_embedding, settings.timeout_s)
    if name == "gemini":
        return EMBEDDING_PROVIDERS[name](models.gemini_embedding, settings.timeout_s)
    if name == "cohere":
        return EMBEDDING_PROVIDERS[name](models.cohere_embedding, settings.timeout_s)
    if name == "local":
        return EMBEDDING_PROVIDERS[name](
            models.local_embedding,
            cfg.redundancy.device,
            cfg.redundancy.batch_size,
        )
    raise ValueError(f"unknown embedding provider {name!r}; known: {EMBEDDING_NAMES}")


def _generation_provider(name: str, cfg, timeout_s: float, num_ctx: int | None):
    models = cfg.providers.models
    host = cfg.providers.local_host
    if name == "groq":
        return GENERATION_PROVIDERS[name](models.groq_generation, timeout_s)
    if name == "gemini":
        return GENERATION_PROVIDERS[name](models.gemini_generation, timeout_s)
    if name == "openrouter":
        return GENERATION_PROVIDERS[name](models.openrouter_generation, timeout_s)
    if name == "openai":
        return GENERATION_PROVIDERS[name](models.openai_generation, timeout_s)
    if name == "local":
        return GENERATION_PROVIDERS[name](
            models.local_generation, host, max(timeout_s, 45.0), num_ctx
        )
    raise ValueError(f"unknown generation provider {name!r}; known: {GENERATION_NAMES}")


def _ordered(primary: str | None, rest: list[str]) -> list[str]:
    """Primary first, then the fallbacks, with duplicates removed in place."""
    seen: list[str] = []
    for name in ([primary] if primary else []) + list(rest):
        if name and name not in seen:
            seen.append(name)
    return seen


def _pinned(name: str, registry: dict, role: str) -> list[str]:
    """Resolve an explicit provider choice to a single-entry chain.

    A pinned provider gets **no fallback**, for the same reason cloud mode does
    not fall back to local: someone who picked Groq and silently received
    Gemini would attribute Gemini's latency and answers to Groq. If the pinned
    provider cannot run, that is an error worth seeing.
    """
    if name not in registry:
        raise InvalidProvider(
            f"unknown {role} provider {name!r}; choose from {sorted(registry)}"
        )
    if name not in LOCAL_PROVIDERS and offline_mode():
        # A pin is an explicit choice, but CCE_OFFLINE is a hard guarantee that
        # no outbound call happens - and a guarantee an explicit choice can
        # quietly override is not a guarantee. Refuse instead.
        raise NoProviderAvailable(
            role,
            [
                Attempt(
                    name,
                    ok=False,
                    skipped=True,
                    reason="CCE_OFFLINE is set, which forbids outbound calls",
                )
            ],
        )
    return [name]


class InvalidProvider(ValueError):
    """Raised for a provider name this build does not know."""


def build_embedding_chain(
    cfg,
    *,
    force_local: bool | None = None,
    mode: str | None = None,
    pin: str | None = None,
) -> EmbeddingChain:
    """Stage 3's embedding chain: configured provider, then its fallbacks.

    ``mode`` narrows the chain per request - see :data:`MODES`. It must be the
    same mode the generation chain is built with, or a single run would embed
    in the cloud and generate locally, and the latency and cost numbers it
    reports would describe a configuration nobody chose.
    """
    settings = cfg.providers
    resolved = normalise_mode(mode)

    if pin:
        names = _pinned(pin, EMBEDDING_PROVIDERS, "embedding")
    elif force_local is True or (force_local is None and resolved == "local"):
        names = ["local"]
    elif force_local is None and resolved == "auto" and offline_mode():
        names = ["local"]
    else:
        names = _filter_for_mode(
            _ordered(settings.embedding_provider, settings.embedding_fallback),
            resolved,
            "embedding",
        )

    providers = [_embedding_provider(name, cfg) for name in names]
    return EmbeddingChain(providers, cooldown_s=settings.cooldown_s)


def build_generation_chain(
    cfg,
    *,
    timeout_s: float | None = None,
    num_ctx: int | None = None,
    force_local: bool | None = None,
    names: list[str] | None = None,
    mode: str | None = None,
    pin: str | None = None,
) -> GenerationChain:
    """Stage 6 / eval harness generation chain.

    ``timeout_s`` and ``num_ctx`` differ by caller - stage 6 wants a hard 10s
    ceiling so a live demo stays responsive, while the harness needs minutes and
    a 16k local context window - so they are arguments rather than config.
    """
    settings = cfg.providers
    budget = settings.timeout_s if timeout_s is None else timeout_s
    execution = normalise_mode(mode)
    configured = names if names is not None else list(settings.generation_providers)

    if pin:
        resolved = _pinned(pin, GENERATION_PROVIDERS, "generation")
    elif force_local is True or (force_local is None and execution == "local"):
        resolved = ["local"]
    elif force_local is None and execution == "auto" and offline_mode():
        resolved = ["local"]
    else:
        resolved = _filter_for_mode(configured, execution, "generation")

    providers = [
        _generation_provider(name, cfg, budget, num_ctx) for name in resolved
    ]
    return GenerationChain(providers, cooldown_s=settings.cooldown_s)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def provider_status(cfg) -> dict:
    """What ``GET /health`` reports: which providers can actually run.

    Never contains a key, and makes no model calls - it only asks each provider
    whether it is configured, so it is cheap enough to poll from the dashboard.
    The local providers are the exception and do a 3s reachability probe, which
    is the only way "is Ollama running" can be answered truthfully.
    """
    load_dotenv()
    embedding = build_embedding_chain(cfg)
    generation = build_generation_chain(cfg)
    keys_present = configured_keys()
    return {
        "offline_mode": offline_mode(),
        "keys_configured": keys_present,
        "keys_missing": sorted(n for n, ok in keys_present.items() if not ok),
        "embedding": embedding.describe(),
        "generation": generation.describe(),
        "any_generation_available": generation.available()[0],
        "any_embedding_available": embedding.available()[0],
        # Independent readiness per mode, so the dashboard can disable a toggle
        # option rather than let someone pick one that is going to fail.
        "modes": mode_readiness(cfg),
    }


def provider_catalogue(cfg) -> dict:
    """Every provider/model the build can offer, per role.

    This is what a model picker is built from: one entry per selectable option,
    each carrying the concrete model id and whether it can actually run right
    now. A caller should never have to hardcode a model name to offer it, and
    should never offer one that will 503.

    Configured-ness is key presence only - no model call is made, so this is
    cheap enough to serve on every dashboard poll. The local providers are the
    exception and pay a 3 s reachability probe, because "is Ollama running" has
    no cheaper truthful answer.
    """
    load_dotenv()
    catalogue: dict[str, list[dict]] = {}

    for role, registry, builder in (
        ("embedding", EMBEDDING_PROVIDERS, _embedding_provider),
        ("generation", GENERATION_PROVIDERS, _generation_provider),
    ):
        entries = []
        for name in registry:
            if role == "embedding":
                provider = builder(name, cfg)
            else:
                provider = builder(name, cfg, cfg.providers.timeout_s, None)
            ok, reason = provider.configured()
            entries.append({
                "provider": name,
                "model": provider.model,
                "role": role,
                "local": name in LOCAL_PROVIDERS,
                "configured": ok,
                "reason": reason or None,
                "requires_key": provider.env_key,
            })
        catalogue[role] = entries

    return {
        "embedding": catalogue["embedding"],
        "generation": catalogue["generation"],
        # The chain each role falls back through when nothing is pinned.
        "default_chains": {
            "embedding": _ordered(
                cfg.providers.embedding_provider, cfg.providers.embedding_fallback
            ),
            "generation": list(cfg.providers.generation_providers),
        },
    }


#: Preference order for the embedding side of a preset, when the chosen
#: generation vendor does not offer embeddings of its own. Ordered by measured
#: cost/latency on this corpus: Cohere answers in ~400 ms, Gemini ~600 ms,
#: OpenAI is paid, local is free but slowest to load.
_EMBEDDING_PREFERENCE = ("cohere", "gemini", "openai", "local")


def selection_presets(cfg) -> list[dict]:
    """One selectable entry per model, each resolving to a *coherent* pair.

    A single "which model?" dropdown is the control users actually want, but
    the honest answer is that a compression uses two models, and no vendor
    supplies both for every choice - Groq has no embeddings API, Cohere has no
    chat API. Rather than push that on the UI, each preset here names the model
    a user is choosing (the one that rewrites text, which is what "which model"
    colloquially means) and states the embedding provider it resolves to.

    The pairing lives here rather than in the frontend so it is one testable
    rule instead of a guess repeated per client, and so the detail line the UI
    shows is generated from the same resolution the request will actually use.
    """
    catalogue = provider_catalogue(cfg)
    embedders = {e["provider"]: e for e in catalogue["embedding"]}
    generators = {g["provider"]: g for g in catalogue["generation"]}

    def best_embedder(prefer: str | None) -> str | None:
        """Same vendor if it embeds and is usable, else the first that is."""
        order = ([prefer] if prefer else []) + list(_EMBEDDING_PREFERENCE)
        for name in order:
            entry = embedders.get(name)
            if entry and entry["configured"]:
                return name
        return None

    presets: list[dict] = [
        {
            "id": "auto",
            "label": "Auto — best available",
            "detail": (
                f"embeddings {' → '.join(catalogue['default_chains']['embedding'])}"
                f" · rewriting {' → '.join(catalogue['default_chains']['generation'])}"
            ),
            "mode": "auto",
            "embedding_provider": None,
            "generation_provider": None,
            "local": False,
            "available": True,
            "reason": None,
        }
    ]

    local_embed = embedders.get("local")
    local_generate = generators.get("local")
    presets.append({
        "id": "local",
        "label": f"Local — {local_generate['model'] if local_generate else 'ollama'}",
        "detail": (
            f"embeddings {local_embed['model'].split('/')[-1] if local_embed else 'MiniLM'}"
            f" · runs on this machine, no API call"
        ),
        "mode": "local",
        "embedding_provider": "local",
        "generation_provider": "local",
        "local": True,
        "available": bool(local_generate and local_generate["configured"]),
        "reason": None if (local_generate and local_generate["configured"])
                  else (local_generate or {}).get("reason"),
    })

    for name, entry in generators.items():
        if name == "local":
            continue
        embedding = best_embedder(name if name in embedders else None)
        presets.append({
            "id": name,
            "label": f"{name} — {entry['model']}",
            "detail": (
                f"embeddings {embedders[embedding]['model']} ({embedding})"
                if embedding
                else "no embedding provider available"
            ),
            "mode": "auto",
            "embedding_provider": embedding,
            "generation_provider": name,
            "local": False,
            "available": entry["configured"],
            "reason": entry["reason"],
        })

    return presets


def mode_readiness(cfg) -> dict:
    """Can each execution mode actually run, reported independently?

    Deliberately separate from the chain descriptions above: the toggle needs
    to know "is local usable" and "is cloud usable" as two unrelated questions.
    Ollama being down must not make cloud look broken, and having no keys must
    not make local look broken.

    Local does a real 3 s reachability probe - the only truthful way to answer
    "is Ollama running" - while cloud only checks key presence, because
    verifying a key costs quota on every poll.
    """
    out: dict = {}
    for mode in ("local", "cloud"):
        entry: dict = {"mode": mode}
        try:
            embedding = build_embedding_chain(cfg, mode=mode)
            generation = build_generation_chain(cfg, mode=mode)
        except NoProviderAvailable as exc:
            out[mode] = {
                **entry,
                "ready": False,
                "reason": str(exc),
                "embedding": {"ready": False, "providers": []},
                "generation": {"ready": False, "providers": []},
            }
            continue

        embedding_ok, embedding_why = embedding.available()
        generation_ok, generation_why = generation.available()
        # Embeddings degrade gracefully (stage 3 falls back to exact-hash
        # dedup), so a mode is usable as long as it can generate. Reported
        # separately regardless so the UI can explain a partial state.
        entry.update(
            ready=generation_ok,
            reason=None if generation_ok else generation_why,
            embedding={
                "ready": embedding_ok,
                "reason": None if embedding_ok else embedding_why,
                "providers": embedding.names,
                "active": embedding.active_provider_name(),
            },
            generation={
                "ready": generation_ok,
                "reason": None if generation_ok else generation_why,
                "providers": generation.names,
                "active": generation.active_provider_name(),
            },
        )
        out[mode] = entry
    return out
