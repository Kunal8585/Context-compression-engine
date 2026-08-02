"""Embedding providers for stage 3.

Four backends, one interface. What differs between them and why it matters:

============  ==========================  =====  ==========  ==================
provider      model                       dim    batch cap   cost
============  ==========================  =====  ==========  ==================
openai        text-embedding-3-small       1536  2048 items  $0.02 / 1M tokens
gemini        text-embedding-004            768  100 items   free tier
cohere        embed-english-v3.0           1024  96 items    free (trial key)
local         all-MiniLM-L6-v2              384  n/a         free, no network
============  ==========================  =====  ==========  ==================

The batch caps are the vendors' real documented limits, not round numbers: they
are the reason this layer batches at all. Stage 3 embeds every surviving chunk,
which on the sample log is several hundred texts - one request per chunk would
exhaust a free tier's per-minute request budget on a single compression while
also being ~200x slower than the batched call.

``local`` is kept deliberately. It is the zero-cost tail of every fallback
chain, the only backend that works with no network, and the control condition
when comparing whether a paid embedding actually clusters better than MiniLM.
"""

from __future__ import annotations

import logging

from . import keys
from ._http import dig, post_json
from .base import EmbeddingProvider, ProviderFailed, ProviderUnavailable

log = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """text-embedding-3-small: cheapest high-quality option, already L2-normal."""

    name = "openai"
    env_key = "OPENAI_API_KEY"
    endpoint = "https://api.openai.com/v1/embeddings"
    dimension = 1536
    #: The documented item cap is 2048, but the request also has a 300k-token
    #: ceiling. Chunks top out at 400 tokens, so 256 stays comfortably inside
    #: both while still collapsing a 400-chunk document into two requests.
    batch_limit = 256

    def __init__(self, model: str = "text-embedding-3-small", timeout_s: float = 30.0):
        super().__init__()
        self.model = model
        self.timeout_s = timeout_s

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = post_json(
            self.name,
            self.endpoint,
            {"model": self.model, "input": texts},
            {
                "Authorization": f"Bearer {keys.get(self.env_key)}",
                "Content-Type": "application/json",
            },
            self.timeout_s,
        )
        rows = dig(self.name, body, "data")
        # The API documents order-preservation but also returns an explicit
        # index; sorting on it costs nothing and removes the assumption.
        ordered = sorted(rows, key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in ordered]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """text-embedding-004 via batchEmbedContents. Free tier, 768 dimensions."""

    name = "gemini"
    env_key = "GOOGLE_API_KEY"
    base = "https://generativelanguage.googleapis.com/v1beta"
    dimension = 768
    #: Google's documented ceiling for one batchEmbedContents call.
    batch_limit = 100

    def __init__(self, model: str = "text-embedding-004", timeout_s: float = 30.0):
        super().__init__()
        self.model = model
        self.timeout_s = timeout_s

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        body = post_json(
            self.name,
            f"{self.base}/{model_path}:batchEmbedContents",
            {
                "requests": [
                    {"model": model_path, "content": {"parts": [{"text": text}]}}
                    for text in texts
                ]
            },
            # Header, not ?key= - a requests exception carries the URL.
            {
                "x-goog-api-key": keys.get(self.env_key),
                "Content-Type": "application/json",
            },
            self.timeout_s,
        )
        rows = dig(self.name, body, "embeddings")
        return [row["values"] for row in rows]


class CohereEmbeddingProvider(EmbeddingProvider):
    """embed-english-v3.0. Free trial keys are rate-limited but generous."""

    name = "cohere"
    env_key = "COHERE_API_KEY"
    endpoint = "https://api.cohere.com/v2/embed"
    dimension = 1024
    #: Cohere rejects a request with more than 96 texts outright.
    batch_limit = 96

    def __init__(self, model: str = "embed-english-v3.0", timeout_s: float = 30.0):
        super().__init__()
        self.model = model
        self.timeout_s = timeout_s

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = post_json(
            self.name,
            self.endpoint,
            {
                "model": self.model,
                "texts": texts,
                # Stage 3 compares chunks against each other rather than a
                # query against documents, so both sides must use the same
                # asymmetric input type or the similarities are meaningless.
                "input_type": "search_document",
                "embedding_types": ["float"],
                "truncate": "END",
            },
            {
                "Authorization": f"Bearer {keys.get(self.env_key)}",
                "Content-Type": "application/json",
            },
            self.timeout_s,
        )
        return dig(self.name, body, "embeddings", "float")


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers all-MiniLM-L6-v2, on MPS/CUDA/CPU.

    The original engine backend, kept as a first-class provider rather than
    deleted: it is the zero-cost fallback that keeps the pipeline working with
    no key and no network, and the control condition for "is a paid embedding
    actually better here". Model loading is lazy, cached process-wide, and
    prefers the local HF cache before touching the network.
    """

    name = "local"
    env_key = None
    #: Not an API limit - just the encode() minibatch size passed to torch.
    batch_limit = 256

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        encode_batch_size: int = 64,
    ):
        super().__init__()
        self.model = model
        self.encode_batch_size = encode_batch_size
        self._requested_device = device
        self._device: str | None = None
        self._backend = None
        self._loaded = False
        self._error: str | None = None

    # -- device / loading --------------------------------------------------
    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._requested_device)
        return self._device

    def _load(self):
        if self._loaded:
            return self._backend
        self._loaded = True
        cache_key = (self.model, self.device)
        if cache_key in _MODEL_CACHE:
            self._backend = _MODEL_CACHE[cache_key]
            return self._backend
        try:
            self._backend = _load_offline_first(self.model, self.device)
            _MODEL_CACHE[cache_key] = self._backend
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            log.warning("local embedding model %s unavailable: %s", self.model, exc)
            self._backend = None
        return self._backend

    def configured(self) -> tuple[bool, str]:
        """Loadable counts as configured; a missing wheel means "skip me"."""
        if self._load() is None:
            return False, f"sentence-transformers unavailable ({self._error})"
        return True, ""

    def describe(self) -> dict:
        return {**super().describe(), "device": self.device}

    # -- encoding ----------------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        backend = self._load()
        if backend is None:
            raise ProviderUnavailable(self.name, self._error or "model not loaded")
        try:
            vectors = backend.encode(
                texts,
                batch_size=self.encode_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            # An MPS kernel fault or OOM mid-encode must not take the pipeline
            # down; fall to CPU once, then let the chain move on.
            if self.device != "cpu":
                log.warning("encode failed on %s (%s); retrying on cpu", self.device, exc)
                try:
                    backend.to("cpu")
                    self._device = "cpu"
                    vectors = backend.encode(
                        texts,
                        batch_size=self.encode_batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    raise ProviderFailed(self.name, retry_exc) from None
            else:
                raise ProviderFailed(self.name, exc) from None
        if not self.dimension and len(vectors):
            self.dimension = int(vectors.shape[1])
        return [row.tolist() for row in vectors]


# --------------------------------------------------------------------------
# local model loading helpers
# --------------------------------------------------------------------------
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device. ``auto`` prefers Apple Silicon's MPS backend."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch missing entirely
        pass
    return "cpu"


def _load_offline_first(model_name: str, device: str):
    """Load from the local HF cache first, only then reach for the network.

    Without this, every load round-trips to huggingface.co to revalidate the
    snapshot. On bad conference wifi that turns a 1-second model load into a
    multi-second hang - or a failure - in the middle of a live demo.
    """
    import os

    from sentence_transformers import SentenceTransformer

    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        log.info("model %s not in the local cache (%s); downloading", model_name, exc)
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
    return SentenceTransformer(model_name, device=device)


#: Name used in config.yaml -> constructor.
EMBEDDING_PROVIDERS = {
    "openai": OpenAIEmbeddingProvider,
    "gemini": GeminiEmbeddingProvider,
    "cohere": CohereEmbeddingProvider,
    "local": LocalEmbeddingProvider,
}
