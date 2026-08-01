"""Sentence embeddings with hard graceful-degradation guarantees.

Wraps sentence-transformers (all-MiniLM-L6-v2, 384-d) behind a small interface
that *never raises* into the pipeline. If the model cannot be loaded - no
network on first run, no local cache, MPS misbehaving - ``available`` is False
and callers fall back to the exact-hash path. A live demo must not die because
a model download timed out.

Vectors are L2-normalised on the way out, so cosine similarity is a dot product
and clustering never has to re-normalise.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import RedundancyConfig

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _load_offline_first(model_name: str, device: str):
    """Load from the local HF cache first, only then reach for the network.

    Without this, every load round-trips to huggingface.co to revalidate the
    snapshot. On a bad conference wifi that turns a 1-second model load into a
    multi-second hang - or a failure - in the middle of a live demo. Once the
    model has been pulled by ``scripts/setup.sh`` the offline path always wins.
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


@dataclass
class EncodeStats:
    count: int = 0
    duration_ms: float = 0.0
    device: str = "cpu"
    model: str = ""
    dimension: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "encoded": self.count,
            "encode_ms": round(self.duration_ms, 2),
            "device": self.device,
            "model": self.model,
            "dimension": self.dimension,
            **self.extra,
        }


class EmbeddingModel:
    """Lazily-loaded embedding backend that degrades instead of raising."""

    def __init__(self, cfg: RedundancyConfig | None = None) -> None:
        self.cfg = cfg or RedundancyConfig()
        self.device = resolve_device(self.cfg.device)
        self._model = None
        self._load_error: str | None = None
        self._loaded = False
        self.stats = EncodeStats(device=self.device, model=self.cfg.model)

    # -- loading -----------------------------------------------------------
    def _load(self) -> object | None:
        if self._loaded:
            return self._model
        self._loaded = True
        key = (self.cfg.model, self.device)
        if key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[key]
            return self._model
        try:
            started = time.perf_counter()
            model = _load_offline_first(self.cfg.model, self.device)
            log.info(
                "loaded %s on %s in %.0f ms",
                self.cfg.model,
                self.device,
                (time.perf_counter() - started) * 1000,
            )
            _MODEL_CACHE[key] = model
            self._model = model
        except Exception as exc:
            self._load_error = str(exc)
            log.warning(
                "embedding model %s unavailable (%s); redundancy detection will "
                "fall back to exact-hash dedup only",
                self.cfg.model,
                exc,
            )
            self._model = None
        return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    @property
    def load_error(self) -> str | None:
        self._load()
        return self._load_error

    # -- encoding ----------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Return L2-normalised embeddings, or None if the backend is down."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        if model is None:
            return None
        started = time.perf_counter()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            # A runtime failure mid-encode (an MPS kernel fault, OOM) must not
            # take the pipeline down either. Retry once on CPU, then give up.
            log.warning("encode failed on %s (%s); retrying on cpu", self.device, exc)
            if self.device != "cpu":
                try:
                    model.to("cpu")
                    self.device = "cpu"
                    self.stats.device = "cpu"
                    vectors = model.encode(
                        texts,
                        batch_size=self.cfg.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                except Exception as retry_exc:  # pragma: no cover - defensive
                    log.error("cpu retry also failed: %s", retry_exc)
                    self._load_error = str(retry_exc)
                    return None
            else:
                self._load_error = str(exc)
                return None

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.stats.count += len(texts)
        self.stats.duration_ms += (time.perf_counter() - started) * 1000
        self.stats.dimension = int(vectors.shape[1]) if vectors.size else 0
        return vectors

    def warmup(self) -> float:
        """Load the model and run one encode, returning the cost in ms.

        Loading MiniLM and initialising the MPS backend costs ~9 s the first
        time in a process; encoding 400 chunks after that costs ~0.5 s. The API
        and dashboard call this at startup so a judge's first request measures
        compression, not a lazy import.
        """
        started = time.perf_counter()
        self.encode(["warmup"])
        return (time.perf_counter() - started) * 1000

    def describe(self) -> dict:
        return {
            "model": self.cfg.model,
            "device": self.device,
            "available": self.available,
            "error": self._load_error,
        }
