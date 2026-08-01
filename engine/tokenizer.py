"""Token counting.

Every compression-ratio number in this project is a *token* ratio, not a
character ratio, so this module is load-bearing for the headline metric.

We use tiktoken's ``cl100k_base`` (GPT-4o / GPT-4 / GPT-3.5). If the BPE file
cannot be loaded - e.g. a first run with no network - we degrade to a word-based
estimate and set ``is_exact = False``. Any report produced with an inexact
tokenizer carries that flag, so a judge is never shown an estimate that looks
like a measurement.
"""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache

from .config import TokenizerConfig

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    def __init__(self, cfg: TokenizerConfig | None = None) -> None:
        self.cfg = cfg or TokenizerConfig()
        self._enc = None
        self._lock = threading.Lock()
        self.is_exact = False
        self.backend = "estimate"
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(self.cfg.encoding)
            self.is_exact = True
            self.backend = f"tiktoken:{self.cfg.encoding}"
        except Exception as exc:  # pragma: no cover - offline path
            log.warning(
                "tiktoken unavailable (%s); falling back to word-based token "
                "estimates. Reported counts are approximate.",
                exc,
            )

    # -- counting ----------------------------------------------------------
    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            # tiktoken's Rust core is not documented as re-entrant per handle;
            # the lock costs nothing at our call volume.
            with self._lock:
                return len(self._enc.encode(text, disallowed_special=()))
        return self._estimate(text)

    def count_many(self, texts: list[str]) -> list[int]:
        if self._enc is not None:
            with self._lock:
                return [
                    len(t)
                    for t in self._enc.encode_batch(texts, disallowed_special=())
                ]
        return [self._estimate(t) for t in texts]

    def _estimate(self, text: str) -> int:
        words = len(_WORD_RE.findall(text))
        return max(1, round(words * self.cfg.fallback_tokens_per_word))

    def token_ids(self, text: str) -> list[int]:
        """Token ids for entropy calculations.

        Falls back to hashed words when tiktoken is unavailable. Entropy only
        needs a consistent identity per token, so hashes serve as well as ids.
        """
        if not text:
            return []
        if self._enc is not None:
            with self._lock:
                return self._enc.encode(text, disallowed_special=())
        return [hash(word) for word in _WORD_RE.findall(text)]

    # -- truncation --------------------------------------------------------
    def truncate(self, text: str, max_tokens: int) -> str:
        """Cut ``text`` down to at most ``max_tokens`` tokens."""
        if max_tokens <= 0:
            return ""
        if self._enc is not None:
            with self._lock:
                ids = self._enc.encode(text, disallowed_special=())
                if len(ids) <= max_tokens:
                    return text
                return self._enc.decode(ids[:max_tokens])
        if self.count(text) <= max_tokens:
            return text
        approx_chars = int(max_tokens / max(self.cfg.fallback_tokens_per_word, 0.1) * 5)
        return text[:approx_chars]

    def describe(self) -> dict[str, object]:
        return {"backend": self.backend, "exact": self.is_exact}


@lru_cache(maxsize=4)
def _cached_tokenizer(encoding: str, fallback: float) -> Tokenizer:
    return Tokenizer(TokenizerConfig(encoding=encoding, fallback_tokens_per_word=fallback))


def get_tokenizer(cfg: TokenizerConfig | None = None) -> Tokenizer:
    """Shared tokenizer instance (loading the BPE table is not free)."""
    cfg = cfg or TokenizerConfig()
    return _cached_tokenizer(cfg.encoding, cfg.fallback_tokens_per_word)
