"""Stage 2 - chunking.

Public entry points::

    from engine.chunker import chunk_document, chunk_file, select_chunker

Backend selection is automatic: a supported source extension routes to
tree-sitter, log-shaped content routes to the record chunker, everything else
to the prose chunker. Pass ``kind=`` to force one.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ChunkingConfig, get_config
from ..tokenizer import Tokenizer, get_tokenizer
from ..types import Chunk
from .base import Chunker
from .code import CodeChunker, KNOWN_CODE_EXTENSIONS, language_for
from .logs import LogChunker, looks_like_log
from .text import TextChunker

__all__ = [
    "Chunker",
    "CodeChunker",
    "LogChunker",
    "TextChunker",
    "chunk_document",
    "chunk_file",
    "detect_kind",
    "select_chunker",
]

AUTO = "auto"
CODE = "code"
TEXT = "text"
LOG = "log"


def detect_kind(name: str, source: str) -> str:
    """Classify a document as code / log / text."""
    spec = language_for(name)
    if spec is not None and CodeChunker.available_for(name):
        return CODE
    if looks_like_log(source):
        return LOG
    if Path(name).suffix.lower() in KNOWN_CODE_EXTENSIONS:
        # Recognised as code but no grammar installed: prose chunking on blank
        # lines is the honest fallback, and it degrades rather than crashing.
        return TEXT
    return TEXT


def select_chunker(
    name: str,
    source: str,
    kind: str = AUTO,
    cfg: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> Chunker:
    """Return the chunker backend appropriate for this document."""
    cfg = cfg or get_config().chunking
    tokenizer = tokenizer or get_tokenizer()

    resolved = detect_kind(name, source) if kind == AUTO else kind
    if resolved == CODE:
        spec = language_for(name)
        if spec is not None and CodeChunker.available_for(name):
            return CodeChunker(spec, cfg, tokenizer)
        return TextChunker(cfg, tokenizer)
    if resolved == LOG:
        return LogChunker(cfg, tokenizer)
    if resolved == TEXT:
        return TextChunker(cfg, tokenizer)
    raise ValueError(f"unknown chunker kind: {kind!r}")


def chunk_document(
    source: str,
    name: str = "input",
    kind: str = AUTO,
    cfg: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Chunk an in-memory document."""
    chunker = select_chunker(source=source, name=name, kind=kind, cfg=cfg, tokenizer=tokenizer)
    return chunker.chunk(source, name)


def chunk_file(
    path: str | Path,
    kind: str = AUTO,
    cfg: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Chunk a file from disk, labelling chunks with its name."""
    resolved = Path(path)
    source = resolved.read_text(encoding="utf-8", errors="replace")
    return chunk_document(source, resolved.name, kind=kind, cfg=cfg, tokenizer=tokenizer)
