"""Chunker interface and the shared post-processing pipeline.

Backends implement one method - ``_spans`` - which returns semantic spans over
the source. Everything that is common to all backends (size normalisation,
merging runt chunks, token counting, stable ids) lives here so the backends
stay small and testable.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from ..config import ChunkingConfig
from ..tokenizer import Tokenizer, get_tokenizer
from ..types import Chunk, ChunkKind
from ._spans import (
    LineIndex,
    Span,
    line_spans,
    merge_spans,
    pack_spans,
    sentence_spans,
)

log = logging.getLogger(__name__)

_ID_SAFE = re.compile(r"[^A-Za-z0-9_.\-/]+")


class Chunker(ABC):
    """Splits a document into semantically coherent chunks."""

    name: str = "base"

    def __init__(
        self,
        cfg: ChunkingConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.cfg = cfg or ChunkingConfig()
        self.tokenizer = tokenizer or get_tokenizer()

    # -- backend contract --------------------------------------------------
    @abstractmethod
    def _spans(self, source: str) -> list[Span]:
        """Return semantic spans in document order. Must not overlap."""

    # -- public API --------------------------------------------------------
    def chunk(self, source: str, name: str = "input") -> list[Chunk]:
        """Split ``source`` into chunks. ``name`` labels the origin document."""
        if not source or not source.strip():
            return []

        spans = self._spans(source)
        spans = [s for s in spans if source[s.start : s.end].strip()]
        spans.sort(key=lambda s: s.start)

        spans = self._enforce_max_size(source, spans)
        spans = self._merge_runts(source, spans)
        return self._to_chunks(source, spans, name)

    # -- shared post-processing -------------------------------------------
    def _enforce_max_size(self, source: str, spans: list[Span]) -> list[Span]:
        """Split any span above ``max_chunk_tokens`` into packed sub-spans.

        A backend gets first refusal via ``_split_oversized`` (the code chunker
        uses it to break a class into methods). Whatever is still too big is
        split by sentence, or by line for code-like content.
        """
        out: list[Span] = []
        for span in spans:
            tokens = self.tokenizer.count(source[span.start : span.end])
            if tokens <= self.cfg.max_chunk_tokens:
                out.append(span)
                continue

            pieces = self._split_oversized(source, span)
            if pieces and len(pieces) > 1:
                # Recurse once: a class split into methods may still contain a
                # single oversized method.
                for piece in pieces:
                    piece_tokens = self.tokenizer.count(source[piece.start : piece.end])
                    if piece_tokens <= self.cfg.max_chunk_tokens:
                        out.append(piece)
                    else:
                        out.extend(self._pack_fallback(source, piece))
            else:
                out.extend(self._pack_fallback(source, span))
        return out

    def _split_oversized(self, source: str, span: Span) -> list[Span]:
        """Backend hook: structure-aware split of a too-large span."""
        return []

    def _pack_fallback(self, source: str, span: Span) -> list[Span]:
        """Last-resort split: pack sentences (prose) or lines (code)."""
        is_codeish = span.kind in ChunkKind.CODE_KINDS or span.kind == ChunkKind.CODE_BLOCK
        units = (
            line_spans(source, span.start, span.end)
            if is_codeish
            else sentence_spans(source, span.start, span.end)
        )
        if len(units) <= 1:
            return [span]

        counts = self.tokenizer.count_many([source[u.start : u.end] for u in units])
        groups = pack_spans(
            units,
            counts,
            target_tokens=self.cfg.target_chunk_tokens,
            max_tokens=self.cfg.max_chunk_tokens,
        )
        pieces: list[Span] = []
        for index, group in enumerate(groups):
            merged = merge_spans(group, kind=span.kind)
            merged.symbol = span.symbol
            merged.meta = dict(span.meta or {})
            merged.meta["split_part"] = index + 1
            merged.meta["split_of"] = len(groups)
            pieces.append(merged)
        return pieces

    #: Kinds that must attach to what *follows* them. A `## Root Cause` heading
    #: belongs to the section it introduces; folding it into the paragraph above
    #: would file it under the previous section and mislabel both.
    forward_only_kinds: frozenset[str] = frozenset({ChunkKind.HEADING})

    def _merge_runts(self, source: str, spans: list[Span]) -> list[Span]:
        """Fold sub-minimum spans into an adjacent one.

        Only kinds listed in ``chunking.mergeable_kinds`` participate. Atomic
        units (functions, log records, code blocks) are never merged - keeping
        them whole is what makes dedup and density scoring meaningful.
        """
        if not self.cfg.merge_small_chunks or len(spans) < 2:
            return spans
        return self._fold_runts(source, self._attach_forward_only(source, spans))

    def _attach_forward_only(self, source: str, spans: list[Span]) -> list[Span]:
        """Pull a runt heading down into the chunk(s) it labels.

        It keeps absorbing forwards until the result clears ``min_chunk_tokens``
        - otherwise a `## Timeline` + `All times UTC.` pair is still a runt and
        the next pass would drag the whole thing back into the section above.
        """
        mergeable = set(self.cfg.mergeable_kinds)
        out: list[Span] = []
        index = 0
        while index < len(spans):
            span = spans[index]
            is_runt_anchor = (
                span.kind in self.forward_only_kinds
                and span.kind in mergeable
                and self.tokenizer.count(source[span.start : span.end])
                < self.cfg.min_chunk_tokens
            )
            if not is_runt_anchor:
                out.append(span)
                index += 1
                continue

            combined = span
            parts = (span.meta or {}).get("merged_parts", 1)
            cursor = index + 1
            while cursor < len(spans):
                following = spans[cursor]
                if following.kind not in mergeable:
                    break
                if (
                    self.tokenizer.count(source[combined.start : following.end])
                    > self.cfg.max_chunk_tokens
                ):
                    break
                merged = merge_spans([combined, following], kind=following.kind)
                # The section name becomes the label for the content below it.
                merged.symbol = combined.symbol or following.symbol
                parts += (following.meta or {}).get("merged_parts", 1)
                merged.meta = {
                    **(combined.meta or {}),
                    **(following.meta or {}),
                    "merged_parts": parts,
                    "forward_anchor": True,
                }
                combined = merged
                cursor += 1
                if (
                    self.tokenizer.count(source[combined.start : combined.end])
                    >= self.cfg.min_chunk_tokens
                ):
                    break

            out.append(combined)
            index = max(cursor, index + 1)
        return out

    def _fold_runts(self, source: str, spans: list[Span]) -> list[Span]:
        mergeable = set(self.cfg.mergeable_kinds)
        minimum = self.cfg.min_chunk_tokens
        out: list[Span] = []
        out_tokens: list[int] = []

        for span in spans:
            tokens = self.tokenizer.count(source[span.start : span.end])

            # Merge when either side is a runt, both kinds opt in, and the
            # result still respects the hard ceiling.
            if (
                out
                and (out_tokens[-1] < minimum or tokens < minimum)
                and span.kind in mergeable
                and out[-1].kind in mergeable
                # A heading never merges backwards - see forward_only_kinds -
                # and neither does a chunk that a heading was folded into.
                and span.kind not in self.forward_only_kinds
                and not (span.meta or {}).get("forward_anchor")
                and out_tokens[-1] + tokens <= self.cfg.max_chunk_tokens
            ):
                previous = out.pop()
                out_tokens.pop()
                # The dominant span decides the kind of the result.
                kind = previous.kind if previous.length >= span.length else span.kind
                combined = merge_spans([previous, span], kind=kind)
                combined.symbol = previous.symbol or span.symbol
                combined.meta = {
                    **(combined.meta or {}),
                    "merged_parts": (previous.meta or {}).get("merged_parts", 1)
                    + (span.meta or {}).get("merged_parts", 1),
                }
                out.append(combined)
                # Recount rather than sum: the merged slice also swallows the
                # whitespace that separated the two spans.
                out_tokens.append(
                    self.tokenizer.count(source[combined.start : combined.end])
                )
                continue

            out.append(span)
            out_tokens.append(tokens)
        return out

    def _to_chunks(self, source: str, spans: list[Span], name: str) -> list[Chunk]:
        texts = [source[s.start : s.end] for s in spans]
        counts = self.tokenizer.count_many(texts)
        index = LineIndex(source)
        safe_name = _ID_SAFE.sub("_", name) or "input"

        chunks: list[Chunk] = []
        for order, (span, text, tokens) in enumerate(zip(spans, texts, counts)):
            start_line, end_line = index.span_lines(span)
            symbol_part = f":{span.symbol}" if span.symbol else ""
            chunks.append(
                Chunk(
                    text=text,
                    kind=span.kind,
                    source=name,
                    order=order,
                    start_line=start_line,
                    end_line=end_line,
                    start_char=span.start,
                    end_char=span.end,
                    token_count=tokens,
                    id=f"{safe_name}#{order:04d}:{span.kind}{symbol_part}",
                    symbol=span.symbol,
                    metadata={"chunker": self.name, **(span.meta or {})},
                )
            )
        return chunks
