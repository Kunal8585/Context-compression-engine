"""Text chunker: paragraph-level with markdown awareness.

Paragraphs are the right unit for prose: they are the smallest span that
usually survives removal without stranding a dangling pronoun, and they are
what the redundancy detector can meaningfully compare.

Structure we respect:
* fenced code blocks stay atomic (splitting one produces uncompilable noise);
* headings become their own span and carry a ``section`` annotation forward,
  which the density scorer uses as a structural prior;
* list runs stay together;
* paragraphs above ``max_chunk_tokens`` fall back to sentence packing.
"""

from __future__ import annotations

import re

from ..types import ChunkKind
from ._spans import Span, paragraph_spans
from .base import Chunker

_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)", re.MULTILINE)
_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_SETEXT_RE = re.compile(r"^[ \t]{0,3}(=+|-{2,})[ \t]*$")
_LIST_LINE_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\S")


class TextChunker(Chunker):
    """Paragraph/sentence chunker for prose, markdown and unstructured docs."""

    name = "text"

    def _spans(self, source: str) -> list[Span]:
        spans: list[Span] = []
        section: str | None = None

        for region_start, region_end, is_fence in _split_fences(source):
            if is_fence:
                spans.append(
                    Span(
                        start=region_start,
                        end=region_end,
                        kind=ChunkKind.CODE_BLOCK,
                        symbol=_fence_language(source, region_start),
                        meta={"section": section} if section else None,
                    )
                )
                continue

            for span in paragraph_spans(source, region_start, region_end):
                text = source[span.start : span.end]
                heading = _HEADING_RE.match(text.strip())
                if heading:
                    section = heading.group(2).strip()
                    span.kind = ChunkKind.HEADING
                    span.symbol = section
                    # Deliberately NOT "level": that key means log severity
                    # elsewhere, and the collision fed an int to the density
                    # scorer's severity lookup.
                    span.meta = {"heading_level": len(heading.group(1))}
                    spans.append(span)
                    continue

                if _is_list_block(text):
                    span.kind = ChunkKind.LIST
                else:
                    span.kind = ChunkKind.PARAGRAPH
                if section:
                    span.meta = {**(span.meta or {}), "section": section}
                spans.append(span)

        return spans


def _split_fences(source: str) -> list[tuple[int, int, bool]]:
    """Partition the document into (start, end, is_fenced_block) regions."""
    fences = [m.start() for m in _FENCE_RE.finditer(source)]
    if not fences:
        return [(0, len(source), False)]

    regions: list[tuple[int, int, bool]] = []
    cursor = 0
    index = 0
    while index < len(fences):
        open_at = fences[index]
        if open_at > cursor:
            regions.append((cursor, open_at, False))
        if index + 1 < len(fences):
            close_at = fences[index + 1]
            line_end = source.find("\n", close_at)
            block_end = len(source) if line_end == -1 else line_end + 1
            regions.append((open_at, block_end, True))
            cursor = block_end
            index += 2
        else:
            # Unterminated fence: treat the remainder as a code block.
            regions.append((open_at, len(source), True))
            cursor = len(source)
            index += 1
    if cursor < len(source):
        regions.append((cursor, len(source), False))
    return regions


def _fence_language(source: str, start: int) -> str | None:
    line_end = source.find("\n", start)
    first_line = source[start : line_end if line_end != -1 else len(source)]
    label = first_line.strip().lstrip("`~").strip()
    return label or None


def _is_list_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    matches = sum(1 for line in lines if _LIST_LINE_RE.match(line))
    return matches >= max(1, len(lines) // 2)
