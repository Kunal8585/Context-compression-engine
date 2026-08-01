"""Span primitives shared by all chunker backends.

Every backend works in terms of ``Span`` (a character range over the original
source string) and only converts to :class:`~engine.types.Chunk` at the end.
Two consequences that matter:

* merging/splitting is always a slice of the *original* text, so chunk text is
  byte-faithful to the input - no whitespace is invented or lost;
* line numbers and character offsets stay exact, which is what makes the
  drop-markers in stage 7 auditable.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(slots=True)
class Span:
    """A character range over the source document."""

    start: int
    end: int
    kind: str
    symbol: str | None = None
    meta: dict | None = None

    def text(self, source: str) -> str:
        return source[self.start : self.end]

    @property
    def length(self) -> int:
        return self.end - self.start


class LineIndex:
    """char offset -> 1-indexed line number."""

    def __init__(self, source: str) -> None:
        starts = [0]
        for i, ch in enumerate(source):
            if ch == "\n":
                starts.append(i + 1)
        self._starts = starts
        self._len = len(source)

    def line_of(self, char_offset: int) -> int:
        offset = max(0, min(char_offset, self._len))
        return bisect_right(self._starts, offset)

    def span_lines(self, span: Span) -> tuple[int, int]:
        start_line = self.line_of(span.start)
        # `end` is exclusive; step back one char so a span ending on a newline
        # is not attributed to the following line.
        end_line = self.line_of(max(span.start, span.end - 1))
        return start_line, end_line


class OffsetMap:
    """tree-sitter byte offset -> character offset.

    ASCII sources (the overwhelming majority of code) take an identity fast
    path and cost nothing.
    """

    def __init__(self, source: str) -> None:
        self.is_ascii = source.isascii()
        self._byte_starts: list[int] = []
        if not self.is_ascii:
            offset = 0
            for ch in source:
                self._byte_starts.append(offset)
                offset += len(ch.encode("utf-8"))
            self._byte_starts.append(offset)

    def to_char(self, byte_offset: int) -> int:
        if self.is_ascii:
            return byte_offset
        return bisect_left(self._byte_starts, byte_offset)


# --------------------------------------------------------------------------
# text splitting helpers
# --------------------------------------------------------------------------

# Abbreviations that end in a period but do not end a sentence. Python's `re`
# has no variable-width lookbehind, so we split first and re-join fragments
# whose tail is one of these.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg",
    "ie", "fig", "no", "inc", "ltd", "co", "corp", "dept", "est", "al",
    "approx", "min", "max", "avg", "ref", "vol", "cf", "resp",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
# Captures the final dotted token, including internal periods, so that "i.e."
# yields "i.e" (-> "ie") rather than just the trailing "e".
_TRAILING_TOKEN = re.compile(r"([A-Za-z](?:\.?[A-Za-z])*)\.[\"')\]]*\s*$")
_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n")


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _TRAILING_TOKEN.search(fragment)
    if not match:
        return False
    word = match.group(1).replace(".", "")
    if word.lower() in _ABBREVIATIONS:
        return True
    # Single capital letter -> an initial ("J. Smith"), not a sentence end.
    return len(word) == 1 and word.isupper()


def sentence_spans(source: str, start: int, end: int) -> list[Span]:
    """Split ``source[start:end]`` into sentence spans."""
    text = source[start:end]
    if not text.strip():
        return []

    cuts: list[int] = []
    for match in _SENTENCE_BOUNDARY.finditer(text):
        cuts.append(match.end())

    pieces: list[tuple[int, int]] = []
    prev = 0
    for cut in cuts:
        pieces.append((prev, cut))
        prev = cut
    pieces.append((prev, len(text)))

    # Re-join fragments that were split on an abbreviation or an initial.
    merged: list[tuple[int, int]] = []
    for piece in pieces:
        if merged and _ends_with_abbreviation(text[merged[-1][0] : merged[-1][1]]):
            merged[-1] = (merged[-1][0], piece[1])
        else:
            merged.append(piece)

    spans = []
    for lo, hi in merged:
        if text[lo:hi].strip():
            spans.append(Span(start + lo, start + hi, "sentence"))
    return spans


def paragraph_spans(source: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Split a region into blank-line-separated paragraph spans."""
    end = len(source) if end is None else end
    text = source[start:end]
    spans: list[Span] = []
    cursor = 0
    for match in _PARAGRAPH_BOUNDARY.finditer(text):
        piece = text[cursor : match.start()]
        if piece.strip():
            spans.append(Span(start + cursor, start + match.start(), "paragraph"))
        cursor = match.end()
    if text[cursor:].strip():
        spans.append(Span(start + cursor, end, "paragraph"))
    return spans


def line_spans(source: str, start: int, end: int) -> list[Span]:
    """Split a region into one span per line (used to break up huge code units)."""
    spans: list[Span] = []
    cursor = start
    while cursor < end:
        newline = source.find("\n", cursor, end)
        stop = end if newline == -1 else newline + 1
        if source[cursor:stop].strip():
            spans.append(Span(cursor, stop, "other"))
        cursor = stop
    return spans


def pack_spans(
    spans: list[Span],
    token_counts: list[int],
    target_tokens: int,
    max_tokens: int,
) -> list[list[Span]]:
    """Greedily group consecutive spans into batches of ~``target_tokens``."""
    groups: list[list[Span]] = []
    current: list[Span] = []
    current_tokens = 0
    for span, tokens in zip(spans, token_counts):
        if current and current_tokens + tokens > target_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(span)
        current_tokens += tokens
        if current_tokens >= max_tokens:
            groups.append(current)
            current, current_tokens = [], 0
    if current:
        groups.append(current)
    return groups


def merge_spans(spans: list[Span], kind: str | None = None) -> Span:
    """Collapse consecutive spans into the enclosing span."""
    if not spans:
        raise ValueError("cannot merge an empty span list")
    first, last = spans[0], spans[-1]
    meta: dict = {}
    for span in spans:
        if span.meta:
            meta.update(span.meta)
    return Span(
        start=first.start,
        end=last.end,
        kind=kind or first.kind,
        symbol=first.symbol,
        meta=meta or None,
    )
