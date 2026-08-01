"""Log chunker: one chunk per log *record*, not per line.

Logs are the highest-value input for this engine because they are pathologically
redundant - the same INFO line can appear thousands of times with only a
timestamp and a request id changing.

Two things this chunker does that a naive line splitter cannot:

1. **Record grouping.** A Java stack trace or a Python traceback is one event
   spanning 30 lines. Splitting it per line would destroy the only chunk in the
   file a judge actually cares about.
2. **Template extraction.** Each record gets a normalised template with
   volatile fields (timestamps, ids, numbers, paths, hex, quoted values)
   replaced by placeholders. Stage 3 hashes that template, which collapses
   thousands of near-identical records for free - no embedding model needed.
"""

from __future__ import annotations

import re

from ..types import ChunkKind
from ._spans import Span
from .base import Chunker

# A line that starts a new record: ISO/clock timestamp, syslog date, bracketed
# timestamp, bare log level, or a standalone JSON object.
_RECORD_START = re.compile(
    r"""^\s*(?:
        \[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}          # 2024-01-05 10:11:12
      | \[?\d{2}:\d{2}:\d{2}[.,]?\d*\]?                    # 10:11:12.345
      | [A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}        # Jan  5 10:11:12
      | \[?(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL|SEVERE)\b
      | \{".*\}\s*$                                        # JSON line
      | <\d+>                                              # syslog priority
    )""",
    re.VERBOSE,
)

_CONTINUATION = re.compile(
    r"^(?:\s+|Traceback|\s*at\s|\s*Caused by:|\s*\.\.\.\s*\d+\s+more|\s*File\s\")"
)

# Once a stack trace opens, every following line belongs to it until a line
# that clearly starts a new record. Without this the *last* line of a Python
# traceback - the exception type and message, i.e. the one line anyone actually
# reads - is orphaned into a record of its own.
_TRACE_OPEN = re.compile(
    r"^(?:Traceback \(most recent call last\)|\s*at\s+\S+|\s*Caused by:|\s*File\s\")"
)

_LEVEL_RE = re.compile(
    r"\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL|SEVERE)\b"
)

# Volatile field patterns, applied in order, to build a stable template.
_TEMPLATE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<TIME>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "<HASH>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    # NOTE: there is deliberately no blanket path or quoted-string rule here.
    # Collapsing `/v1/checkout` to `<PATH>` would merge two genuinely different
    # endpoints into one cluster and misattribute their counts. Volatile path
    # segments (`/v1/cart/ORD-427039`) are normalised by the <ID> rule below,
    # which leaves the static prefix - the part carrying the information -
    # intact. Over-collapsing costs reasoning retention; under-collapsing only
    # costs a little compression ratio.
    # Identifiers: a word, a separator, then >=4 hex/digit characters
    # (ORD-427039, usr_1234, ch_5f3a2b). Deliberately requires 4+ so that
    # `eu-west-1`, `v2.31.0` and `checkout-api` survive intact - collapsing a
    # region or a version into a placeholder would merge records that a judge
    # can legitimately ask us to tell apart.
    (re.compile(r"\b[A-Za-z][A-Za-z]*[-_][0-9a-fA-F]{4,}\b"), "<ID>"),
    # Standalone numbers only. The lookbehind stops us eating the `1` in
    # `eu-west-1` or the `31` in `v2.31.0`.
    (
        re.compile(r"(?<![-\w.])\d+(?:\.\d+)?(?:ms|s|kb|mb|gb|%)?\b", re.IGNORECASE),
        "<NUM>",
    ),
]


class LogChunker(Chunker):
    """Groups log lines into records and annotates each with a template."""

    name = "log"

    def _spans(self, source: str) -> list[Span]:
        starts = _record_line_starts(source)
        if not starts:
            # No recognisable record boundaries - treat each line as a record.
            starts = [i for i, _ in _iter_lines(source)]

        spans: list[Span] = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(source)
            text = source[start:end]
            if not text.strip():
                continue
            template = log_template(text)
            spans.append(
                Span(
                    start=start,
                    end=end,
                    kind=ChunkKind.LOG_RECORD,
                    symbol=template[:120],
                    meta={
                        "template": template,
                        "level": _level_of(text),
                        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
                        "multiline": "\n" in text.strip(),
                    },
                )
            )
        return spans


def _iter_lines(source: str):
    """Yield (offset, line_with_newline) pairs."""
    cursor = 0
    length = len(source)
    while cursor < length:
        newline = source.find("\n", cursor)
        stop = length if newline == -1 else newline + 1
        yield cursor, source[cursor:stop]
        cursor = stop


def _record_line_starts(source: str) -> list[int]:
    starts: list[int] = []
    in_trace = False
    for offset, line in _iter_lines(source):
        if not line.strip():
            continue
        if _RECORD_START.match(line):
            starts.append(offset)
            in_trace = False
        elif not starts:
            # Leading preamble before the first recognisable record.
            starts.append(offset)
        elif in_trace or _CONTINUATION.match(line):
            pass  # part of the record already in progress
        else:
            # Unrecognised, non-indented line: a record in an unknown format.
            starts.append(offset)
        if _TRACE_OPEN.match(line):
            in_trace = True
    return starts


def _level_of(text: str) -> str | None:
    match = _LEVEL_RE.search(text.split("\n", 1)[0])
    return match.group(1).upper() if match else None


def log_template(text: str) -> str:
    """Normalise a record into a stable template for exact-hash dedup."""
    template = text.strip()
    for pattern, placeholder in _TEMPLATE_RULES:
        template = pattern.sub(placeholder, template)
    return re.sub(r"\s+", " ", template).strip()


def looks_like_log(source: str, sample_lines: int = 200, threshold: float = 0.4) -> bool:
    """Heuristic used by the chunker dispatcher."""
    matched = 0
    considered = 0
    for _, line in _iter_lines(source):
        if not line.strip():
            continue
        considered += 1
        if _RECORD_START.match(line):
            matched += 1
        if considered >= sample_lines:
            break
    if considered < 3:
        return False
    return matched / considered >= threshold
