"""Chunk inspector - verify chunk quality on any file.

    python -m engine.inspect_chunks data/sample_corpus/code/auth_service.py
    python -m engine.inspect_chunks path/to/file.log --full
    python -m engine.inspect_chunks path/to/doc.md --kind text --json

Prints one row per chunk with its kind, line range, token count and symbol, so
chunk boundaries can be eyeballed against the source. ``--verify`` additionally
asserts that the chunks tile the source without gaps or overlaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunker import chunk_document, detect_kind, select_chunker
from .config import get_config
from .tokenizer import get_tokenizer
from .types import Chunk


def _table(chunks: list[Chunk], full: bool) -> str:
    if not chunks:
        return "(no chunks)"
    lines = []
    header = f"{'#':>4}  {'kind':<13} {'lines':>11} {'tok':>6}  {'symbol':<28} preview"
    lines.append(header)
    lines.append("-" * min(len(header) + 60, 160))
    for chunk in chunks:
        span = f"{chunk.start_line}-{chunk.end_line}"
        symbol = (chunk.symbol or "")[:28]
        preview = chunk.preview(90 if full else 60)
        lines.append(
            f"{chunk.order:>4}  {chunk.kind:<13} {span:>11} {chunk.token_count:>6}  "
            f"{symbol:<28} {preview}"
        )
    return "\n".join(lines)


def _verify(source: str, chunks: list[Chunk]) -> list[str]:
    """Check that chunks tile the source: ordered, non-overlapping, complete."""
    problems: list[str] = []
    cursor = 0
    for chunk in chunks:
        if chunk.start_char < cursor:
            problems.append(
                f"chunk {chunk.order} overlaps the previous one "
                f"(starts at {chunk.start_char}, previous ended at {cursor})"
            )
        dropped = source[cursor : chunk.start_char]
        if dropped.strip():
            problems.append(
                f"non-whitespace gap before chunk {chunk.order}: {dropped.strip()[:60]!r}"
            )
        if source[chunk.start_char : chunk.end_char] != chunk.text:
            problems.append(f"chunk {chunk.order} text does not match its char span")
        cursor = max(cursor, chunk.end_char)
    trailing = source[cursor:]
    if trailing.strip():
        problems.append(f"non-whitespace tail dropped: {trailing.strip()[:60]!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inspect_chunks", description=__doc__)
    parser.add_argument("path", help="file to chunk")
    parser.add_argument(
        "--kind",
        default="auto",
        choices=["auto", "code", "text", "log"],
        help="force a chunker backend (default: auto-detect)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--full", action="store_true", help="wider previews")
    parser.add_argument("--show", type=int, metavar="N", help="print chunk N in full")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="assert chunks tile the source with no gaps or overlaps",
    )
    parser.add_argument("--config", help="path to an alternate config.yaml")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    source = path.read_text(encoding="utf-8", errors="replace")
    cfg = get_config(args.config)
    tokenizer = get_tokenizer(cfg.tokenizer)
    chunker = select_chunker(path.name, source, args.kind, cfg.chunking, tokenizer)
    chunks = chunker.chunk(source, path.name)

    if args.show is not None:
        match = next((c for c in chunks if c.order == args.show), None)
        if match is None:
            print(f"error: no chunk #{args.show}", file=sys.stderr)
            return 2
        print(f"--- chunk {match.order} [{match.kind}] "
              f"lines {match.start_line}-{match.end_line} "
              f"({match.token_count} tokens) ---")
        print(match.text)
        return 0

    if args.json:
        print(json.dumps([c.to_dict() for c in chunks], indent=2))
        return 0

    total_tokens = sum(c.token_count for c in chunks)
    source_tokens = tokenizer.count(source)
    detected = detect_kind(path.name, source)
    kinds: dict[str, int] = {}
    for chunk in chunks:
        kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1

    print(f"file        : {path}")
    print(f"backend     : {chunker.name}  (auto-detected: {detected}, requested: {args.kind})")
    print(f"tokenizer   : {tokenizer.backend}"
          f"{'' if tokenizer.is_exact else '  [ESTIMATE - counts are approximate]'}")
    print(f"source      : {len(source):,} chars, {source_tokens:,} tokens, "
          f"{source.count(chr(10)) + 1:,} lines")
    print(f"chunks      : {len(chunks)}  ({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))})")
    if chunks:
        sizes = sorted(c.token_count for c in chunks)
        print(f"tokens/chunk: min={sizes[0]} p50={sizes[len(sizes) // 2]} max={sizes[-1]} "
              f"total={total_tokens:,} (coverage {100 * total_tokens / max(source_tokens, 1):.1f}%)")
    print()
    print(_table(chunks, args.full))

    if args.verify:
        problems = _verify(source, chunks)
        print()
        if problems:
            print(f"VERIFY FAILED - {len(problems)} problem(s):")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("VERIFY OK - chunks tile the source with no gaps, overlaps or text drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
