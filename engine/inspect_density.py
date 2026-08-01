"""Density inspector - see why each chunk scored what it scored.

    python -m engine.inspect_density data/sample_corpus/logs/checkout_service.log
    python -m engine.inspect_density <file> --bottom 10
    python -m engine.inspect_density <file> --grep "pool wait"

Prints the per-signal breakdown behind every score, so a ranking can be argued
with rather than taken on faith. ``--grep`` reports where specific content lands
in the ranking, which is how the retention tests were calibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunker import select_chunker
from .config import get_config
from .density import DensityScorer
from .redundancy import RedundancyDetector
from .tokenizer import get_tokenizer


def _row(rank: int, chunk, total: int) -> str:
    parts = chunk.density_parts
    def cell(name: str) -> str:
        value = parts.get(name)
        return "  -  " if value is None else f"{value:5.2f}"

    percentile = 100.0 * (1.0 - rank / total)
    return (
        f"{rank:>4} {percentile:>5.1f}% {chunk.density:6.3f} | "
        f"{cell('entropy')} {cell('tfidf')} {cell('entities')} "
        f"{cell('novelty')} {cell('structure')} {cell('frequency')} | "
        f"{chunk.token_count:>5} x{chunk.duplicate_count:<4} "
        f"{chunk.preview(58)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inspect_density", description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--kind", default="auto", choices=["auto", "code", "text", "log"])
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--bottom", type=int, default=8)
    parser.add_argument("--grep", action="append", default=[],
                        help="report where matching chunks rank (repeatable)")
    parser.add_argument("--no-redundancy", action="store_true",
                        help="score raw chunks without collapsing duplicates first")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    cfg = get_config(args.config)
    source = path.read_text(encoding="utf-8", errors="replace")
    tokenizer = get_tokenizer(cfg.tokenizer)
    chunker = select_chunker(path.name, source, args.kind, cfg.chunking, tokenizer)
    chunks = chunker.chunk(source, path.name)

    embeddings = None
    if args.no_redundancy:
        survivors = chunks
    else:
        redundancy = RedundancyDetector(cfg, tokenizer).run(chunks)
        survivors, embeddings = redundancy.chunks, redundancy.embeddings

    result = DensityScorer(cfg, tokenizer).run(survivors, embeddings)

    if args.json:
        print(json.dumps({
            **result.to_dict(),
            "chunks": [c.to_dict(include_text=False) for c in result.ranked()],
        }, indent=2))
        return 0

    metrics = result.metrics
    ranked = result.ranked()
    total = len(ranked)

    print(f"file        : {path}")
    print(f"chunks      : {total:,} scored in {metrics.duration_ms:.0f} ms")
    print(f"entities    : {metrics.details['entity_backend']}")
    print(f"weights     : " + "  ".join(
        f"{k}={v:.3f}" for k, v in result.weights.items()))
    if result.unavailable:
        print(f"unavailable : {', '.join(result.unavailable)} "
              f"(weight redistributed over the rest)")
    print(f"scores      : min={metrics.details['score_min']} "
          f"p50={metrics.details['score_p50']} max={metrics.details['score_max']}")

    header = (f"\n{'rank':>4} {'pct':>6} {'score':>6} | "
              f"{'entr':>5} {'tfidf':>5} {'ents':>5} {'novel':>5} {'struc':>5} {'freq':>5} | "
              f"{'tok':>5} {'dup':<5} preview")
    print(header)
    print("-" * 150)
    for rank, chunk in enumerate(ranked[: args.top], start=1):
        print(_row(rank, chunk, total))

    if args.bottom and total > args.top:
        print(f"\n{'... lowest scoring (first to be dropped by stage 5) ':.<150}")
        start = max(args.top, total - args.bottom)
        for rank, chunk in enumerate(ranked[start:], start=start + 1):
            print(_row(rank, chunk, total))

    for needle in args.grep:
        matches = [
            (rank, chunk)
            for rank, chunk in enumerate(ranked, start=1)
            if needle.lower() in chunk.text.lower()
        ]
        print(f"\ngrep {needle!r}: {len(matches)} match(es)")
        for rank, chunk in matches[:5]:
            percentile = 100.0 * (1.0 - rank / total)
            print(f"   rank {rank}/{total} (top {percentile:.1f}%) "
                  f"score={chunk.density:.3f}  {chunk.preview(70)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
