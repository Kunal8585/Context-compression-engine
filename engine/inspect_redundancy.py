"""Redundancy inspector - audit exactly what stage 3 collapsed and why.

    python -m engine.inspect_redundancy data/sample_corpus/logs/checkout_service.log
    python -m engine.inspect_redundancy <file> --threshold 0.95 --show-members
    python -m engine.inspect_redundancy <file> --no-embeddings   # exact+structural only

Every collapse is listed with its method (exact / structural / embedding) and
similarity, so a claimed compression number can be checked line by line rather
than taken on trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunker import chunk_document, select_chunker
from .config import get_config
from .redundancy import RedundancyDetector
from .tokenizer import get_tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inspect_redundancy", description=__doc__)
    parser.add_argument("path", help="file to compress")
    parser.add_argument("--kind", default="auto", choices=["auto", "code", "text", "log"])
    parser.add_argument("--threshold", type=float, help="override similarity_threshold")
    parser.add_argument("--no-embeddings", action="store_true", help="skip the MiniLM pass")
    parser.add_argument("--no-structural", action="store_true", help="skip structural dedup")
    parser.add_argument("--show-members", action="store_true", help="list absorbed chunks")
    parser.add_argument("--top", type=int, default=15, help="how many clusters to print")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--config", help="path to an alternate config.yaml")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    overrides: dict = {"redundancy": {}}
    if args.threshold is not None:
        overrides["redundancy"]["similarity_threshold"] = args.threshold
    if args.no_embeddings:
        overrides["redundancy"]["enabled"] = False
    if args.no_structural:
        overrides["redundancy"]["structural"] = {"enabled": False}
    cfg = get_config(args.config).with_overrides(overrides)

    source = path.read_text(encoding="utf-8", errors="replace")
    tokenizer = get_tokenizer(cfg.tokenizer)
    chunker = select_chunker(path.name, source, args.kind, cfg.chunking, tokenizer)
    chunks = chunker.chunk(source, path.name)

    result = RedundancyDetector(cfg, tokenizer).run(chunks)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    metrics = result.metrics
    details = metrics.details
    print(f"file        : {path}")
    print(f"backend     : {chunker.name}")
    print(f"tokenizer   : {tokenizer.backend}"
          f"{'' if tokenizer.is_exact else '  [ESTIMATE]'}")
    print()
    print(f"chunks      : {metrics.chunks_in:,} -> {metrics.chunks_out:,}")
    print(f"tokens      : {metrics.tokens_in:,} -> {metrics.tokens_out:,} "
          f"({metrics.reduction_pct:.1f}% removed by stage 3 alone)")
    print(f"time        : {metrics.duration_ms:.0f} ms")
    print()
    print(f"collapsed   : exact={details['exact_collapsed']:,}  "
          f"structural={details['structural_collapsed']:,}  "
          f"embedding={details['embedding_collapsed']:,}")
    print(f"clusters    : {details['clusters_formed']:,} "
          f"(largest {details['largest_cluster_size']}x)")
    print(f"threshold   : cosine >= {details['similarity_threshold']}")
    embedding = details.get("embedding", {})
    if embedding.get("available") is False:
        print(f"embeddings  : UNAVAILABLE - {embedding.get('error')}")
        print("              (exact + structural passes still ran)")
    else:
        print(f"embeddings  : {embedding.get('model')} on {embedding.get('device')}, "
              f"{embedding.get('encoded', 0):,} encoded in "
              f"{embedding.get('encode_ms', 0):.0f} ms")
    if metrics.note:
        print(f"note        : {metrics.note}")

    clusters = sorted(result.clusters.values(), key=lambda c: c.size, reverse=True)
    if not clusters:
        print("\nno clusters formed - nothing was collapsed")
        return 0

    print(f"\n{'clusters (what was collapsed, and by which method)':-<100}")
    by_id = {c.id: c for c in result.chunks}
    for cluster in clusters[: args.top]:
        representative = next(
            (c for c in result.chunks if c.id == cluster.representative_id), None
        )
        label = (
            representative.symbol or representative.preview(60)
            if representative
            else cluster.representative_id
        )
        print(
            f"\n  x{cluster.size:<5} [{cluster.method:^10}] sim={cluster.mean_similarity:.3f}  "
            f"saved {cluster.absorbed_tokens:,} tok"
        )
        print(f"         kept: {label}")
        if args.show_members or cluster.method in {"structural", "embedding"}:
            shown = cluster.absorbed_symbols[:6]
            for symbol in shown:
                print(f"         drop: {symbol}")
            if len(cluster.absorbed_symbols) > len(shown):
                print(f"         ... and {len(cluster.absorbed_symbols) - len(shown)} more")

    if len(clusters) > args.top:
        print(f"\n  ... and {len(clusters) - args.top} more clusters")

    total_saved = sum(c.absorbed_tokens for c in clusters)
    print(f"\ntotal saved : {total_saved:,} tokens across {len(clusters)} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
