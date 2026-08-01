"""Stage 7 - reconstruction.

Stitches the selected chunks back into one prompt, in original document order,
with inline markers wherever content was removed.

The markers are the reasoning-retention story. A compressed prompt that silently
omits two thirds of its input invites a model to answer confidently from a gap
it cannot see. A prompt that says ``[... 12 sections omitted ...]`` and ``[x172
near-identical records collapsed]`` tells the model - and the judge reading the
output - exactly what is missing and how much of it there was. Markers cost
tokens, which is why stage 5 reserves budget for them rather than letting them
push the result over.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Config, get_config
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, ChunkKind, Cluster, StageMetrics, StageStatus

_PROSE_KINDS = frozenset(
    {
        ChunkKind.PARAGRAPH,
        ChunkKind.SENTENCE,
        ChunkKind.HEADING,
        ChunkKind.LIST,
        ChunkKind.QUERY,
        ChunkKind.INSTRUCTION,
        ChunkKind.SYSTEM,
    }
)


@dataclass
class ReconstructionResult:
    text: str = ""
    metrics: StageMetrics = field(default_factory=lambda: StageMetrics("reconstruction"))
    marker_count: int = 0
    marker_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "marker_count": self.marker_count,
            "marker_tokens": self.marker_tokens,
        }


class Reconstructor:
    def __init__(
        self, cfg: Config | None = None, tokenizer: Tokenizer | None = None
    ) -> None:
        self.cfg = cfg or get_config()
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)

    def run(
        self,
        all_chunks: list[Chunk],
        kept: list[Chunk],
        clusters: dict[int, Cluster] | None = None,
        absorbed_ids: set[str] | None = None,
    ) -> ReconstructionResult:
        """Stitch `kept` back into a prompt.

        ``absorbed_ids`` are chunks stage 3 folded into a cluster. They get no
        drop marker of their own: the ``[x172 collapsed]`` marker on their
        representative already accounts for them, and emitting both made the
        markers cost more than the content they replaced - a 2,400-record log
        reconstructed to 18,859 tokens against 6,560 tokens of actual content.
        """
        started = time.perf_counter()
        clusters = clusters or {}
        metrics = StageMetrics(
            name="reconstruction",
            chunks_in=len(kept),
            chunks_out=len(kept),
            tokens_in=sum(c.token_count for c in kept),
        )
        if not kept:
            metrics.status = StageStatus.SKIPPED
            metrics.note = "nothing selected to reconstruct"
            return ReconstructionResult(metrics=metrics)

        settings = self.cfg.reconstruction
        kept_ids = {c.id for c in kept}
        absorbed_ids = absorbed_ids or set()
        ordered = [
            c for c in sorted(all_chunks, key=lambda c: c.order)
            if c.id not in absorbed_ids
        ]

        pieces: list[str] = []
        kinds: list[str] = []
        markers: list[str] = []
        pending_drops: list[Chunk] = []

        def flush_drops() -> None:
            if not pending_drops or not settings.drop_markers:
                pending_drops.clear()
                return
            marker = self._drop_marker(pending_drops)
            pieces.append(marker)
            kinds.append("marker")
            markers.append(marker)
            pending_drops.clear()

        for chunk in ordered:
            if chunk.id not in kept_ids:
                pending_drops.append(chunk)
                continue

            flush_drops()
            pieces.append(chunk.output_text.rstrip())
            kinds.append(chunk.kind)

            cluster = clusters.get(chunk.cluster_id) if chunk.cluster_id is not None else None
            if cluster is not None and settings.cluster_markers:
                marker = self._cluster_marker(cluster)
                pieces.append(marker)
                kinds.append("marker")
                markers.append(marker)

        flush_drops()

        text = self._join(pieces, kinds)
        marker_tokens = sum(self.tokenizer.count(m) for m in markers)
        total_tokens = self.tokenizer.count(text)

        metrics.tokens_out = total_tokens
        metrics.duration_ms = (time.perf_counter() - started) * 1000
        metrics.status = StageStatus.OK
        metrics.details = {
            "marker_count": len(markers),
            "marker_tokens": marker_tokens,
            "content_tokens": total_tokens - marker_tokens,
            "characters": len(text),
        }
        return ReconstructionResult(
            text=text,
            metrics=metrics,
            marker_count=len(markers),
            marker_tokens=marker_tokens,
        )

    # -- markers -----------------------------------------------------------
    @staticmethod
    def _drop_marker(dropped: list[Chunk]) -> str:
        tokens = sum(c.token_count for c in dropped)
        lines = f"{dropped[0].start_line}-{dropped[-1].end_line}"
        head = f"[... omitted {len(dropped)} section(s), {tokens} tokens, lines {lines}"

        # Naming what went is worth real budget for code (a dropped function
        # name is a searchable fact) but not for logs, whose "symbol" is a
        # 120-character template that would cost more than the record it stands
        # for. Only short, human-meaningful symbols are listed.
        symbols = [
            c.symbol
            for c in dropped
            if c.symbol and c.kind in ChunkKind.CODE_KINDS and len(c.symbol) <= 40
        ]
        if symbols:
            shown = ", ".join(symbols[:4])
            if len(symbols) > 4:
                shown += f", +{len(symbols) - 4} more"
            return f"{head}: {shown} ...]"
        return f"{head} ...]"

    @staticmethod
    def _cluster_marker(cluster: Cluster) -> str:
        return (
            f"[x{cluster.size} near-identical occurrences collapsed "
            f"({cluster.absorbed_tokens} tokens saved, {cluster.method} match)]"
        )

    @staticmethod
    def _join(pieces: list[str], kinds: list[str]) -> str:
        """Blank line between prose, single newline between records."""
        if not pieces:
            return ""
        out = [pieces[0]]
        for index in range(1, len(pieces)):
            previous, current = kinds[index - 1], kinds[index]
            compact = (
                previous in {ChunkKind.LOG_RECORD, "marker"}
                and current in {ChunkKind.LOG_RECORD, "marker"}
            )
            separator = "\n" if compact else "\n\n"
            out.append(separator)
            out.append(pieces[index])
        return "".join(out)


def reconstruct(
    all_chunks: list[Chunk],
    kept: list[Chunk],
    clusters: dict[int, Cluster] | None = None,
    absorbed_ids: set[str] | None = None,
    cfg: Config | None = None,
    tokenizer: Tokenizer | None = None,
) -> ReconstructionResult:
    """Convenience wrapper around :class:`Reconstructor`."""
    return Reconstructor(cfg, tokenizer).run(all_chunks, kept, clusters, absorbed_ids)
