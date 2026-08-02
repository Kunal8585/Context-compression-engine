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
    #: What each marker hides, keyed by the id printed inside it. Lets a
    #: consumer recover a specific omission instead of re-compressing.
    recoverable: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "marker_count": self.marker_count,
            "recoverable": self.recoverable,
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
        #: Every marker's recoverable payload, keyed by its id. This is what
        #: turns a marker from an apology into an address: a consumer that hits
        #: `[... omitted ... #d3]` and decides it needs that content can ask for
        #: it back rather than re-running the whole compression at a looser
        #: budget. Compression with an escape hatch.
        recoverable: list[dict] = []
        pending_drops: list[Chunk] = []

        def flush_drops() -> None:
            if not pending_drops or not settings.drop_markers:
                pending_drops.clear()
                return
            marker_id = f"d{len(recoverable)}"
            marker = self._drop_marker(pending_drops, marker_id)
            pieces.append(marker)
            kinds.append("marker")
            markers.append(marker)
            recoverable.append({
                "id": marker_id,
                "kind": "dropped",
                "chunk_ids": [c.id for c in pending_drops],
                "sections": len(pending_drops),
                "tokens": sum(c.token_count for c in pending_drops),
                "start_line": pending_drops[0].start_line,
                "end_line": pending_drops[-1].end_line,
                "start_char": pending_drops[0].start_char,
                "end_char": pending_drops[-1].end_char,
            })
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
                marker_id = f"c{cluster.id}"
                marker = self._cluster_marker(cluster, marker_id)
                pieces.append(marker)
                kinds.append("marker")
                markers.append(marker)
                recoverable.append({
                    "id": marker_id,
                    "kind": "collapsed",
                    "chunk_ids": list(cluster.member_ids[1:]),
                    "sections": cluster.size - 1,
                    "tokens": cluster.absorbed_tokens,
                    "representative_id": cluster.representative_id,
                    "method": cluster.method,
                })

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
            "recoverable_markers": len(recoverable),
        }
        return ReconstructionResult(
            text=text,
            metrics=metrics,
            marker_count=len(markers),
            marker_tokens=marker_tokens,
            recoverable=recoverable,
        )

    # -- markers -----------------------------------------------------------
    @staticmethod
    def _drop_marker(dropped: list[Chunk], marker_id: str = "") -> str:
        tokens = sum(c.token_count for c in dropped)
        lines = f"{dropped[0].start_line}-{dropped[-1].end_line}"
        tag = f" #{marker_id}" if marker_id else ""
        head = (
            f"[... omitted{tag} {len(dropped)} section(s), {tokens} tokens, "
            f"lines {lines}"
        )

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
    def _cluster_marker(cluster: Cluster, marker_id: str = "") -> str:
        tag = f" #{marker_id}" if marker_id else ""
        return (
            f"[x{cluster.size}{tag} near-identical occurrences collapsed "
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
