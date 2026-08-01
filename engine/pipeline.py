"""End-to-end compression pipeline.

Runs stages 2 -> 7 and returns one result object carrying the compressed prompt
plus per-stage telemetry. This is what ``POST /compress`` serialises, and what
the dashboard's stage-by-stage accordion reads.

Every stage is optional-by-degradation: a missing model, an unparsable file or
an unreachable Ollama makes a stage report ``skipped`` with a reason and hand
its input to the next stage untouched. The pipeline as a whole has no failure
mode that produces no output.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .abstractive import AbstractiveCompressor
from .chunker import chunk_document, detect_kind
from .config import Config, get_config
from .density import DensityScorer
from .embeddings import EmbeddingModel
from .reconstruct import Reconstructor
from .redundancy import RedundancyDetector
from .selector import BudgetSelector
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, ChunkKind, StageMetrics, StageStatus

log = logging.getLogger(__name__)

#: Fixed stage order. `/compress` always reports all six, using
#: `status: "skipped"` plus a `note` for any that did not run, so the
#: dashboard's stage accordion has the same shape in every environment.
STAGE_NAMES = (
    "chunking",
    "redundancy",
    "density",
    "selection",
    "abstractive",
    "reconstruction",
)


@dataclass
class CompressionResult:
    """Everything the API and dashboard need from one compression run."""

    original_text: str = ""
    compressed_text: str = ""
    source_name: str = "input"
    detected_kind: str = "text"

    original_tokens: int = 0
    compressed_tokens: int = 0
    budget_tokens: int = 0
    budget_ratio: float = 0.0

    stages: list[StageMetrics] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    kept: list[Chunk] = field(default_factory=list)
    dropped: list[Chunk] = field(default_factory=list)
    audit_trail: list[dict] = field(default_factory=list)
    broken_dependencies: list[dict] = field(default_factory=list)
    repaired_dependencies: list[dict] = field(default_factory=list)
    tokenizer_exact: bool = True
    total_ms: float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Fraction of the original tokens removed. The headline number."""
        if self.original_tokens <= 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    def stage(self, name: str) -> StageMetrics | None:
        return next((s for s in self.stages if s.name == name), None)

    def summary(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "detected_kind": self.detected_kind,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.original_tokens - self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 4),
            "compression_pct": round(100 * self.compression_ratio, 2),
            "budget_tokens": self.budget_tokens,
            "budget_ratio": self.budget_ratio,
            "chunks_total": len(self.chunks),
            "chunks_kept": len(self.kept),
            "chunks_dropped": len(self.dropped),
            "broken_dependencies": len(self.broken_dependencies),
            "repaired_dependencies": len(self.repaired_dependencies),
            "total_ms": round(self.total_ms, 2),
            "tokenizer_exact": self.tokenizer_exact,
        }

    def spans(self) -> list[dict[str, Any]]:
        """Character ranges over the original, flagged kept/dropped.

        The client already holds the text it submitted, so echoing it back is
        pure waste - on a 107k-token log that was 2.17 MB of a 2.17 MB
        response. These offsets let the frontend highlight its own copy
        instead, which drops the payload to ~15 KB.

        Injected protected chunks (the query, the instruction) have no position
        in the source and are excluded.

        Consecutive chunks with the same kept/dropped status are merged into one
        span. A 2,400-record log otherwise emits 2,400 spans (~600 KB) that the
        frontend would only ever render as a handful of contiguous highlight
        bands anyway. Merged runs carry a `count` instead of per-chunk detail.
        """
        spans: list[dict[str, Any]] = []
        for chunk in sorted(self.chunks, key=lambda c: c.order):
            if chunk.start_char < 0:
                continue
            kept = bool(chunk.selected)
            if spans and spans[-1]["kept"] == kept:
                previous = spans[-1]
                previous["end"] = chunk.end_char
                previous["count"] += 1
                previous["tokens"] += chunk.token_count
                # Per-chunk detail is only meaningful for a run of one.
                previous["symbol"] = None
                previous["density"] = None
                previous["kind"] = "mixed" if previous["kind"] != chunk.kind else chunk.kind
                continue
            spans.append(
                {
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                    "kept": kept,
                    "count": 1,
                    "tokens": chunk.token_count,
                    "kind": chunk.kind,
                    "symbol": chunk.symbol,
                    "density": chunk.density,
                    "reason": chunk.drop_reason,
                    "duplicate_count": chunk.duplicate_count,
                }
            )
        return spans

    def to_dict(
        self, include_original: bool = False, include_chunks: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary": self.summary(),
            "stages": [s.to_dict() for s in self.stages],
            "audit_trail": self.audit_trail,
            "broken_dependencies": self.broken_dependencies,
            "repaired_dependencies": self.repaired_dependencies,
            "compressed_text": self.compressed_text,
            "spans": self.spans(),
        }
        if include_original:
            payload["original_text"] = self.original_text
        if include_chunks:
            payload["chunks"] = [c.to_dict(include_text=False) for c in self.chunks]
        return payload


class CompressionPipeline:
    """Stages 2-7, wired together and individually observable."""

    def __init__(
        self,
        cfg: Config | None = None,
        tokenizer: Tokenizer | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)
        self.embedder = embedder or EmbeddingModel(self.cfg.redundancy)
        self.redundancy = RedundancyDetector(self.cfg, self.tokenizer, self.embedder)
        self.density = DensityScorer(self.cfg, self.tokenizer)
        self.selector = BudgetSelector(self.cfg)
        self.abstractive = AbstractiveCompressor(self.cfg, self.tokenizer)
        self.reconstructor = Reconstructor(self.cfg, self.tokenizer)

    def warmup(self) -> dict[str, float]:
        """Pre-load models so the first real request measures compression."""
        timings = {"embeddings_ms": self.embedder.warmup()}
        started = time.perf_counter()
        self.density.entity_scorer.count_all(["warmup text with 3 numbers"])
        timings["entities_ms"] = (time.perf_counter() - started) * 1000
        if self.cfg.abstractive.enabled and self.abstractive.client.available():
            timings["ollama_ms"] = self.abstractive.client.warmup()
        return timings

    def compress(
        self,
        source: str,
        name: str = "input",
        budget_ratio: float | None = None,
        kind: str = "auto",
        query: str | None = None,
        instruction: str | None = None,
        fast_mode: bool = False,
    ) -> CompressionResult:
        started = time.perf_counter()
        ratio = (
            budget_ratio if budget_ratio is not None else self.cfg.selection.budget_ratio
        )
        result = CompressionResult(
            original_text=source,
            source_name=name,
            budget_ratio=ratio,
            tokenizer_exact=self.tokenizer.is_exact,
        )
        if not source or not source.strip():
            # Still emit the full six-stage shape: the dashboard accordion is
            # built against a fixed stage list and must not change shape.
            result.stages = [
                StageMetrics(name=name, status=StageStatus.SKIPPED, note="empty input")
                for name in STAGE_NAMES
            ]
            result.total_ms = (time.perf_counter() - started) * 1000
            return result

        # --- stage 2: chunking ---
        chunk_started = time.perf_counter()
        chunks = chunk_document(source, name, kind, self.cfg.chunking, self.tokenizer)
        result.detected_kind = detect_kind(name, source) if kind == "auto" else kind

        # Protected chunks are prepended so they can never be dropped, and are
        # renumbered ahead of the context they apply to.
        protected = self._protected_chunks(query, instruction, name)
        if protected:
            for chunk in chunks:
                chunk.order += len(protected)
            chunks = protected + chunks

        original_tokens = self.tokenizer.count(source)
        chunk_tokens = sum(c.token_count for c in chunks)
        result.original_tokens = original_tokens
        result.chunks = chunks
        result.stages.append(
            StageMetrics(
                name="chunking",
                status=StageStatus.OK,
                duration_ms=(time.perf_counter() - chunk_started) * 1000,
                chunks_in=1,
                chunks_out=len(chunks),
                tokens_in=original_tokens,
                tokens_out=chunk_tokens,
                details={
                    "backend": result.detected_kind,
                    "protected_chunks": len(protected),
                    "coverage_pct": round(
                        100 * chunk_tokens / max(1, original_tokens), 2
                    ),
                },
            )
        )

        # --- stage 3: redundancy ---
        redundancy = self.redundancy.run(chunks)
        result.stages.append(redundancy.metrics)

        # --- stage 4: density ---
        density = self.density.run(redundancy.chunks, redundancy.embeddings)
        result.stages.append(density.metrics)

        # --- stages 5 + 7: select, reconstruct, and hold the budget ---
        #
        # Markers are not free, and their cost is not knowable before selection:
        # it depends on how many separate runs of dropped chunks the kept set
        # leaves behind. A fixed percentage reserve therefore cannot hold the
        # line - measured overshoot ranged from +12 to +119 tokens. So the two
        # stages run as a short feedback loop: reconstruct, measure the real
        # prompt, and if it exceeds the budget, shrink the content allowance by
        # the overshoot and re-select. Converges in one or two passes and makes
        # "under budget" a property of the artifact that ships, not of an
        # intermediate.
        absorbed_ids = {
            member_id
            for cluster in redundancy.clusters.values()
            for member_id in cluster.member_ids[1:]
        }
        target_tokens = max(1, int(round(original_tokens * ratio)))
        allowance = target_tokens
        selection = None
        reconstruction = None
        attempts = 0

        for attempts in range(1, 5):
            candidate = self.selector.run(
                redundancy.chunks,
                original_tokens=original_tokens,
                budget_tokens=allowance,
            )
            rebuilt = self.reconstructor.run(
                chunks, candidate.kept, redundancy.clusters, absorbed_ids
            )
            final_tokens = rebuilt.metrics.tokens_out

            if (
                reconstruction is None
                or final_tokens <= target_tokens < reconstruction.metrics.tokens_out
                or (
                    final_tokens > target_tokens
                    and final_tokens < reconstruction.metrics.tokens_out
                )
            ):
                selection, reconstruction = candidate, rebuilt

            if final_tokens <= target_tokens:
                break
            allowance -= (final_tokens - target_tokens) + 1
            if allowance < 1:
                break

        selection.metrics.details["budget_attempts"] = attempts
        selection.metrics.details["target_tokens"] = target_tokens
        selection.metrics.details["budget_tokens"] = target_tokens
        result.stages.append(selection.metrics)
        # Report the budget the caller asked for. The loop's shrunken internal
        # allowance is an implementation detail and must not leak into the API.
        result.budget_tokens = target_tokens
        result.kept = selection.kept
        result.audit_trail = selection.audit_trail()
        result.broken_dependencies = [d.to_dict() for d in selection.broken_dependencies]
        result.repaired_dependencies = [
            d.to_dict() for d in selection.repaired_dependencies
        ]

        # Chunks absorbed in stage 3 never reached the selector; they are
        # dropped too, and the audit trail should not pretend otherwise.
        kept_ids = {c.id for c in selection.kept}
        result.dropped = [c for c in chunks if c.id not in kept_ids]

        # --- stage 6: abstractive compression ---
        #
        # Runs *after* the budget loop, not inside it: paraphrasing on every
        # iteration would mean up to four rounds of LLM calls. Because it only
        # ever shrinks chunks, applying it afterwards cannot break the budget
        # the loop just established - the result simply lands further under it.
        abstractive = self.abstractive.run(selection.kept, fast_mode=fast_mode)
        result.stages.append(abstractive.metrics)

        if abstractive.accepted:
            reconstruction = self.reconstructor.run(
                chunks, selection.kept, redundancy.clusters, absorbed_ids
            )

        # --- stage 7: reconstruction ---
        result.stages.append(reconstruction.metrics)
        result.compressed_text = reconstruction.text
        result.compressed_tokens = reconstruction.metrics.tokens_out

        result.total_ms = (time.perf_counter() - started) * 1000
        return result

    # -- helpers -----------------------------------------------------------
    def _protected_chunks(
        self, query: str | None, instruction: str | None, name: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for order, (text, kind) in enumerate(
            [(instruction, ChunkKind.INSTRUCTION), (query, ChunkKind.QUERY)]
        ):
            if not text or not text.strip():
                continue
            chunks.append(
                Chunk(
                    text=text.strip(),
                    kind=kind,
                    source=name,
                    order=len(chunks),
                    start_line=0,
                    end_line=0,
                    token_count=self.tokenizer.count(text),
                    id=f"{name}#protected:{kind}",
                    symbol=kind,
                    metadata={"protected": True},
                )
            )
        return chunks


def compress(
    source: str,
    name: str = "input",
    budget_ratio: float | None = None,
    kind: str = "auto",
    query: str | None = None,
    instruction: str | None = None,
    cfg: Config | None = None,
) -> CompressionResult:
    """One-shot compression. Prefer reusing a pipeline to keep models warm."""
    return CompressionPipeline(cfg).compress(
        source, name, budget_ratio, kind, query, instruction
    )
