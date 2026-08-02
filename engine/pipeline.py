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

from .abstractive import AbstractiveCompressor, GenerationClient
from .chunker import chunk_document, detect_kind
from .config import Config, get_config
from .confidence import Confidence, score_compression
from .providers import (
    build_embedding_chain,
    build_generation_chain,
    normalise_mode,
)
from .density import DensityScorer
from .embeddings import EmbeddingModel
from .reconstruct import Reconstructor
from .redundancy import RedundancyDetector
from .selector import BudgetSelector, SelectionResult, KEPT_DENSITY, DROPPED_BUDGET
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
    #: Measured trustworthiness of this compression. See engine/confidence.py.
    confidence: Confidence | None = None
    #: Execution mode this run used: local | cloud | auto.
    mode: str = "auto"
    #: What each marker in the compressed text hides, addressable by the id
    #: printed inside it. See `/expand` - compression with an escape hatch.
    markers: list[dict] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Fraction of the original tokens removed. The headline number."""
        if self.original_tokens <= 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    def stage(self, name: str) -> StageMetrics | None:
        return next((s for s in self.stages if s.name == name), None)

    @property
    def providers_used(self) -> dict[str, str | None]:
        """Which provider served each model-calling stage."""
        return {
            stage.name: stage.provider_used
            for stage in self.stages
            if stage.provider_used
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "detected_kind": self.detected_kind,
            "mode": self.mode,
            "providers_used": self.providers_used,
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
            source_file = chunk.metadata.get("source_file", self.source_name)
            if spans and spans[-1]["kept"] == kept and spans[-1].get("source_file") == source_file:
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
                    "source_file": source_file,
                }
            )
        return spans

    def to_dict(
        self, include_original: bool = False, include_chunks: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary": self.summary(),
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "stages": [s.to_dict() for s in self.stages],
            "audit_trail": self.audit_trail,
            "broken_dependencies": self.broken_dependencies,
            "repaired_dependencies": self.repaired_dependencies,
            "compressed_text": self.compressed_text,
            "spans": self.spans(),
            # Marker *addresses*, not their contents - and deliberately without
            # `chunk_ids`. A 2,400-record log collapses into 146 clusters whose
            # member id lists run to thousands of strings, which would put back
            # exactly the payload weight the spans design exists to remove. The
            # ids stay in-process for `/expand` to resolve.
            "markers": [
                {k: v for k, v in marker.items() if k != "chunk_ids"}
                for marker in self.markers
            ],
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
        mode: str | None = None,
        embedding_provider: str | None = None,
        generation_provider: str | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        # One mode per pipeline instance, fixed at construction. Deliberately
        # not a per-call argument that mutates shared chains: FastAPI runs sync
        # endpoints on a threadpool, so two concurrent requests swapping the
        # embedder on one shared object would interleave and each would report
        # the other's provider. The API keeps one pipeline per mode instead.
        self.mode = normalise_mode(mode)
        # An explicitly chosen provider overrides the mode for that role, and
        # gets no fallback - see _pinned() for why.
        self.embedding_provider = embedding_provider
        self.generation_provider = generation_provider
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)
        self.embedder = embedder or EmbeddingModel(
            self.cfg,
            chain=build_embedding_chain(
                self.cfg, mode=self.mode, pin=embedding_provider
            ),
        )
        self.redundancy = RedundancyDetector(self.cfg, self.tokenizer, self.embedder)
        self.density = DensityScorer(self.cfg, self.tokenizer)
        self.selector = BudgetSelector(self.cfg)
        self.abstractive = AbstractiveCompressor(
            self.cfg,
            self.tokenizer,
            client=GenerationClient(
                build_generation_chain(
                    self.cfg,
                    timeout_s=self.cfg.abstractive.timeout_s,
                    mode=self.mode,
                    pin=generation_provider,
                ),
                self.cfg.abstractive,
            ),
        )
        self.reconstructor = Reconstructor(self.cfg, self.tokenizer)

    def warmup(self) -> dict[str, float]:
        """Pre-load models so the first real request measures compression."""
        timings = {"embeddings_ms": self.embedder.warmup()}
        started = time.perf_counter()
        self.density.entity_scorer.count_all(["warmup text with 3 numbers"])
        timings["entities_ms"] = (time.perf_counter() - started) * 1000
        if self.cfg.abstractive.enabled and self.abstractive.client.available():
            timings["generation_ms"] = self.abstractive.client.warmup()
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
        selection_strategy: str = "density",
        source_files: list[dict[str, Any]] | None = None,
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
            mode=self.mode,
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
        if source_files:
            for chunk in chunks:
                chunk.metadata["source_file"] = next(
                    (entry["name"] for entry in source_files
                     if entry["start"] <= chunk.start_char < entry["end"]),
                    name,
                )
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
        #
        # The query is already a protected chunk, so stage 3 embedded it in the
        # SAME batch as everything else. Reuse that row rather than issuing a
        # second chain call.
        #
        # Two wins, and the second is the important one. It removes a network
        # round trip from every query-aware compression - on a cold chain with
        # exhausted providers ahead of a working one, that was ~1.5 s. And it
        # makes "query and chunks share a vector space" true by construction
        # instead of by check: a separate call can land on a different provider
        # if the first entered cooldown in between, and a 3072-d Gemini query
        # dotted against 1024-d Cohere chunks is a crash, not a weak signal.
        # Same call, same provider, same space - nothing left to verify.
        query_embedding = None
        if query and query.strip() and redundancy.embeddings is not None:
            query_id = f"{name}#protected:{ChunkKind.QUERY}"
            row = next(
                (i for i, c in enumerate(redundancy.chunks) if c.id == query_id),
                None,
            )
            if row is not None and row < len(redundancy.embeddings):
                query_embedding = redundancy.embeddings[row]
            else:
                log.debug("query chunk not among stage 3 survivors; relevance skipped")
        density = self.density.run(
            redundancy.chunks, redundancy.embeddings, query_embedding
        )
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

        protected_kinds = set(self.cfg.selection.protected_kinds)
        previous_used: int | None = None

        for attempts in range(1, 7):
            candidate = self._select(
                redundancy.chunks, original_tokens, allowance, selection_strategy
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

            if candidate.used_tokens == previous_used:
                # Granularity stall: chunks are 14-233 tokens, so trimming the
                # allowance by a 2-token overshoot never crosses a chunk
                # boundary and the loop spins on an identical selection. Drop
                # the allowance below the smallest kept chunk to force progress.
                droppable = [
                    c.token_count
                    for c in candidate.kept
                    if c.kind not in protected_kinds
                ]
                if not droppable:
                    break
                allowance = candidate.used_tokens - min(droppable)
            else:
                allowance -= (final_tokens - target_tokens) + 1
            previous_used = candidate.used_tokens
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
        result.markers = reconstruction.recoverable

        # --- confidence: how much to trust what just came out ---
        #
        # Scored against the *reconstructed* text, not the kept chunk list, so
        # it reflects the artifact that actually ships - drop markers, cluster
        # annotations and any stage 6 paraphrase included. Costs no model call
        # and no measurable time; it is a handful of set operations over text
        # the pipeline is already holding.
        redundancy_metrics = redundancy.metrics
        result.confidence = score_compression(
            source,
            result.compressed_text,
            chunks,
            selection.kept,
            # Stage 3's output is the baseline for lexical retention: what it
            # collapsed was redundant by construction, not lost.
            survivors=redundancy.chunks,
            tokens_removed_by_redundancy=max(
                0, redundancy_metrics.tokens_in - redundancy_metrics.tokens_out
            ),
            tokens_removed_total=max(0, original_tokens - result.compressed_tokens),
            broken_dependencies=len(result.broken_dependencies),
            repaired_dependencies=len(result.repaired_dependencies),
        )

        result.total_ms = (time.perf_counter() - started) * 1000
        return result

    def _select(self, chunks: list[Chunk], original_tokens: int, budget_tokens: int,
                strategy: str) -> SelectionResult:
        """Offline evaluation alternatives; the API keeps the density default."""
        if strategy == "density":
            return self.selector.run(chunks, original_tokens=original_tokens,
                                     budget_tokens=budget_tokens)
        if strategy not in {"truncate", "random"}:
            raise ValueError(f"unknown selection strategy: {strategy}")
        import random
        candidates = list(chunks)
        if strategy == "random":
            random.Random(1337).shuffle(candidates)
        else:
            candidates.sort(key=lambda c: c.order)
        kept: list[Chunk] = []
        used = 0
        for chunk in candidates:
            if used + chunk.token_count <= budget_tokens:
                chunk.selected, chunk.drop_reason = True, KEPT_DENSITY
                kept.append(chunk)
                used += chunk.token_count
            else:
                chunk.selected, chunk.drop_reason = False, DROPPED_BUDGET
        kept.sort(key=lambda c: c.order)
        metrics = StageMetrics(name="selection", chunks_in=len(chunks), chunks_out=len(kept),
                               tokens_in=sum(c.token_count for c in chunks), tokens_out=used,
                               details={"strategy": strategy, "budget_tokens": budget_tokens})
        return SelectionResult(kept=kept, dropped=[c for c in chunks if not c.selected],
                               metrics=metrics, budget_tokens=budget_tokens,
                               used_tokens=used, original_tokens=original_tokens)

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
