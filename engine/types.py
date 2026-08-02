"""Core data structures shared by every pipeline stage.

A single ``Chunk`` object flows through the whole pipeline and accumulates
annotations. Each field below records which stage owns it, so any stage can be
run and inspected in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class ChunkKind:
    """Canonical chunk kinds. Strings (not an Enum) so they serialise cleanly."""

    # --- code ---
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    CLASS_HEADER = "class_header"
    MODULE_LEVEL = "module_level"

    # --- prose ---
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CODE_BLOCK = "code_block"
    LIST = "list"

    # --- logs ---
    LOG_RECORD = "log_record"

    # --- protected: never dropped by the selector ---
    QUERY = "query"
    INSTRUCTION = "instruction"
    SYSTEM = "system"

    OTHER = "other"

    CODE_KINDS = frozenset({FUNCTION, CLASS, METHOD, CLASS_HEADER, MODULE_LEVEL})
    ATOMIC = frozenset({FUNCTION, CLASS, METHOD, LOG_RECORD, CODE_BLOCK})


class StageStatus:
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Chunk:
    """One semantic unit of the input context."""

    # --- stage 2 (chunker) ---
    text: str
    kind: str
    source: str
    order: int  # position in the original document; drives reconstruction order
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    start_char: int = -1  # character offset into the original source
    end_char: int = -1  # exclusive
    token_count: int = 0
    id: str = ""
    symbol: str | None = None  # function/class name, log template, heading text
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- stage 3 (redundancy detector) ---
    cluster_id: int | None = None
    duplicate_count: int = 1  # how many original chunks this one represents
    absorbed_ids: list[str] = field(default_factory=list)

    # --- stage 4 (density scorer) ---
    density: float | None = None
    density_parts: dict[str, float] = field(default_factory=dict)

    # --- stage 5 (selector) ---
    selected: bool | None = None
    drop_reason: str | None = None

    # --- stage 6 (abstractive compressor) ---
    compressed_text: str | None = None

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def output_text(self) -> str:
        """Text as it would appear in the reconstructed prompt."""
        return self.compressed_text if self.compressed_text is not None else self.text

    def preview(self, width: int = 72) -> str:
        """Single-line preview for CLI tables and the dashboard."""
        flat = re.sub(r"\s+", " ", self.text).strip()
        return flat if len(flat) <= width else flat[: width - 1] + "…"

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "order": self.order,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "symbol": self.symbol,
            "duplicate_count": self.duplicate_count,
            "cluster_id": self.cluster_id,
            "density": self.density,
            "density_parts": self.density_parts,
            "selected": self.selected,
            "drop_reason": self.drop_reason,
            "metadata": self.metadata,
        }
        if include_text:
            out["text"] = self.text
            out["compressed_text"] = self.compressed_text
        else:
            out["preview"] = self.preview()
        return out


@dataclass
class Cluster:
    """A group of near-identical chunks collapsed to one representative.

    Produced by stage 3. The representative keeps its full original text; the
    absorbed members survive only as a count and a list of symbols, which
    stage 7 turns into an audit marker. That is the compromise that buys
    compression without silently deleting the *existence* of the duplicates.
    """

    id: int
    representative_id: str
    member_ids: list[str] = field(default_factory=list)
    absorbed_symbols: list[str] = field(default_factory=list)
    absorbed_tokens: int = 0
    method: str = "exact"  # "exact" (hash/template) or "embedding" (cosine)
    mean_similarity: float = 1.0

    @property
    def size(self) -> int:
        return len(self.member_ids)

    @property
    def absorbed_count(self) -> int:
        return max(0, self.size - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "representative_id": self.representative_id,
            "size": self.size,
            "absorbed_count": self.absorbed_count,
            "absorbed_symbols": self.absorbed_symbols,
            "absorbed_tokens": self.absorbed_tokens,
            "method": self.method,
            "mean_similarity": round(self.mean_similarity, 4),
            "member_ids": self.member_ids,
        }


@dataclass
class StageMetrics:
    """Per-stage telemetry. Every number the dashboard shows traces back here."""

    name: str
    status: str = StageStatus.OK
    duration_ms: float = 0.0
    chunks_in: int = 0
    chunks_out: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    note: str | None = None
    #: Which provider actually served this stage, for the stages that call a
    #: model (redundancy, abstractive). Names the chain entry that answered -
    #: not the one that was tried first - so a fallback is visible rather than
    #: implied. None for the stages that call no model at all.
    provider_used: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_removed(self) -> int:
        return max(0, self.tokens_in - self.tokens_out)

    @property
    def reduction_pct(self) -> float:
        if self.tokens_in <= 0:
            return 0.0
        return 100.0 * self.tokens_removed / self.tokens_in

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "chunks_in": self.chunks_in,
            "chunks_out": self.chunks_out,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_removed": self.tokens_removed,
            "reduction_pct": round(self.reduction_pct, 2),
            "note": self.note,
            "provider_used": self.provider_used,
            "details": self.details,
        }
