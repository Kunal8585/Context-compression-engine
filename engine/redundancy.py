"""Stage 3 - redundancy detection.

Two passes, cheapest first:

1. **Exact pass.** Chunks are keyed by a normalised hash - or, for log records,
   by the template stage 2 already extracted. This collapses the bulk of a log
   file (93.9% of our sample) for the cost of a dict lookup, and it works with
   no model loaded at all. That matters: it is the floor the pipeline degrades
   to when sentence-transformers is unavailable.

2. **Embedding pass.** Survivors are embedded with MiniLM and clustered by
   greedy leader assignment: walk chunks in document order, compare against the
   representatives chosen so far, and join the first cluster within
   ``similarity_threshold`` (cosine). Order-deterministic, O(n*k) rather than
   the O(n^2) distance matrix agglomerative clustering would need, and the
   representative is always the *earliest* occurrence - which for a log is the
   first time the event happened, the one you actually want to keep.

What a cluster keeps and what it discards is the core retention decision. The
representative keeps its complete original text. The absorbed members survive
as a count plus their symbols, which stage 7 renders as an audit marker. So
"400 identical cache-hit lines" becomes one line and a "x400" annotation - the
*fact* of the repetition is preserved, only the redundant bytes are gone.

Chunks whose kind is in ``selection.protected_kinds`` (the user's question, the
system instruction) bypass both passes entirely and are never collapsed.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config, get_config
from .embeddings import EmbeddingModel
from .structural import structural_signature
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, Cluster, StageMetrics, StageStatus

log = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def dedup_key(chunk: Chunk) -> str:
    """Key for the exact pass.

    Log records key on their extracted template, so records differing only in
    timestamp / request id collapse. Everything else keys on a hash of its
    whitespace-normalised text - conservative by design: we do not lowercase or
    strip punctuation, because for code those carry meaning.
    """
    template = chunk.metadata.get("template")
    if template:
        return f"tpl:{template}"
    normalised = _WHITESPACE.sub(" ", chunk.text).strip()
    digest = hashlib.blake2b(normalised.encode("utf-8"), digest_size=16).hexdigest()
    return f"txt:{digest}"


class _Absorptions:
    """Union-find over chunk ids, tracking who absorbed whom and how.

    Absorption is transitive and the passes run in sequence, so a chunk that
    represents 30 identical log records in the exact pass can *itself* be
    absorbed by a different leader in the embedding pass. Storing a flat
    leader -> members map loses those 30 records from the audit trail (the
    chunks are still correctly removed, but stage 7 would under-report the
    collapse). Resolving to a root fixes that: find() walks the chain to the
    chunk that actually survived.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._meta: dict[str, tuple[float, str]] = {}

    def absorb(self, leader: Chunk, member: Chunk, similarity: float, method: str) -> None:
        self._parent[member.id] = leader.id
        self._meta[member.id] = (similarity, method)

    def find(self, chunk_id: str) -> str:
        root = chunk_id
        while root in self._parent:
            root = self._parent[root]
        # Path compression keeps repeated lookups cheap on long chains.
        cursor = chunk_id
        while cursor in self._parent and self._parent[cursor] != root:
            nxt = self._parent[cursor]
            self._parent[cursor] = root
            cursor = nxt
        return root

    def is_absorbed(self, chunk_id: str) -> bool:
        return chunk_id in self._parent

    def absorbed_ids(self) -> set[str]:
        return set(self._parent)

    def meta_for(self, chunk_id: str) -> tuple[float, str]:
        return self._meta.get(chunk_id, (1.0, "exact"))


@dataclass
class RedundancyResult:
    """Output of stage 3."""

    chunks: list[Chunk]  # survivors, in original document order
    clusters: dict[int, Cluster] = field(default_factory=dict)
    metrics: StageMetrics = field(default_factory=lambda: StageMetrics("redundancy"))
    #: L2-normalised embeddings aligned row-for-row with ``chunks``; None when
    #: the embedding backend was unavailable and only the exact pass ran.
    embeddings: np.ndarray | None = None

    def cluster_for(self, chunk: Chunk) -> Cluster | None:
        if chunk.cluster_id is None:
            return None
        return self.clusters.get(chunk.cluster_id)

    def to_dict(self, include_clusters: bool = True) -> dict:
        payload: dict = {"metrics": self.metrics.to_dict()}
        if include_clusters:
            payload["clusters"] = [
                cluster.to_dict()
                for cluster in sorted(
                    self.clusters.values(), key=lambda c: c.size, reverse=True
                )
            ]
        return payload


class RedundancyDetector:
    def __init__(
        self,
        cfg: Config | None = None,
        tokenizer: Tokenizer | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)
        # Full config, not just `redundancy`: the embedder now resolves a
        # provider chain and needs to see `providers` to build it.
        self.embedder = embedder or EmbeddingModel(self.cfg)
        #: (threshold, was_calibrated) from the most recent embedding pass.
        self._last_threshold: tuple[float, bool] = (
            self.cfg.redundancy.similarity_threshold, False
        )

    # -- public API --------------------------------------------------------
    def run(self, chunks: list[Chunk]) -> RedundancyResult:
        started = time.perf_counter()
        tokens_in = sum(c.token_count for c in chunks)
        metrics = StageMetrics(
            name="redundancy", chunks_in=len(chunks), tokens_in=tokens_in
        )

        if not chunks:
            metrics.status = StageStatus.SKIPPED
            metrics.note = "no chunks to deduplicate"
            return RedundancyResult(chunks=[], metrics=metrics)

        # Reset any annotations from a previous run so the stage is idempotent.
        for chunk in chunks:
            chunk.cluster_id = None
            chunk.duplicate_count = 1
            chunk.absorbed_ids = []

        protected_kinds = set(self.cfg.selection.protected_kinds)
        absorbed = _Absorptions()

        survivors, exact_collapsed = self._exact_pass(chunks, protected_kinds, absorbed)
        survivors, structural_collapsed = self._structural_pass(
            survivors, protected_kinds, absorbed
        )

        embeddings = None
        embedding_collapsed = 0
        embedding_note: str | None = None
        if not self.cfg.redundancy.enabled:
            embedding_note = "redundancy.enabled is false; exact pass only"
        else:
            embeddings, embedding_collapsed, embedding_note = self._embedding_pass(
                survivors, protected_kinds, absorbed
            )

        kept = [c for c in survivors if not absorbed.is_absorbed(c.id)]
        kept.sort(key=lambda c: c.order)

        clusters = self._build_clusters(chunks, kept, absorbed)
        kept_embeddings = self._align_embeddings(survivors, kept, embeddings)

        metrics.chunks_out = len(kept)
        metrics.tokens_out = sum(c.token_count for c in kept)

        # Accounting invariant: every token that left the pipeline must be
        # attributed to exactly one cluster. If this drifts, the cluster records
        # (and therefore the stage 7 drop markers) are under-reporting what was
        # removed, even though the headline ratio still looks right.
        accounted = sum(c.absorbed_tokens for c in clusters.values())
        removed = metrics.tokens_in - metrics.tokens_out
        if accounted != removed:
            log.warning(
                "redundancy accounting mismatch: %d tokens removed but %d "
                "attributed to clusters",
                removed,
                accounted,
            )
        metrics.duration_ms = (time.perf_counter() - started) * 1000
        metrics.status = StageStatus.OK
        metrics.note = embedding_note
        metrics.provider_used = self.embedder.stats.provider
        largest = max(clusters.values(), key=lambda c: c.size, default=None)
        metrics.details = {
            "exact_collapsed": exact_collapsed,
            "structural_collapsed": structural_collapsed,
            "embedding_collapsed": embedding_collapsed,
            "clusters_formed": len(clusters),
            "largest_cluster_size": largest.size if largest else 0,
            "largest_cluster_symbol": (
                _label_for(largest, kept) if largest else None
            ),
            "similarity_threshold": self._last_threshold[0],
            "similarity_threshold_calibrated": self._last_threshold[1],
            "tokens_attributed_to_clusters": accounted,
            "accounting_ok": accounted == removed,
            "protected_chunks": sum(1 for c in kept if c.kind in protected_kinds),
            # Promoted to the top level because they are now real cost, not
            # just diagnostics: over an API the embedding pass is N HTTP
            # requests against a rate limit, where locally it was one matrix
            # multiply whose only cost was wall-clock on this machine.
            "embedding_provider": self.embedder.stats.provider,
            "embedding_api_calls": self.embedder.stats.api_calls,
            "embedding_latency_ms": round(self.embedder.stats.provider_latency_ms, 2),
            "embedding_fell_back": self.embedder.stats.fell_back,
            "embedding": self.embedder.stats.to_dict()
            if embeddings is not None
            else {"available": False, "error": self.embedder.load_error},
        }
        return RedundancyResult(
            chunks=kept, clusters=clusters, metrics=metrics, embeddings=kept_embeddings
        )

    # -- pass 1: exact -----------------------------------------------------
    def _exact_pass(
        self,
        chunks: list[Chunk],
        protected_kinds: set[str],
        absorbed: _Absorptions,
    ) -> tuple[list[Chunk], int]:
        if not self.cfg.redundancy.exact_hash_dedup:
            return list(chunks), 0

        representatives: dict[str, Chunk] = {}
        survivors: list[Chunk] = []
        collapsed = 0

        for chunk in chunks:
            if chunk.kind in protected_kinds:
                survivors.append(chunk)
                continue
            key = dedup_key(chunk)
            existing = representatives.get(key)
            if existing is None:
                representatives[key] = chunk
                survivors.append(chunk)
                continue
            absorbed.absorb(existing, chunk, 1.0, "exact")
            collapsed += 1
        return survivors, collapsed

    # -- pass 1.5: structural (code only) ----------------------------------
    def _structural_pass(
        self,
        survivors: list[Chunk],
        protected_kinds: set[str],
        absorbed: _Absorptions,
    ) -> tuple[list[Chunk], int]:
        """Collapse code chunks with an identical token shape.

        See :mod:`engine.structural` for why this exists and why numbers are
        deliberately *not* normalised.
        """
        settings = self.cfg.redundancy.structural
        if not settings.enabled:
            return survivors, 0

        eligible_kinds = set(settings.kinds)
        representatives: dict[str, Chunk] = {}
        kept: list[Chunk] = []
        collapsed = 0

        for chunk in survivors:
            if (
                chunk.kind in protected_kinds
                or chunk.kind not in eligible_kinds
                or chunk.token_count < settings.min_tokens
            ):
                kept.append(chunk)
                continue

            signature = structural_signature(chunk.text, chunk.kind)
            if signature is None:
                kept.append(chunk)
                continue

            existing = representatives.get(signature)
            if existing is None:
                representatives[signature] = chunk
                kept.append(chunk)
                continue
            absorbed.absorb(existing, chunk, 1.0, "structural")
            collapsed += 1
        return kept, collapsed

    # -- pass 2: embeddings ------------------------------------------------
    def _embedding_pass(
        self,
        survivors: list[Chunk],
        protected_kinds: set[str],
        absorbed: _Absorptions,
    ) -> tuple[np.ndarray | None, int, str | None]:
        """Greedy leader clustering over cosine similarity.

        Returns (embeddings aligned to ``survivors``, collapsed count, note).
        """
        if len(survivors) < 2:
            return None, 0, "fewer than two chunks; embedding pass skipped"

        vectors = self.embedder.encode([c.text for c in survivors])
        if vectors is None:
            return (
                None,
                0,
                f"embedding backend unavailable ({self.embedder.load_error}); "
                f"exact-hash dedup only",
            )

        threshold, calibrated = self._threshold_for(self.embedder.stats.provider)
        dimension = vectors.shape[1]
        leaders = np.zeros((len(survivors), dimension), dtype=np.float32)
        leader_chunks: list[Chunk] = []
        collapsed = 0

        skipped_precise = 0
        for index, chunk in enumerate(survivors):
            if chunk.kind in protected_kinds:
                continue  # never collapsed, and never a magnet for others

            # A chunk that already has a precise dedup key (a log template) is
            # exempt. Stage 2 deliberately keeps the service and region in the
            # template so records from different services never merge; fuzzy
            # matching scores those at 0.998 and would merge them anyway,
            # silently undoing that decision. Fuzzy clustering can only destroy
            # distinctions a precise key has already captured, never add any.
            if chunk.metadata.get("template"):
                skipped_precise += 1
                continue
            count = len(leader_chunks)
            if count:
                # Vectors are L2-normalised, so a dot product *is* cosine.
                similarities = leaders[:count] @ vectors[index]
                best = int(np.argmax(similarities))
                score = float(similarities[best])
                if score >= threshold:
                    absorbed.absorb(leader_chunks[best], chunk, score, "embedding")
                    collapsed += 1
                    continue
            leaders[count] = vectors[index]
            leader_chunks.append(chunk)

        notes = []
        if skipped_precise:
            notes.append(
                f"{skipped_precise} chunks with an exact dedup key were exempt "
                f"from fuzzy clustering"
            )
        if not calibrated:
            # Loud, because the failure mode is silent: an uncalibrated
            # threshold does not error, it just collapses nothing.
            notes.append(
                f"similarity threshold {threshold} is not calibrated for "
                f"'{self.embedder.stats.provider}'; fuzzy dedup may under- or "
                f"over-collapse (see redundancy.similarity_threshold_by_provider)"
            )
        self._last_threshold = (threshold, calibrated)
        return vectors, collapsed, "; ".join(notes) or None

    def _threshold_for(self, provider: str | None) -> tuple[float, bool]:
        """Similarity threshold for whichever provider actually embedded.

        Cosine similarity is not comparable across embedding models. Measured
        on the same 26 near-duplicate support tickets: MiniLM's most similar
        pair scores 0.908 and Cohere's 0.877, so the 0.88 default that merges
        7 pairs under MiniLM merges **zero** under Cohere - the fuzzy pass
        stops working with no error at all. Hence a per-provider map.
        """
        overrides = self.cfg.redundancy.similarity_threshold_by_provider
        if provider and provider in overrides:
            return overrides[provider], True
        return self.cfg.redundancy.similarity_threshold, False

    # -- bookkeeping -------------------------------------------------------
    @staticmethod
    def _build_clusters(
        all_chunks: list[Chunk],
        kept: list[Chunk],
        absorbed: _Absorptions,
    ) -> dict[int, Cluster]:
        """Group every absorbed chunk under the survivor it resolves to."""
        members_by_root: dict[str, list[Chunk]] = {}
        for chunk in all_chunks:
            if not absorbed.is_absorbed(chunk.id):
                continue
            members_by_root.setdefault(absorbed.find(chunk.id), []).append(chunk)

        clusters: dict[int, Cluster] = {}
        next_id = 0
        for chunk in kept:
            members = members_by_root.get(chunk.id)
            if not members:
                continue
            members.sort(key=lambda c: c.order)
            details = [absorbed.meta_for(m.id) for m in members]
            methods = {method for _, method in details}
            similarities = [score for score, _ in details]
            cluster = Cluster(
                id=next_id,
                representative_id=chunk.id,
                member_ids=[chunk.id] + [m.id for m in members],
                absorbed_symbols=[
                    label for label in (_member_label(m) for m in members) if label
                ],
                absorbed_tokens=sum(m.token_count for m in members),
                method="mixed" if len(methods) > 1 else next(iter(methods)),
                mean_similarity=float(np.mean(similarities)) if similarities else 1.0,
            )
            clusters[next_id] = cluster
            chunk.cluster_id = next_id
            chunk.duplicate_count = cluster.size
            chunk.absorbed_ids = [m.id for m in members]
            next_id += 1
        return clusters

    @staticmethod
    def _align_embeddings(
        survivors: list[Chunk], kept: list[Chunk], embeddings: np.ndarray | None
    ) -> np.ndarray | None:
        if embeddings is None:
            return None
        row_of = {chunk.id: index for index, chunk in enumerate(survivors)}
        rows = [row_of[chunk.id] for chunk in kept if chunk.id in row_of]
        if len(rows) != len(kept):
            # Should not happen; refuse to return a misaligned matrix rather
            # than let stage 4 score chunks against the wrong vectors.
            log.warning("embedding alignment mismatch; dropping embeddings")
            return None
        return embeddings[rows]


def _member_label(chunk: Chunk) -> str | None:
    """Short human label for an absorbed chunk, used in stage 7 markers."""
    if chunk.symbol:
        return chunk.symbol[:80]
    return chunk.preview(60)


def _label_for(cluster: Cluster, kept: list[Chunk]) -> str | None:
    representative = next(
        (c for c in kept if c.id == cluster.representative_id), None
    )
    if representative is None:
        return None
    return representative.symbol or representative.preview(60)


def detect_redundancy(
    chunks: list[Chunk],
    cfg: Config | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: EmbeddingModel | None = None,
) -> RedundancyResult:
    """Convenience wrapper around :class:`RedundancyDetector`."""
    return RedundancyDetector(cfg, tokenizer, embedder).run(chunks)
