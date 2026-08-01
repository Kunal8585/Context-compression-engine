"""Stage 4 - density scoring.

Ranks every surviving chunk by how much reasoning-relevant information it
carries per token, so stage 5 can spend a fixed token budget on the best of
them. Five signals, combined with weights from ``config.yaml``:

``entropy``    token-level Shannon entropy, normalised by chunk length. Detects
               repetitive filler - a chunk that says the same thing four ways.
``tfidf``      mean IDF of the chunk's terms across the document. Rewards rare
               vocabulary (``PAYMENT_POOL_SIZE``, ``ReadTimeout``) and punishes
               chunks built entirely from words every other chunk also uses.
``entities``   named entities, numbers, identifiers and signatures per token -
               concrete facts rather than connective prose. See
               :mod:`engine.entities`.
``novelty``    cosine distance from the corpus centroid (see below).
``structure``  priors from chunk kind, log severity and failure vocabulary.

Plus a small additive ``frequency`` bonus: a stage-3 representative standing in
for 400 absorbed records carries aggregate information a singleton does not.

**Deviation from the brief: novelty is measured against the corpus centroid,
not the chunk's own cluster centroid.** Distance-from-*cluster*-centroid is zero
for every singleton cluster, which would score every unique chunk as maximally
un-novel - exactly backwards, since a one-off ``ERROR`` line is the most novel
thing in a log file. Distance from the *corpus* centroid has the intended
behaviour: boilerplate sits near the mean, the anomaly sits far from it.

Every signal is normalised across the document before weighting, and every
per-signal value is recorded on ``chunk.density_parts`` so any score can be
explained rather than asserted. When a signal is unavailable - no embeddings, no
scikit-learn - its weight is redistributed over the remaining signals and the
stage says so, rather than silently scoring it zero.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import Config, get_config
from .entities import EntityScorer
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, StageMetrics, StageStatus

log = logging.getLogger(__name__)

SIGNALS = ("entropy", "tfidf", "entities", "novelty", "structure")

_PLACEHOLDER = re.compile(r"<(?:TS|TIME|UUID|HEX|HASH|IP|EMAIL|ID|NUM|STR|PATH)>")


def analysis_text(chunk: Chunk) -> str:
    """The text a lexical signal should actually look at.

    For a log record this is its *template*, with placeholders stripped - not
    the raw line. Scoring `order=ORD-427039 user=usr_1234 status=200
    duration_ms=234` on raw text makes routine traffic look maximally
    information-dense: every volatile id is a unique term (so TF-IDF calls it
    rare) and reads as a number or identifier (so the entity counter calls it a
    fact). Those fields are precisely what stage 2 identified as *volatile*.
    Scoring the invariant part instead measures what the line actually says.
    """
    template = chunk.metadata.get("template")
    if template:
        return _PLACEHOLDER.sub(" ", template)
    return chunk.text


@dataclass
class DensityResult:
    chunks: list[Chunk]  # same objects, now carrying .density / .density_parts
    metrics: StageMetrics = field(default_factory=lambda: StageMetrics("density"))
    weights: dict[str, float] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)

    def ranked(self) -> list[Chunk]:
        """Chunks best-first. Ties break on document order for determinism."""
        return sorted(
            self.chunks, key=lambda c: (-(c.density or 0.0), c.order)
        )

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "unavailable_signals": self.unavailable,
        }


def _normalise(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1], clipping outliers at the 5th/95th percentile.

    A signal with no variance returns 0.5 for every chunk - neutral. Returning
    zeros (or an arbitrary ordering) would fabricate a ranking the data does not
    support.
    """
    if values.size == 0:
        return values
    if values.size < 8:
        low, high = float(values.min()), float(values.max())
    else:
        low, high = (float(x) for x in np.percentile(values, [5, 95]))
    if high - low < 1e-9:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


class DensityScorer:
    def __init__(
        self,
        cfg: Config | None = None,
        tokenizer: Tokenizer | None = None,
        entity_scorer: EntityScorer | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.tokenizer = tokenizer or get_tokenizer(self.cfg.tokenizer)
        density_cfg = self.cfg.density
        self.entity_scorer = entity_scorer or EntityScorer(
            density_cfg.spacy_model, density_cfg.spacy_max_chars
        )

    # -- public API --------------------------------------------------------
    def run(
        self, chunks: list[Chunk], embeddings: np.ndarray | None = None
    ) -> DensityResult:
        started = time.perf_counter()
        tokens = sum(c.token_count for c in chunks)
        metrics = StageMetrics(
            name="density",
            chunks_in=len(chunks),
            chunks_out=len(chunks),
            tokens_in=tokens,
            tokens_out=tokens,  # scoring never removes anything
        )
        if not chunks:
            metrics.status = StageStatus.SKIPPED
            metrics.note = "no chunks to score"
            return DensityResult(chunks=[], metrics=metrics)

        # Lexical signals read the invariant text (see `analysis_text`);
        # embeddings were computed on the raw text back in stage 3.
        texts = [analysis_text(c) for c in chunks]
        raw: dict[str, np.ndarray] = {}
        unavailable: list[str] = []
        details: dict = {}

        # --- signal 1: entropy ---
        raw["entropy"] = np.array(
            [self._entropy(text) for text in texts], dtype=np.float64
        )

        # --- signal 2: tf-idf ---
        tfidf, tfidf_note = self._tfidf(texts)
        if tfidf is None:
            unavailable.append("tfidf")
            details["tfidf_note"] = tfidf_note
        else:
            raw["tfidf"] = tfidf

        # --- signal 3: entities ---
        counts, backend = self.entity_scorer.count_all(texts)
        raw["entities"] = np.array(
            [
                count.weighted() / max(1, self.tokenizer.count(text))
                for count, text in zip(counts, texts)
            ],
            dtype=np.float64,
        )
        details["entity_backend"] = backend

        # --- signal 4: novelty ---
        if embeddings is None or len(embeddings) != len(chunks):
            unavailable.append("novelty")
            if embeddings is not None:
                log.warning(
                    "embeddings length %d != %d chunks; novelty skipped",
                    len(embeddings),
                    len(chunks),
                )
                details["novelty_note"] = "embedding/chunk count mismatch"
            else:
                details["novelty_note"] = "no embeddings supplied by stage 3"
        else:
            raw["novelty"] = self._novelty(embeddings)

        # --- signal 5: structure ---
        raw["structure"] = np.array(
            [self._structure(chunk) for chunk in chunks], dtype=np.float64
        )

        # --- combine ---
        weights = self._effective_weights(unavailable)
        # `structure` is deliberately NOT normalised. The other four signals are
        # unbounded and relative, so they need scaling to be comparable. The
        # structural prior is already an absolute, hand-designed [0, 1] scale -
        # and it is low-cardinality (a code file has ~3 distinct values). Running
        # percentile min-max over 3 values re-binarises it: with priors of
        # 0.60/0.65/0.70, everything at or below the 5th percentile collapses to
        # 0.0, so a block of pure constants scored 0.00 on structure while
        # scoring 1.00 on both entropy and entity density, and ranked 18/22.
        # Raising the prior did nothing, because normalisation just rescaled the
        # new value back to zero.
        normalised = {
            name: (
                np.clip(values, 0.0, 1.0) if name == "structure" else _normalise(values)
            )
            for name, values in raw.items()
        }
        frequency = self._frequency(chunks)
        boost = self.cfg.density.frequency_boost

        combined = np.zeros(len(chunks), dtype=np.float64)
        for name, weight in weights.items():
            combined += weight * normalised[name]
        combined = (combined + boost * frequency) / (1.0 + boost)

        protected = set(self.cfg.selection.protected_kinds)
        for index, chunk in enumerate(chunks):
            parts = {
                name: round(float(normalised[name][index]), 4) for name in normalised
            }
            parts["frequency"] = round(float(frequency[index]), 4)
            for missing in unavailable:
                parts[missing] = None
            chunk.density_parts = parts
            if chunk.kind in protected:
                # The user's own question is never a candidate for dropping;
                # scoring it below anything else would be meaningless.
                chunk.density = 1.0
                chunk.density_parts["protected"] = True
            else:
                chunk.density = round(float(combined[index]), 6)

        scores = np.array([c.density for c in chunks], dtype=np.float64)
        metrics.duration_ms = (time.perf_counter() - started) * 1000
        metrics.status = StageStatus.OK
        if unavailable:
            metrics.note = (
                f"signals unavailable: {', '.join(unavailable)}; "
                f"weights redistributed over {', '.join(weights)}"
            )
        metrics.details = {
            **details,
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "unavailable_signals": unavailable,
            "score_min": round(float(scores.min()), 4),
            "score_p50": round(float(np.median(scores)), 4),
            "score_max": round(float(scores.max()), 4),
            "frequency_boost": boost,
        }
        return DensityResult(
            chunks=chunks, metrics=metrics, weights=weights, unavailable=unavailable
        )

    # -- signals -----------------------------------------------------------
    def _entropy(self, text: str) -> float:
        """Shannon entropy over tokens, normalised to [0, 1] by chunk length.

        Normalising by log2(n) makes this "how repetitive is this chunk",
        independent of size - which is what matters when ranking for a
        per-token budget.

        The estimate is then damped by sample size. Normalised entropy is
        systematically inflated for short chunks: in 14 tokens almost every
        token is distinct, so the ratio pins to 1.0 regardless of content. That
        put a 14-token `class AuthError(Exception)` stub above the entire
        authentication flow. Damping by log2(n)/log2(reference) discounts an
        estimate drawn from too few samples to be trusted.
        """
        ids = self.tokenizer.token_ids(text)
        total = len(ids)
        if total < 2:
            return 0.0
        counts = Counter(ids)
        entropy = -sum(
            (n / total) * math.log2(n / total) for n in counts.values()
        )
        ceiling = math.log2(total)
        if ceiling <= 0:
            return 0.0
        normalised = entropy / ceiling

        reference = max(2, self.cfg.density.entropy_reference_tokens)
        confidence = min(1.0, math.log2(total) / math.log2(reference))
        return normalised * confidence

    def _tfidf(self, texts: list[str]) -> tuple[np.ndarray | None, str | None]:
        """Mean IDF of the distinct terms in each chunk."""
        if len(texts) < 2:
            return None, "fewer than two chunks; IDF is degenerate"
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as exc:  # pragma: no cover - depends on install
            return None, f"scikit-learn unavailable: {exc}"

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                sublinear_tf=True,
                norm=None,
                min_df=1,
                token_pattern=r"(?u)\b\w[\w.\-]*\b",
            )
            matrix = vectorizer.fit_transform(texts)
        except ValueError as exc:
            # Raised when every chunk is stop words / empty after tokenising.
            return None, f"vectoriser found no usable vocabulary: {exc}"

        scores = np.zeros(len(texts), dtype=np.float64)
        matrix = matrix.tocsr()
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            values = matrix.data[start:end]
            scores[row] = float(values.mean()) if values.size else 0.0
        return scores, None

    @staticmethod
    def _novelty(embeddings: np.ndarray) -> np.ndarray:
        """Cosine distance from the corpus centroid.

        Vectors arrive L2-normalised from stage 3, so a dot product against the
        renormalised centroid is the cosine similarity; 1 - that is the distance.
        """
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-9:
            return np.full(len(embeddings), 0.5, dtype=np.float64)
        centroid = centroid / norm
        similarity = embeddings @ centroid
        return 1.0 - similarity.astype(np.float64)

    def _structure(self, chunk: Chunk) -> float:
        cfg = self.cfg.density
        # str() defensively: metadata keys are populated by several chunkers.
        level = str(chunk.metadata.get("level") or "").upper()
        if level and level in cfg.level_priors:
            base = cfg.level_priors[level]
        else:
            base = cfg.structure_priors.get(
                chunk.kind, cfg.structure_priors.get("default", 0.4)
            )
        # The failure vocabulary describes *events*. In source code the same
        # words are type names and documentation - `class AuthError(Exception)`
        # is boilerplate, not an incident - and boosting on them ranked 14-token
        # exception stubs above the entire authentication flow.
        if chunk.kind not in set(cfg.keyword_boost_excluded_kinds):
            lowered = chunk.text.lower()
            # Graded, not binary: one keyword earns half the boost, two or more
            # earn all of it. A single on/off jump made the signal cluster at
            # two values, and normalisation then amplified the gap into the
            # dominant term - ranking a substantive section second-to-last
            # purely for lacking a magic word.
            matches = sum(1 for keyword in cfg.keywords if keyword in lowered)
            if matches:
                base += cfg.keyword_boost * min(1.0, matches / 2.0)
        return min(1.0, base)

    @staticmethod
    def _frequency(chunks: list[Chunk]) -> np.ndarray:
        """Log-scaled stage-3 duplicate count, normalised to [0, 1]."""
        counts = np.array(
            [max(1, c.duplicate_count) for c in chunks], dtype=np.float64
        )
        scaled = np.log1p(counts - 1.0)
        peak = float(scaled.max())
        if peak < 1e-9:
            return np.zeros(len(chunks), dtype=np.float64)
        return scaled / peak

    def _effective_weights(self, unavailable: list[str]) -> dict[str, float]:
        """Configured weights, renormalised over the signals we actually have."""
        configured = self.cfg.density.weights.normalised()
        usable = {
            name: weight
            for name, weight in configured.items()
            if name not in unavailable
        }
        total = sum(usable.values())
        if total <= 0:
            # Every weighted signal is missing; fall back to a uniform blend of
            # whatever is left rather than returning an all-zero ranking.
            return {name: 1.0 / len(usable) for name in usable} if usable else {}
        return {name: weight / total for name, weight in usable.items()}


def score_density(
    chunks: list[Chunk],
    embeddings: np.ndarray | None = None,
    cfg: Config | None = None,
    tokenizer: Tokenizer | None = None,
    entity_scorer: EntityScorer | None = None,
) -> DensityResult:
    """Convenience wrapper around :class:`DensityScorer`."""
    return DensityScorer(cfg, tokenizer, entity_scorer).run(chunks, embeddings)
