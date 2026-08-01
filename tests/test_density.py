"""Stage 4 tests.

The important ones here are *ranking* tests: they assert that content a judge
would ask about ends up near the top, because a density score that is merely
well-formed but ranks routine DEBUG noise above a stack trace is worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.chunker import chunk_document, chunk_file
from engine.config import get_config
from engine.density import DensityScorer, _normalise, analysis_text, score_density
from engine.embeddings import EmbeddingModel
from engine.entities import EntityScorer, regex_counts
from engine.redundancy import RedundancyDetector
from engine.types import Chunk, ChunkKind, StageStatus


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingModel(get_config().redundancy)


def _pipeline(path, embedder):
    """Stages 2 -> 3 -> 4, the way the real pipeline runs them."""
    chunks = chunk_file(path)
    redundancy = RedundancyDetector(embedder=embedder).run(chunks)
    return DensityScorer().run(redundancy.chunks, redundancy.embeddings)


@pytest.fixture(scope="module")
def log_density(corpus, embedder):
    return _pipeline(corpus / "logs" / "checkout_service.log", embedder)


@pytest.fixture(scope="module")
def code_density(corpus, embedder):
    return _pipeline(corpus / "code" / "auth_service.py", embedder)


def _rank_of(result, needle: str) -> float:
    """Best percentile rank of any chunk containing `needle` (1.0 = top)."""
    ranked = result.ranked()
    for index, chunk in enumerate(ranked):
        if needle.lower() in chunk.text.lower():
            return 1.0 - index / len(ranked)
    raise AssertionError(f"{needle!r} not found in any chunk")


def _make(text: str, order: int, kind: str = ChunkKind.LOG_RECORD, **kw) -> Chunk:
    metadata = kw.pop("metadata", {})
    return Chunk(
        text=text,
        kind=kind,
        source="test",
        order=order,
        start_line=order + 1,
        end_line=order + 1,
        token_count=kw.pop("token_count", max(1, len(text) // 4)),
        id=kw.pop("id", f"test#{order:04d}"),
        metadata=metadata,
        **kw,
    )


# ---------------------------------------------------------------------------
# Ranking quality - the point of the whole stage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "evidence",
    [
        "PAYMENT_POOL_SIZE resolved to 8",
        "Traceback (most recent call last)",
        "connection pool wait exceeded threshold",
        "gateway timeout after 8000ms",
        "rollback initiated",
    ],
)
def test_incident_critical_content_ranks_in_the_top_decile(log_density, evidence):
    """Calibration regression.

    With the original weights (entropy/tfidf/entities .25 each, structure .10,
    frequency_boost .15) routine INFO cache-hit lines took the top four ranks
    while the stack trace fell to 90/146 and the root-cause WARN to 78/146.
    """
    percentile = _rank_of(log_density, evidence)
    assert percentile >= 0.88, (
        f"{evidence!r} ranked at the {percentile:.0%} percentile; "
        f"stage 5 would drop the evidence before the noise"
    )


def test_routine_telemetry_ranks_at_the_bottom(log_density):
    ranked = log_density.ranked()
    bottom_quartile = ranked[int(len(ranked) * 0.75) :]
    debug_share = sum(
        1 for c in bottom_quartile if c.metadata.get("level") in {"DEBUG", "INFO"}
    ) / len(bottom_quartile)
    assert debug_share > 0.8, "the low end should be routine INFO/DEBUG traffic"


def test_severity_outranks_frequency(embedder):
    """Regression: a 400x routine line used to outrank a one-off ERROR.

    Frequency is worth acknowledging (the representative stands in for 400
    events) but it must never dominate severity.
    """
    routine = _make(
        "2024-03-14 09:00:01.000 INFO cart-store [eu-west-1] cache hit "
        "key=cart:ORD-100000 ttl_remaining_s=42",
        0,
        metadata={"level": "INFO", "template": "<TS> INFO cart-store cache hit key=cart:<ID>"},
    )
    routine.duplicate_count = 400
    critical = _make(
        "2024-03-14 09:07:52.869 ERROR checkout-api [eu-west-1] gateway timeout "
        "after 8000ms order=ORD-427039 attempt=3",
        1,
        metadata={"level": "ERROR", "template": "<TS> ERROR checkout-api gateway timeout after <NUM> order=<ID>"},
    )
    filler = [
        _make(f"2024-03-14 09:0{i}:00.000 DEBUG inventory [us-east-1] emitted metric v={i}",
              i + 2, metadata={"level": "DEBUG", "template": "<TS> DEBUG inventory emitted metric v=<NUM>"})
        for i in range(8)
    ]
    result = score_density([routine, critical, *filler])
    assert critical.density > routine.density, (
        f"routine x400 line ({routine.density:.3f}) outranked a one-off ERROR "
        f"({critical.density:.3f})"
    )


def test_code_ranks_real_logic_above_trivial_stubs(code_density):
    """Regression: 14-token exception stubs used to take ranks 1 and 2."""
    logic = _rank_of(code_density, "def hash_password")
    stub = _rank_of(code_density, "class TokenRevokedError")
    assert logic > stub, "a one-line exception stub outranked password hashing"


# ---------------------------------------------------------------------------
# Signal-level behaviour
# ---------------------------------------------------------------------------
def test_entropy_damping_discounts_tiny_chunks():
    """A short all-distinct chunk must not score a perfect 1.0 on entropy."""
    scorer = DensityScorer()
    short = scorer._entropy("class AuthError(Exception): pass")
    long_text = " ".join(f"unique_term_{i}" for i in range(80))
    assert scorer._entropy(long_text) > short
    assert short < 1.0


def test_entropy_penalises_repetition():
    scorer = DensityScorer()
    varied = " ".join(f"term{i}" for i in range(60))
    repeated = "the same thing " * 30
    assert scorer._entropy(varied) > scorer._entropy(repeated)


def test_analysis_text_uses_the_log_template():
    """Volatile ids must not be scored as rare, information-dense terms."""
    chunk = _make(
        "2024-03-14 09:00:01.000 INFO api handled POST /v1/checkout order=ORD-427039",
        0,
        metadata={"template": "<TS> INFO api handled POST /v1/checkout order=<ID>"},
    )
    text = analysis_text(chunk)
    assert "ORD-427039" not in text
    assert "<ID>" not in text and "<TS>" not in text
    assert "handled POST /v1/checkout" in text


def test_analysis_text_passes_through_non_log_chunks():
    chunk = _make("A paragraph with no template.", 0, ChunkKind.PARAGRAPH)
    assert analysis_text(chunk) == chunk.text


def test_keyword_boost_does_not_apply_to_code():
    """`class AuthError(Exception)` is boilerplate, not an incident."""
    scorer = DensityScorer()
    code = _make("class AuthError(Exception):\n    '''auth failure'''", 0, ChunkKind.CLASS)
    prose = _make("The deployment failed with a gateway timeout.", 1, ChunkKind.PARAGRAPH)

    priors = get_config().density.structure_priors
    assert scorer._structure(code) == pytest.approx(priors["class"])
    assert scorer._structure(prose) > priors["paragraph"]


def test_structure_uses_log_severity(embedder):
    scorer = DensityScorer()
    error = _make("boom", 0, metadata={"level": "ERROR"})
    debug = _make("tick", 1, metadata={"level": "DEBUG"})
    assert scorer._structure(error) > scorer._structure(debug)


def test_markdown_heading_level_is_not_read_as_log_severity(markdown_source):
    """Regression: headings stored an int under `level` and crashed the scorer."""
    chunks = chunk_document(markdown_source, "doc.md")
    headings = [c for c in chunks if "heading_level" in c.metadata]
    assert headings, "expected at least one heading annotation"
    for chunk in chunks:
        assert not isinstance(chunk.metadata.get("level"), int)

    result = score_density(chunks)  # must not raise
    assert all(c.density is not None for c in result.chunks)


def test_novelty_is_distance_from_the_corpus_centroid():
    """Documented deviation: cluster-centroid distance is 0 for singletons."""
    vectors = np.array(
        [[1.0, 0.0], [0.99, 0.14], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    novelty = DensityScorer._novelty(vectors)
    assert novelty[3] > novelty[0], "the outlier should be the most novel"


def test_normalise_returns_neutral_for_a_constant_signal():
    """A signal with no variance must not fabricate a ranking."""
    assert np.allclose(_normalise(np.full(20, 3.7)), 0.5)


def test_normalise_clips_outliers():
    values = np.array([0.0] * 20 + [1000.0], dtype=np.float64)
    scaled = _normalise(values)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0


# ---------------------------------------------------------------------------
# Explainability and invariants
# ---------------------------------------------------------------------------
def test_every_score_is_explainable(log_density):
    for chunk in log_density.chunks:
        assert chunk.density is not None
        assert 0.0 <= chunk.density <= 1.0
        for signal in ("entropy", "tfidf", "entities", "novelty", "structure", "frequency"):
            assert signal in chunk.density_parts, f"{signal} missing from breakdown"


def test_scoring_removes_nothing(log_density):
    metrics = log_density.metrics
    assert metrics.chunks_in == metrics.chunks_out
    assert metrics.tokens_in == metrics.tokens_out


def test_weights_sum_to_one(log_density):
    assert sum(log_density.weights.values()) == pytest.approx(1.0)


def test_ranking_is_deterministic(corpus, embedder):
    first = _pipeline(corpus / "support" / "tickets.txt", embedder)
    order_a = [c.id for c in first.ranked()]
    second = _pipeline(corpus / "support" / "tickets.txt", embedder)
    assert order_a == [c.id for c in second.ranked()]


def test_protected_chunks_score_maximum():
    chunks = [
        _make("Some ordinary context paragraph.", 0, ChunkKind.PARAGRAPH),
        _make("What caused the outage?", 1, ChunkKind.QUERY),
        _make("Answer using only the context.", 2, ChunkKind.INSTRUCTION),
    ]
    result = score_density(chunks)
    for chunk in result.chunks:
        if chunk.kind in {ChunkKind.QUERY, ChunkKind.INSTRUCTION}:
            assert chunk.density == 1.0
            assert chunk.density_parts.get("protected") is True
    assert result.ranked()[0].kind in {ChunkKind.QUERY, ChunkKind.INSTRUCTION}


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------
def test_missing_embeddings_redistribute_the_novelty_weight(corpus):
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    result = score_density(chunks, embeddings=None)

    assert "novelty" in result.unavailable
    assert "novelty" not in result.weights
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert all(c.density_parts["novelty"] is None for c in result.chunks)
    assert "novelty" in (result.metrics.note or "")


def test_mismatched_embeddings_are_refused_not_misused(corpus):
    """Scoring chunks against the wrong vectors is worse than not scoring."""
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    wrong = np.zeros((len(chunks) - 3, 384), dtype=np.float32)
    result = score_density(chunks, embeddings=wrong)
    assert "novelty" in result.unavailable


def test_scoring_works_without_spacy(corpus):
    """The regex proxies must carry the entity signal on their own."""
    chunks = chunk_file(corpus / "logs" / "checkout_service.log")[:60]
    dead = EntityScorer(model="definitely-not-a-model")
    result = DensityScorer(entity_scorer=dead).run(chunks)

    assert result.metrics.details["entity_backend"].startswith("regex")
    assert all(c.density is not None for c in result.chunks)


def test_regex_counts_find_code_and_numbers():
    counts = regex_counts("def validate_username(x): return len(x) > 64")
    assert counts.signatures >= 1
    assert counts.numbers >= 1
    assert counts.identifiers >= 1


def test_oversized_input_falls_back_to_regex(corpus):
    chunks = chunk_file(corpus / "logs" / "checkout_service.log")
    scorer = EntityScorer(max_chars=100)
    counts, backend = scorer.count_all([c.text for c in chunks[:20]])
    assert backend == "regex:oversized"
    assert len(counts) == 20


def test_empty_input():
    result = score_density([])
    assert result.chunks == []
    assert result.metrics.status == StageStatus.SKIPPED


def test_single_chunk_scores_without_tfidf():
    """IDF is degenerate with one document; the stage must degrade, not crash."""
    result = score_density([_make("only one chunk here", 0, ChunkKind.PARAGRAPH)])
    assert "tfidf" in result.unavailable
    assert result.chunks[0].density is not None
