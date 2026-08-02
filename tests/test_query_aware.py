"""Query-aware selection and the marker escape hatch.

Two features that share a theme: the compressor should know what the output is
*for*, and should be able to give back what it removed.

The load-bearing assertion is that a question actually changes what survives.
A signal that is computed, weighted and reported but does not move the output
is decoration, so it is tested by comparing two compressions of the same text
under two different questions.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.config import get_config
from engine.density import DensityScorer
from engine.pipeline import CompressionPipeline
from engine.types import Chunk, ChunkKind


def _chunk(text: str, order: int) -> Chunk:
    return Chunk(
        text=text, kind=ChunkKind.PARAGRAPH, source="t", order=order,
        start_line=order, end_line=order, token_count=10, id=f"t#{order}",
    )


# ---------------------------------------------------------------------------
# The signal itself
# ---------------------------------------------------------------------------
def test_no_query_leaves_scoring_exactly_as_it_was(config):
    """Backward compatibility, asserted rather than assumed.

    The five original weights must renormalise to their original values when
    query_relevance is absent, or every previously-reported number silently
    changed meaning.
    """
    chunks = [_chunk(f"paragraph {i}", i) for i in range(3)]
    result = DensityScorer(config).run(chunks, np.random.rand(3, 8).astype("float32"))

    assert "query_relevance" in result.unavailable
    assert result.weights == pytest.approx(
        {"entropy": 0.15, "tfidf": 0.25, "entities": 0.15,
         "novelty": 0.20, "structure": 0.25},
        abs=1e-6,
    )


def test_a_query_adds_the_signal_and_reweights(config):
    chunks = [_chunk(f"paragraph {i}", i) for i in range(3)]
    embeddings = np.random.rand(3, 8).astype("float32")
    result = DensityScorer(config).run(chunks, embeddings, embeddings[0])

    assert "query_relevance" not in result.unavailable
    assert result.weights["query_relevance"] > 0
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_relevance_scores_similar_chunks_higher(config):
    """The signal must be monotonic in similarity, not merely present."""
    scorer = DensityScorer(config)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    scores = scorer._query_relevance(embeddings, query)
    assert scores[0] > scores[1] > scores[2]
    assert 0.0 <= scores.min() and scores.max() <= 1.0


def test_a_mismatched_query_dimension_is_dropped_not_crashed(config):
    """Two providers, two vector spaces - a dot product there is a crash."""
    chunks = [_chunk(f"paragraph {i}", i) for i in range(3)]
    result = DensityScorer(config).run(
        chunks,
        np.random.rand(3, 1024).astype("float32"),
        np.random.rand(3072).astype("float32"),
    )
    assert "query_relevance" in result.unavailable
    assert "3072" in result.metrics.details["query_relevance_note"]


# ---------------------------------------------------------------------------
# End to end: does the question change the output?
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipe():
    return CompressionPipeline()


def test_different_questions_produce_different_compressions(pipe, markdown_source):
    """The whole claim. If this fails the signal is decoration."""
    a = pipe.compress(markdown_source, "p.md", budget_ratio=0.25,
                      fast_mode=True, query="What was the connection pool size?")
    b = pipe.compress(markdown_source, "p.md", budget_ratio=0.25,
                      fast_mode=True, query="How long did the rollback take?")
    assert a.compressed_text != b.compressed_text


def test_a_query_is_never_dropped_from_its_own_compression(pipe, markdown_source):
    question = "What was the connection pool size and the gateway timeout?"
    result = pipe.compress(markdown_source, "p.md", budget_ratio=0.15,
                           fast_mode=True, query=question)
    assert question in result.compressed_text


def test_query_relevance_is_reported_in_the_stage_details(pipe, markdown_source):
    result = pipe.compress(markdown_source, "p.md", budget_ratio=0.30,
                           fast_mode=True, query="pool size")
    weights = result.stage("density").details["weights"]
    assert "query_relevance" in weights


def test_a_queryless_run_reports_why_the_signal_is_absent(pipe, markdown_source):
    result = pipe.compress(markdown_source, "p.md", budget_ratio=0.30, fast_mode=True)
    details = result.stage("density").details
    assert "query_relevance" in details["unavailable_signals"]
    assert "no query" in details["query_relevance_note"]


def test_query_aware_keeps_more_of_what_was_asked_about(pipe, corpus):
    """Directionally the point: ask about a fact, get that fact kept."""
    source = (corpus / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")
    blind = pipe.compress(source, "p.md", budget_ratio=0.25, fast_mode=True)
    aware = pipe.compress(source, "p.md", budget_ratio=0.25, fast_mode=True,
                          query="How long did the rollback take and in how many regions?")
    assert "regions" in aware.compressed_text
    assert "regions" not in blind.compressed_text


# ---------------------------------------------------------------------------
# The escape hatch
# ---------------------------------------------------------------------------
def test_every_marker_is_addressable(pipe, python_source):
    result = pipe.compress(python_source, "auth_service.py",
                           budget_ratio=0.30, fast_mode=True)
    assert result.markers, "a 30% budget must drop something"
    for marker in result.markers:
        assert marker["id"]
        assert f"#{marker['id']}" in result.compressed_text
        assert marker["kind"] in {"dropped", "collapsed"}
        assert marker["tokens"] > 0


def test_marker_ids_are_unique(pipe, python_source):
    result = pipe.compress(python_source, "auth_service.py",
                           budget_ratio=0.30, fast_mode=True)
    ids = [m["id"] for m in result.markers]
    assert len(ids) == len(set(ids))


def test_markers_record_the_chunks_they_hide(pipe, python_source):
    result = pipe.compress(python_source, "auth_service.py",
                           budget_ratio=0.30, fast_mode=True)
    kept = {c.id for c in result.kept}
    for marker in result.markers:
        assert marker["chunk_ids"], f"marker {marker['id']} recovers nothing"
        if marker["kind"] == "dropped":
            assert not (set(marker["chunk_ids"]) & kept), "a kept chunk is not omitted"


def test_the_api_payload_omits_chunk_ids(pipe, python_source):
    """They run to thousands of strings on a log - exactly the payload weight
    the spans design exists to remove."""
    result = pipe.compress(python_source, "auth_service.py",
                           budget_ratio=0.30, fast_mode=True)
    for marker in result.to_dict()["markers"]:
        assert "chunk_ids" not in marker
        assert marker["id"] and marker["tokens"]


# ---------------------------------------------------------------------------
# Per-provider similarity thresholds
# ---------------------------------------------------------------------------
def test_a_calibrated_provider_uses_its_own_threshold(config):
    from engine.redundancy import RedundancyDetector

    detector = RedundancyDetector(config)
    assert detector._threshold_for("local") == (0.88, True)
    assert detector._threshold_for("cohere") == (0.84, True)


def test_an_uncalibrated_provider_falls_back_and_is_flagged(config):
    """The failure mode is silent, so the flag is the whole point.

    An uncalibrated threshold does not error - it just collapses nothing, or
    too much. The stage has to say which case it is in.
    """
    from engine.redundancy import RedundancyDetector

    threshold, calibrated = RedundancyDetector(config)._threshold_for("gemini")
    assert threshold == config.redundancy.similarity_threshold
    assert calibrated is False


def test_an_unknown_provider_never_silently_uses_a_calibrated_value(config):
    from engine.redundancy import RedundancyDetector

    _, calibrated = RedundancyDetector(config)._threshold_for("some-new-vendor")
    assert calibrated is False


def test_the_stage_reports_the_threshold_it_actually_applied(pipe, corpus):
    source = (corpus / "support" / "tickets.txt").read_text(encoding="utf-8")
    details = pipe.compress(
        source, "tickets.txt", budget_ratio=0.30, fast_mode=True
    ).stage("redundancy").details

    assert "similarity_threshold" in details
    assert "similarity_threshold_calibrated" in details
    provider = details.get("embedding_provider")
    if provider in config_thresholds():
        assert details["similarity_threshold"] == config_thresholds()[provider]
        assert details["similarity_threshold_calibrated"] is True


def config_thresholds() -> dict:
    from engine.config import get_config

    return get_config().redundancy.similarity_threshold_by_provider


def test_an_uncalibrated_run_says_so_in_the_stage_note(config, corpus):
    """A judge reading the accordion should see the caveat, not infer it."""
    from engine.pipeline import CompressionPipeline

    # model_copy, not with_overrides: the latter deep-merges, so an empty dict
    # merges into the existing map and changes nothing.
    tweaked = config.model_copy(
        update={
            "redundancy": config.redundancy.model_copy(
                update={"similarity_threshold_by_provider": {}}
            )
        }
    )
    source = (corpus / "support" / "tickets.txt").read_text(encoding="utf-8")
    stage = CompressionPipeline(tweaked).compress(
        source, "tickets.txt", budget_ratio=0.30, fast_mode=True
    ).stage("redundancy")

    assert stage.details["similarity_threshold_calibrated"] is False
    assert "not calibrated" in (stage.note or "")


def test_markers_are_absent_when_disabled(config, python_source):
    disabled = config.with_overrides(
        {"reconstruction": {"drop_markers": False, "cluster_markers": False}}
    )
    result = CompressionPipeline(disabled).compress(
        python_source, "auth_service.py", budget_ratio=0.30, fast_mode=True
    )
    assert result.markers == []
