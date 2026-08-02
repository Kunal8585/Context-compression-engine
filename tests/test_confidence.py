"""Confidence score: does it mean anything, and does it stay honest?

Two kinds of test here. The first kind pins the mechanics - a score built from
ratios should move in the obvious direction when you change the input. The
second kind is the one that matters: the score must **rank** the four sample
contexts the same way the eval harness's measured fact survival does. A
well-formed number that does not predict anything is worse than no number,
because it invites trust it has not earned.

The regression at the centre of this file: lexical retention must be measured
against the stage-3 survivors, not the raw input. Scored against the raw input,
the 2,400-record log retains 7.7% of its distinct numbers - they are timestamp
fractions that dedup collapses on purpose - and the context with *perfect*
measured fact survival scores lowest of the four.
"""

from __future__ import annotations

import itertools

import pytest

from engine.confidence import (
    CALIBRATION,
    HIGH,
    MODERATE,
    WEIGHTS,
    Confidence,
    score_compression,
)
from engine.pipeline import CompressionPipeline
from engine.types import Chunk, ChunkKind


def _chunk(text: str, order: int = 0, density: float = 0.5, tokens: int = 20) -> Chunk:
    return Chunk(
        text=text, kind=ChunkKind.PARAGRAPH, source="test", order=order,
        start_line=order + 1, end_line=order + 1, token_count=tokens,
        id=f"test#{order:04d}", density=density,
    )


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------
def test_weights_sum_to_one():
    """Otherwise the score is not on the [0, 1] scale the bands assume."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_a_lossless_compression_scores_at_the_top():
    text = "Pool size 8, timeout 8000ms, service payments.internal"
    chunks = [_chunk(text)]
    result = score_compression(text, text, chunks, chunks, survivors=chunks)
    assert result.score == pytest.approx(1.0)
    assert result.band == "high"


def test_dropping_numbers_lowers_the_score_and_names_them():
    original = "Pool size was 8 and the gateway timeout was 8000ms at 10:41"
    compressed = "Pool size was 8"
    chunks = [_chunk(original)]
    result = score_compression(original, compressed, chunks, chunks[:1], survivors=chunks)

    assert result.components["number_retention"] < 1.0
    reasons = " ".join(result.reasons)
    assert "numbers" in reasons
    assert "8000" in reasons or "41" in reasons, "a reason must name what was lost"


def test_empty_input_is_unknown_not_zero():
    """Zero would read as 'very bad'; there is simply nothing to judge."""
    result = score_compression("", "", [], [])
    assert result.band == "unknown"
    assert result.score == 0.0


def test_score_is_bounded():
    original = "alpha 1 beta 2 gamma 3"
    for compressed in (original, "alpha 1", "", "entirely unrelated text"):
        result = score_compression(original, compressed, [_chunk(original)], [])
        assert 0.0 <= result.score <= 1.0


def test_an_input_with_no_numbers_is_not_penalised():
    """Nothing could be lost, so nothing was - retention is 1.0, not 0.0."""
    text = "some prose with no digits at all in it whatsoever"
    chunks = [_chunk(text)]
    result = score_compression(text, text, chunks, chunks, survivors=chunks)
    assert result.components["number_retention"] == 1.0


def test_density_retention_is_token_weighted():
    """Dropping one large dense chunk must hurt more than one small one."""
    big = _chunk("big chunk " * 40, order=0, density=0.9, tokens=200)
    small = _chunk("small", order=1, density=0.9, tokens=5)
    chunks = [big, small]

    dropped_big = score_compression("x 1", "x 1", chunks, [small], survivors=chunks)
    dropped_small = score_compression("x 1", "x 1", chunks, [big], survivors=chunks)
    assert dropped_big.components["density_retention"] < dropped_small.components["density_retention"]


def test_redundancy_removal_counts_as_more_lossless_than_budget_eviction():
    text = "alpha 1"
    chunks = [_chunk(text)]
    dedup = score_compression(text, text, chunks, chunks, survivors=chunks,
                              tokens_removed_by_redundancy=900, tokens_removed_total=1000)
    budget = score_compression(text, text, chunks, chunks, survivors=chunks,
                               tokens_removed_by_redundancy=0, tokens_removed_total=1000)
    assert dedup.components["lossless_share"] > budget.components["lossless_share"]
    assert dedup.score > budget.score


def test_unrepaired_dependencies_lower_the_score():
    text = "alpha 1"
    chunks = [_chunk(text)]
    clean = score_compression(text, text, chunks, chunks, survivors=chunks,
                              broken_dependencies=0, repaired_dependencies=4)
    broken = score_compression(text, text, chunks, chunks, survivors=chunks,
                               broken_dependencies=4, repaired_dependencies=0)
    assert broken.components["dependency_integrity"] < clean.components["dependency_integrity"]
    assert any("references definitions" in r for r in broken.reasons)


def test_the_payload_never_claims_to_be_a_guarantee():
    text = "alpha 1"
    chunks = [_chunk(text)]
    payload = score_compression(text, text, chunks, chunks, survivors=chunks).to_dict()
    assert "does not guarantee" in payload["method"]
    assert set(payload) >= {"score", "band", "components", "reasons", "evidence"}


# ---------------------------------------------------------------------------
# The regression: volatile fields are not losses
# ---------------------------------------------------------------------------
def test_collapsed_duplicates_do_not_count_against_the_score():
    """Stage 3 removing 400 identical records is not information loss.

    The representative and its count marker survive, so the fact of the
    repetition is preserved. Measuring retention against the raw input instead
    of the survivors makes a perfect log compression look catastrophic.
    """
    original = "\n".join(f"09:00:0{i % 10}.{i:03d} INFO cache hit" for i in range(200))
    survivors = [_chunk("09:00:00.001 INFO cache hit")]
    compressed = "09:00:00.001 INFO cache hit\n[... 199 identical records collapsed ...]"

    against_survivors = score_compression(
        original, compressed, survivors, survivors, survivors=survivors
    )
    against_raw = score_compression(original, compressed, survivors, survivors)

    assert against_survivors.components["number_retention"] > 0.9
    assert against_raw.components["number_retention"] < 0.2
    assert against_survivors.score > against_raw.score


# ---------------------------------------------------------------------------
# The test that matters: does the score predict measured fact survival?
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored(corpus):
    pipeline = CompressionPipeline()
    files = {
        "log_incident": corpus / "logs" / "checkout_service.log",
        "auth_code": corpus / "code" / "auth_service.py",
        "tickets": corpus / "support" / "tickets.txt",
        "postmortem": corpus / "docs" / "incident_postmortem.md",
    }
    out = {}
    for key, path in files.items():
        result = pipeline.compress(
            path.read_text(encoding="utf-8"), path.name,
            budget_ratio=0.30, fast_mode=True,
        )
        out[key] = result.confidence
    return out


def test_every_compression_gets_a_confidence(scored):
    for key, confidence in scored.items():
        assert isinstance(confidence, Confidence), key
        assert 0.0 <= confidence.score <= 1.0
        assert confidence.band in {"high", "moderate", "low"}
        assert confidence.reasons, "a score with no explanation is not auditable"


def test_the_score_ranks_contexts_the_way_measured_fact_survival_does(scored):
    """The load-bearing assertion. If this fails the number is decoration."""
    rows = [(key, CALIBRATION[key], scored[key].score) for key in CALIBRATION]
    pairs = list(itertools.combinations(rows, 2))
    concordant = [
        (a[0], b[0]) for a, b in pairs if (a[1] - b[1]) * (a[2] - b[2]) > 0
    ]
    assert len(concordant) >= 5, (
        "confidence must rank inputs like fact survival does; got "
        f"{len(concordant)}/{len(pairs)} concordant pairs: "
        + ", ".join(f"{k}={s:.3f}" for k, _, s in rows)
    )


def test_the_redundant_log_scores_highest_and_the_dense_prose_lowest(scored):
    """The two ends of the ranking, asserted by name.

    A 2,400-record log compresses 93.9% at essentially no cost; a postmortem
    where every paragraph states a different fact cannot lose 70% without
    losing something. The score has to say so.
    """
    assert scored["log_incident"].score > scored["postmortem"].score
    assert scored["log_incident"].band == "high"
    assert scored["postmortem"].band in {"low", "moderate"}


def test_bands_are_not_all_the_same(scored):
    """A score that says one thing about every input carries no information."""
    assert len({c.band for c in scored.values()}) >= 2


def test_thresholds_are_ordered():
    assert 0 < MODERATE < HIGH < 1
