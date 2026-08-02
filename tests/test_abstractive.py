"""Stage 6 tests.

Stage 6 is the only stage that rewrites text, so it is the only one that can
invent a fact. These tests are mostly about what it *refuses* to do. They use a
scripted provider rather than a live model so they are fast and deterministic;
the real-model behaviour is measured separately and recorded in the README.

Since the provider migration the scripted client is a real
:class:`~engine.abstractive.GenerationClient` over a real ``GenerationChain``,
with only the HTTP call replaced - so these still exercise the shipped path.
The safety checks below are deliberately unchanged by that migration: a cloud
model can drop a fact in a paraphrase exactly like a local one can, and the net
that catches it is provider-agnostic.
"""

from __future__ import annotations

import time

import pytest

from engine.abstractive import (
    AbstractiveCompressor,
    GenerationClient,
    _strip_fences,
    critical_tokens,
    numbers_in,
)
from engine.config import get_config
from engine.types import Chunk, ChunkKind, StageStatus
from provider_doubles import ScriptedClient

LONG = 200  # comfortably above min_tokens_to_compress


def _chunk(text: str, order: int = 0, tokens: int = LONG, kind=ChunkKind.PARAGRAPH) -> Chunk:
    return Chunk(
        text=text, kind=kind, source="test", order=order,
        start_line=order + 1, end_line=order + 1, token_count=tokens,
        id=f"test#{order:04d}", selected=True,
    )


def _compressor(client, **overrides):
    cfg = get_config()
    if overrides:
        cfg = cfg.with_overrides({"abstractive": overrides})
    return AbstractiveCompressor(cfg, client=client)


# ---------------------------------------------------------------------------
# The safety net - what it refuses
# ---------------------------------------------------------------------------
def test_a_paraphrase_that_drops_a_number_is_rejected():
    """64 must never silently become 128, and 10:41 must not vanish."""
    original = (
        "The pool size was reduced from 64 to 8 connections at 10:41, and the "
        "request timeout stayed at 8000ms throughout the incident window."
    )
    chunk = _chunk(original)
    client = ScriptedClient({original: "Pool size dropped from 64 to 8. Timeout unchanged."})
    result = _compressor(client).run([chunk])

    assert result.accepted == 0
    assert chunk.compressed_text is None, "the original must be kept intact"
    assert result.rejected[0].reason == "numbers_lost"
    assert result.metrics.details["rejections_by_reason"] == {"numbers_lost": 1}


def test_a_paraphrase_that_drops_identifiers_is_rejected():
    original = (
        "The PAYMENT_POOL_SIZE default moved into config/defaults.yaml and the "
        "checkout_service handler began raising ReadTimeout on every charge "
        "attempt against payments.internal for order 551884."
    )
    chunk = _chunk(original)
    client = ScriptedClient({original: "A config default changed and requests began failing for order 551884."})
    result = _compressor(client).run([chunk])

    assert result.accepted == 0
    assert result.rejected[0].reason in {"entity_overlap", "numbers_lost"}


def test_a_faithful_paraphrase_is_accepted():
    original = (
        "It is worth noting that, in the opinion of the team, the rollback "
        "tooling did in fact work exactly as it was designed to work, and it "
        "completed in 22 minutes across 3 regions without any incident."
    )
    shorter = "Rollback tooling worked as designed, completing in 22 minutes across 3 regions."
    chunk = _chunk(original)
    client = ScriptedClient({original: shorter})
    result = _compressor(client).run([chunk])

    assert result.accepted == 1
    assert chunk.compressed_text == shorter
    assert chunk.output_text == shorter
    assert result.tokens_saved > 0


def test_a_paraphrase_that_is_not_shorter_is_rejected():
    """The stage is monotonic: it may only ever reduce tokens."""
    original = "Short original text with number 42."
    chunk = _chunk(original, tokens=10)
    client = ScriptedClient({original: original + " " * 5 + "Plus considerably more words added, 42."})
    result = _compressor(client, min_tokens_to_compress=1).run([chunk])

    assert result.accepted == 0
    assert result.rejected[0].reason == "no_gain"


def test_overlap_threshold_is_configurable():
    original = "Service alpha_one and beta_two and gamma_three all failed at 500."
    lossy = "Service alpha_one failed at 500."
    chunk = _chunk(original)

    strict = _compressor(ScriptedClient({original: lossy})).run([_chunk(original)])
    assert strict.accepted == 0

    lenient = _compressor(
        ScriptedClient({original: lossy}), entity_overlap_threshold=0.1
    ).run([chunk])
    assert lenient.accepted == 1


def test_number_check_can_be_disabled():
    original = "Latency was 8400 ms across 3 regions and 14200 checkouts."
    chunk = _chunk(original)
    client = ScriptedClient({original: "Latency was high across regions."})
    result = _compressor(
        client, require_numbers_preserved=False, entity_overlap_threshold=0.0
    ).run([chunk])
    assert result.accepted == 1


# ---------------------------------------------------------------------------
# It must never break the pipeline
# ---------------------------------------------------------------------------
def test_unavailable_ollama_skips_without_touching_chunks():
    chunk = _chunk("Some long passage " * 30)
    result = _compressor(ScriptedClient(available=False)).run([chunk])

    assert result.metrics.status == StageStatus.SKIPPED
    assert "unavailable" in (result.metrics.note or "")
    assert chunk.compressed_text is None


def test_no_response_falls_back_to_the_original():
    chunk = _chunk("A passage the model will not answer for.")
    result = _compressor(ScriptedClient({})).run([chunk])

    assert result.metrics.status == StageStatus.OK
    assert result.accepted == 0
    assert result.rejected[0].reason == "no_response"
    assert chunk.compressed_text is None


def test_fast_mode_skips_the_stage_entirely():
    chunk = _chunk("Some long passage " * 30)
    client = ScriptedClient({})
    result = _compressor(client).run([chunk], fast_mode=True)

    assert result.metrics.status == StageStatus.SKIPPED
    assert "fast_mode" in (result.metrics.note or "")
    assert client.calls == [], "fast mode must not call the model at all"


def test_disabled_by_config():
    chunk = _chunk("Some long passage " * 30)
    client = ScriptedClient({})
    result = _compressor(client, enabled=False).run([chunk])

    assert result.metrics.status == StageStatus.SKIPPED
    assert client.calls == []


def test_stage_respects_its_wall_clock_ceiling():
    """A slow model must not be allowed to dominate a live /compress call."""
    chunks = [_chunk(f"Passage number {i} " * 30, i) for i in range(6)]
    client = ScriptedClient({}, delay=0.25)
    started = time.perf_counter()
    result = _compressor(client, total_timeout_s=0.6, timeout_s=0.3).run(chunks)
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0
    assert result.metrics.details["skipped_no_time"] > 0
    assert "budget exhausted" in (result.metrics.note or "")


def test_protected_chunks_are_never_rewritten():
    """The user's question must reach the model verbatim."""
    query = _chunk("What caused the outage? " * 20, kind=ChunkKind.QUERY)
    client = ScriptedClient({})
    _compressor(client).run([query])
    assert client.calls == []
    assert query.compressed_text is None


def test_short_chunks_are_not_attempted():
    """Below the threshold the call costs more than the paraphrase saves."""
    chunk = _chunk("Tiny.", tokens=12)
    client = ScriptedClient({})
    result = _compressor(client).run([chunk])

    assert client.calls == []
    assert result.metrics.status == StageStatus.SKIPPED
    assert "threshold" in (result.metrics.note or "")


def test_max_chunks_bounds_the_work():
    chunks = [_chunk(f"Passage {i} " * 40, i) for i in range(20)]
    client = ScriptedClient({})
    _compressor(client, max_chunks=3, total_timeout_s=30).run(chunks)
    assert len(client.calls) == 3


def test_largest_chunks_are_attempted_first():
    """If the ceiling cuts us off, spend the time where the payoff is."""
    small = _chunk("small " * 60, 0, tokens=160)
    large = _chunk("large " * 200, 1, tokens=500)
    client = ScriptedClient({})
    _compressor(client, max_chunks=1).run([small, large])

    assert len(client.calls) == 1
    assert "large" in client.calls[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_critical_tokens_capture_facts_a_prose_model_would_miss():
    text = "PAYMENT_POOL_SIZE fell to 8 in checkout_service.handlers at 09:41 for ORD-427039."
    tokens = critical_tokens(text)
    assert "PAYMENT_POOL_SIZE" in tokens
    assert "checkout_service.handlers" in tokens
    assert "8" in tokens and "09" in tokens and "41" in tokens


def test_numbers_in_finds_decimals():
    assert numbers_in("p99 was 8.4 s over 3 regions") == {"99", "8.4", "3"}


def test_markdown_fences_are_stripped():
    assert _strip_fences("```python\ndef f():\n    pass\n```") == "def f():\n    pass"
    assert _strip_fences("no fences here") == "no fences here"


def test_empty_input():
    result = _compressor(ScriptedClient({})).run([])
    assert result.metrics.status == StageStatus.SKIPPED


def test_rejection_rate_is_reported():
    chunks = [_chunk(f"Passage {i} with number {i}00 " * 20, i) for i in range(3)]
    client = ScriptedClient({})
    result = _compressor(client).run(chunks)
    assert result.rejection_rate == 1.0
    assert result.metrics.details["rejection_rate"] == 1.0
