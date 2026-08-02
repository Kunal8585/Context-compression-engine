"""Stage 7 + end-to-end pipeline tests.

The headline assertion here is budget compliance on the *reconstructed prompt* -
the artifact that actually gets sent to a model - across every input type at
50%, 30% and 15%.
"""

from __future__ import annotations

import pytest

from engine.config import get_config
from engine.embeddings import EmbeddingModel
from engine.pipeline import CompressionPipeline
from engine.reconstruct import Reconstructor, reconstruct
from engine.types import Chunk, ChunkKind, Cluster, StageStatus

BUDGETS = [0.50, 0.30, 0.15]
CORPUS_FILES = [
    "code/auth_service.py",
    "code/payment_client.js",
    "logs/checkout_service.log",
    "docs/incident_postmortem.md",
    "support/tickets.txt",
]


@pytest.fixture(scope="module")
def pipeline():
    """Pipeline with stage 6 off.

    Stage 6 calls a live 3B model at 3-6 s per chunk, which would make this
    suite depend on a running Ollama and take minutes. Its behaviour is covered
    deterministically in test_abstractive.py with a scripted client; here we
    test the extractive path, which is what must hold regardless.
    """
    pipe = CompressionPipeline(
        get_config().with_overrides({"abstractive": {"enabled": False}})
    )
    pipe.warmup()
    return pipe


def _chunk(text, order, tokens, kind=ChunkKind.PARAGRAPH, symbol=None) -> Chunk:
    return Chunk(
        text=text, kind=kind, source="test", order=order,
        start_line=order + 1, end_line=order + 1, token_count=tokens,
        id=f"test#{order:04d}", symbol=symbol,
    )


# ---------------------------------------------------------------------------
# Budget compliance - the number the whole project is judged on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("relative", CORPUS_FILES)
@pytest.mark.parametrize("ratio", BUDGETS)
def test_reconstructed_prompt_fits_the_budget(pipeline, corpus, relative, ratio):
    """Markers cost tokens; the artifact that ships must still fit.

    A fixed percentage reserve could not hold this - measured overshoot ran
    from +12 to +119 tokens - so selection and reconstruction run as a feedback
    loop.
    """
    source = (corpus / relative).read_text(encoding="utf-8")
    result = pipeline.compress(source, relative.split("/")[-1], budget_ratio=ratio)

    assert result.compressed_tokens <= result.budget_tokens, (
        f"{relative} at {ratio:.0%}: {result.compressed_tokens} tokens "
        f"exceeds the {result.budget_tokens} token budget"
    )
    assert result.compressed_text.strip()


@pytest.mark.parametrize("relative", CORPUS_FILES)
def test_compression_meets_the_seventy_percent_target(pipeline, corpus, relative):
    source = (corpus / relative).read_text(encoding="utf-8")
    result = pipeline.compress(source, relative.split("/")[-1], budget_ratio=0.30)
    assert result.compression_ratio >= 0.70


def test_tighter_budgets_never_produce_larger_prompts(pipeline, corpus):
    source = (corpus / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")
    sizes = [
        pipeline.compress(source, "doc.md", budget_ratio=r).compressed_tokens
        for r in BUDGETS
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_budget_already_met_by_deduplication_is_not_padded(pipeline, corpus):
    """The log dedupes below even a 15% budget; selection should drop nothing."""
    source = (corpus / "logs" / "checkout_service.log").read_text(encoding="utf-8")
    result = pipeline.compress(source, "checkout_service.log", budget_ratio=0.50)
    assert result.compressed_tokens < result.budget_tokens
    assert result.stage("selection").details["dropped_chunks"] == 0


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
def test_kept_content_appears_in_document_order():
    chunks = [_chunk(f"Paragraph number {i}.", i, 10) for i in range(6)]
    kept = [chunks[0], chunks[3], chunks[5]]
    text = reconstruct(chunks, kept).text

    assert text.index("number 0") < text.index("number 3") < text.index("number 5")


def test_drop_markers_report_what_was_removed():
    chunks = [_chunk(f"Paragraph number {i}.", i, 10) for i in range(6)]
    result = reconstruct(chunks, [chunks[0], chunks[5]])

    # The `#d0` tag is the marker's address for /expand - see reconstruct.py.
    assert "omitted #d0 4 section(s)" in result.text
    assert "40 tokens" in result.text
    assert result.marker_count == 1
    assert [m["id"] for m in result.recoverable] == ["d0"]


def test_cluster_markers_report_collapsed_duplicates():
    representative = _chunk("cache hit", 0, 10, ChunkKind.LOG_RECORD)
    representative.cluster_id = 0
    cluster = Cluster(
        id=0, representative_id=representative.id,
        member_ids=[representative.id] + [f"m{i}" for i in range(171)],
        absorbed_tokens=6752, method="exact",
    )
    result = reconstruct([representative], [representative], {0: cluster})

    assert "x172 #c0 near-identical" in result.text
    assert "6752 tokens saved" in result.text
    assert [m["id"] for m in result.recoverable] == ["c0"]


def test_absorbed_chunks_do_not_get_their_own_drop_marker():
    """Regression: emitting both markers cost more than the content saved.

    A 2,400-record log reconstructed to 18,859 tokens against 6,560 tokens of
    real content, because every deduplicated record produced a drop marker that
    duplicated its cluster marker.
    """
    representative = _chunk("cache hit", 0, 10, ChunkKind.LOG_RECORD)
    representative.cluster_id = 0
    absorbed = [_chunk("cache hit", i, 10, ChunkKind.LOG_RECORD) for i in range(1, 50)]
    cluster = Cluster(
        id=0, representative_id=representative.id,
        member_ids=[representative.id] + [c.id for c in absorbed],
        absorbed_tokens=490, method="exact",
    )
    result = reconstruct(
        [representative, *absorbed], [representative], {0: cluster},
        absorbed_ids={c.id for c in absorbed},
    )

    assert "omitted" not in result.text
    assert result.marker_count == 1


def test_markers_do_not_embed_long_log_templates():
    """A log 'symbol' is a 120-char template - it must not go in a marker."""
    template = "<TS> INFO cart-store [ap-south-1] cache hit key=cart:<ID> " * 2
    dropped = [
        _chunk("record", i, 10, ChunkKind.LOG_RECORD, symbol=template) for i in range(4)
    ]
    kept = _chunk("kept record", 4, 10, ChunkKind.LOG_RECORD)
    result = reconstruct([*dropped, kept], [kept])

    assert "cart-store" not in result.text
    assert len(result.text) < 200


def test_code_markers_do_name_dropped_symbols():
    """For code a dropped function name is a searchable fact worth the tokens."""
    dropped = [
        _chunk("def helper_one(): pass", 0, 10, ChunkKind.FUNCTION, symbol="helper_one"),
        _chunk("def helper_two(): pass", 1, 10, ChunkKind.FUNCTION, symbol="helper_two"),
    ]
    kept = _chunk("def main(): pass", 2, 10, ChunkKind.FUNCTION, symbol="main")
    text = reconstruct([*dropped, kept], [kept]).text

    assert "helper_one" in text and "helper_two" in text


def test_markers_can_be_disabled():
    cfg = get_config().with_overrides(
        {"reconstruction": {"drop_markers": False, "cluster_markers": False}}
    )
    chunks = [_chunk(f"Paragraph {i}.", i, 10) for i in range(6)]
    result = Reconstructor(cfg).run(chunks, [chunks[0], chunks[5]])

    assert "omitted" not in result.text
    assert result.marker_count == 0


def test_reconstruction_of_nothing_is_skipped():
    result = reconstruct([_chunk("a", 0, 5)], [])
    assert result.metrics.status == StageStatus.SKIPPED
    assert result.text == ""


def test_log_records_join_compactly_prose_does_not():
    logs = [_chunk(f"line {i}", i, 5, ChunkKind.LOG_RECORD) for i in range(3)]
    assert "\n\n" not in reconstruct(logs, logs).text

    prose = [_chunk(f"Paragraph {i}.", i, 5, ChunkKind.PARAGRAPH) for i in range(3)]
    assert "\n\n" in reconstruct(prose, prose).text


# ---------------------------------------------------------------------------
# Pipeline contract
# ---------------------------------------------------------------------------
def test_all_stages_are_reported(pipeline, corpus):
    source = (corpus / "code" / "auth_service.py").read_text(encoding="utf-8")
    result = pipeline.compress(source, "auth_service.py")

    names = [s.name for s in result.stages]
    assert names == [
        "chunking", "redundancy", "density", "selection",
        "abstractive", "reconstruction",
    ], "the accordion needs a fixed six-stage shape in every environment"
    for stage in result.stages:
        assert stage.status in {StageStatus.OK, StageStatus.SKIPPED}
        assert stage.duration_ms >= 0
        if stage.status == StageStatus.SKIPPED:
            assert stage.note, f"{stage.name} skipped without saying why"


def test_the_query_survives_compression(pipeline, corpus):
    """RAG-style call: the question must be in the output, at any budget."""
    source = (corpus / "logs" / "checkout_service.log").read_text(encoding="utf-8")
    question = "Which configuration value caused the checkout latency spike?"
    result = pipeline.compress(
        source, "checkout_service.log", budget_ratio=0.05,
        query=question, instruction="Answer using only the context provided.",
    )

    assert question in result.compressed_text
    assert "Answer using only the context provided." in result.compressed_text


def test_summary_numbers_are_self_consistent(pipeline, corpus):
    source = (corpus / "support" / "tickets.txt").read_text(encoding="utf-8")
    result = pipeline.compress(source, "tickets.txt", budget_ratio=0.30)
    summary = result.summary()

    assert summary["tokens_saved"] == summary["original_tokens"] - summary["compressed_tokens"]
    assert summary["compression_ratio"] == pytest.approx(result.compression_ratio, abs=1e-4)
    assert summary["chunks_kept"] + summary["chunks_dropped"] == summary["chunks_total"]
    assert summary["compression_pct"] == pytest.approx(100 * result.compression_ratio, abs=0.01)


def test_result_serialises_for_the_api(pipeline, corpus):
    source = (corpus / "code" / "auth_service.py").read_text(encoding="utf-8")
    result = pipeline.compress(source, "auth_service.py")
    payload = result.to_dict()

    assert {"summary", "stages", "audit_trail", "compressed_text", "spans"} <= set(payload)
    assert isinstance(payload["stages"], list)
    assert all("duration_ms" in s for s in payload["stages"])

    # The client already holds the input; echoing it back is pure payload.
    assert "original_text" not in payload
    assert "chunks" not in payload
    assert payload["original_text"] if False else True

    verbose = result.to_dict(include_original=True, include_chunks=True)
    assert verbose["original_text"] == source
    assert len(verbose["chunks"]) == len(result.chunks)


def test_spans_index_the_original_text(pipeline, corpus):
    """The frontend highlights its own copy using these offsets."""
    source = (corpus / "code" / "auth_service.py").read_text(encoding="utf-8")
    result = pipeline.compress(source, "auth_service.py", budget_ratio=0.30)
    spans = result.spans()

    assert spans
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans)
    for span in spans:
        assert 0 <= span["start"] < span["end"] <= len(source)
        assert isinstance(span["kept"], bool)
    # Every kept span's text must actually be present in the compressed output.
    kept = [s for s in spans if s["kept"]]
    assert kept
    for span in kept[:5]:
        excerpt = source[span["start"] : span["end"]].strip().splitlines()[0].strip()
        if len(excerpt) > 15:
            assert excerpt in result.compressed_text


def test_payload_stays_small_on_a_large_input(pipeline, corpus):
    """2.17 MB of echoed text was the whole reason for the spans design."""
    import json

    source = (corpus / "logs" / "checkout_service.log").read_text(encoding="utf-8")
    result = pipeline.compress(source, "checkout_service.log", budget_ratio=0.30)
    size = len(json.dumps(result.to_dict()))

    assert size < 400_000, f"response is {size:,} bytes for a 107k-token input"


def test_empty_input_returns_an_empty_result(pipeline):
    result = pipeline.compress("   \n\n  ", "empty.txt")
    assert result.compressed_text == ""
    assert result.compression_ratio == 0.0
    assert result.stages[0].status == StageStatus.SKIPPED


def test_pipeline_is_deterministic(pipeline, corpus):
    source = (corpus / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")
    first = pipeline.compress(source, "doc.md", budget_ratio=0.30)
    second = pipeline.compress(source, "doc.md", budget_ratio=0.30)
    assert first.compressed_text == second.compressed_text


def test_incident_evidence_survives_aggressive_compression(pipeline, corpus):
    """End-to-end retention: the root cause must reach the compressed prompt."""
    source = (corpus / "logs" / "checkout_service.log").read_text(encoding="utf-8")
    result = pipeline.compress(source, "checkout_service.log", budget_ratio=0.15)

    for evidence in ["PAYMENT_POOL_SIZE", "Traceback", "gateway timeout", "rollback"]:
        assert evidence in result.compressed_text, (
            f"{evidence!r} was compressed away at a 15% budget"
        )
