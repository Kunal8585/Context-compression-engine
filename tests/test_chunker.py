"""Stage 2 tests: chunk boundaries must be semantically sensible and lossless."""

from __future__ import annotations

import pytest

from conftest import assert_ordered, assert_tiles
from engine.chunker import (
    CodeChunker,
    LogChunker,
    TextChunker,
    chunk_document,
    chunk_file,
    detect_kind,
    select_chunker,
)
from engine.chunker.code import PYTHON, language_for
from engine.chunker.logs import log_template, looks_like_log
from engine.config import ChunkingConfig, get_config
from engine.types import ChunkKind


# ---------------------------------------------------------------------------
# The core invariant, checked on every backend and every sample file
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "relative",
    [
        "code/auth_service.py",
        "code/payment_client.js",
        "docs/incident_postmortem.md",
        "logs/checkout_service.log",
    ],
)
def test_chunks_tile_the_source_losslessly(corpus, relative):
    source = (corpus / relative).read_text(encoding="utf-8")
    chunks = chunk_file(corpus / relative)

    assert chunks, "chunker produced no chunks"
    assert_tiles(source, chunks)
    assert_ordered(chunks)


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("code/auth_service.py", "code"),
        ("code/payment_client.js", "code"),
        ("docs/incident_postmortem.md", "text"),
        ("logs/checkout_service.log", "log"),
    ],
)
def test_backend_auto_detection(corpus, relative, expected):
    source = (corpus / relative).read_text(encoding="utf-8")
    assert detect_kind((corpus / relative).name, source) == expected


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------
def test_python_functions_become_individual_chunks(python_source):
    chunks = chunk_document(python_source, "auth_service.py")
    symbols = {c.symbol for c in chunks}

    for expected in [
        "validate_username",
        "validate_password",
        "hash_password",
        "verify_password",
        "authenticate",
        "record_failed_attempt",
    ]:
        assert expected in symbols, f"{expected} was not chunked as its own unit"

    # Each of those is a whole function, not a fragment.
    validate = next(c for c in chunks if c.symbol == "validate_username")
    assert validate.kind == ChunkKind.FUNCTION
    assert validate.text.lstrip().startswith("def validate_username")
    assert "username must be ASCII" in validate.text


def test_python_exception_classes_are_whole_chunks(python_source):
    chunks = chunk_document(python_source, "auth_service.py")
    token_error = next(c for c in chunks if c.symbol == "TokenExpiredError")
    assert token_error.kind == ChunkKind.CLASS
    assert token_error.text.lstrip().startswith("class TokenExpiredError")
    assert "exp claim" in token_error.text


def test_decorator_travels_with_its_definition(python_source):
    chunks = chunk_document(python_source, "auth_service.py")
    token_pair = next(c for c in chunks if c.symbol == "TokenPair")
    assert token_pair.text.lstrip().startswith("@dataclass")


def test_oversized_class_is_split_into_methods(python_source):
    chunks = chunk_document(python_source, "auth_service.py")
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    symbols = {c.symbol for c in methods}

    assert {"TokenService.issue", "TokenService.verify", "TokenService.refresh"} <= symbols
    header = next(c for c in chunks if c.kind == ChunkKind.CLASS_HEADER)
    assert header.symbol == "TokenService"
    # The class docstring stays with the header, not with a random method.
    assert "stateless apart from" in header.text


def test_module_level_statements_are_grouped(python_source):
    chunks = chunk_document(python_source, "auth_service.py")
    module_chunks = [c for c in chunks if c.kind == ChunkKind.MODULE_LEVEL]

    assert module_chunks, "imports and constants were not captured"
    joined = "\n".join(c.text for c in module_chunks)
    assert "import jwt" in joined
    assert "ACCESS_TOKEN_TTL_SECONDS = 900" in joined


def test_javascript_class_split_keeps_interior_comments(js_source):
    """Regression: cherry-picking method nodes used to drop body comments."""
    chunks = chunk_document(js_source, "payment_client.js")
    charge = next(c for c in chunks if c.symbol == "PaymentClient.charge")
    assert "Charge a card with exponential backoff" in charge.text
    assert "async charge(" in charge.text


def test_javascript_top_level_functions(js_source):
    chunks = chunk_document(js_source, "payment_client.js")
    symbols = {c.symbol for c in chunks}
    assert {"sleep", "isRetryable", "buildIdempotencyKey", "PaymentError"} <= symbols


def test_leading_comment_attaches_to_following_function(js_source):
    chunks = chunk_document(js_source, "payment_client.js")
    sleep_chunk = next(c for c in chunks if c.symbol == "sleep")
    assert "Sleep helper" in sleep_chunk.text


def test_unparsable_code_falls_back_instead_of_crashing():
    """Graceful degradation: a broken file must still produce usable chunks."""
    broken = "def f(:\n  <<<<<<< HEAD\n  ???\n\n" + ("garbage ~~~ !!!\n" * 40)
    chunks = chunk_document(broken, "broken.py")
    assert chunks
    assert_tiles(broken, chunks)


def test_unsupported_language_degrades_to_text_chunking():
    source = "package main\n\nfunc main() {\n\tprintln(\"hi\")\n}\n\nfunc other() {}\n"
    chunker = select_chunker("main.go", source)
    assert isinstance(chunker, TextChunker)
    chunks = chunker.chunk(source, "main.go")
    assert chunks
    assert_tiles(source, chunks)


def test_language_registry():
    assert language_for("x.py") is PYTHON
    assert language_for("x.unknown") is None
    assert CodeChunker.available_for("x.py") is True


# ---------------------------------------------------------------------------
# Text / markdown chunking
# ---------------------------------------------------------------------------
def test_fenced_code_blocks_stay_atomic(markdown_source):
    chunks = chunk_document(markdown_source, "incident_postmortem.md")
    blocks = [c for c in chunks if c.kind == ChunkKind.CODE_BLOCK]

    assert len(blocks) == 2, "expected the yaml diff and the sql query"
    sql = next(c for c in blocks if c.symbol == "sql")
    assert sql.text.count("```") == 2, "code fence was split in half"
    assert "percentile_cont" in sql.text
    assert "HAVING" in sql.text


def test_headings_merge_forwards_into_the_section_they_label(markdown_source):
    """Regression: runt headings used to fold into the *previous* section."""
    chunks = chunk_document(markdown_source, "incident_postmortem.md")

    # `## Root Cause` is a 3-token runt; it should not survive on its own, and
    # it must land on the section it introduces.
    root_cause = next(c for c in chunks if "## Root Cause" in c.text)
    assert root_cause.token_count > 10
    assert "unrelated refactor" in root_cause.text
    assert root_cause.symbol == "Root Cause"
    assert "**11:48**" not in root_cause.text, "heading merged backwards"

    # And the timeline list keeps its own heading, not the next one.
    timeline = next(c for c in chunks if "**09:05**" in c.text)
    assert timeline.symbol == "Timeline"


def test_paragraph_boundaries_are_blank_line_separated(markdown_source):
    chunks = chunk_document(markdown_source, "incident_postmortem.md")
    paragraphs = [c for c in chunks if c.kind == ChunkKind.PARAGRAPH]
    assert paragraphs
    for chunk in paragraphs:
        # A merged chunk legitimately spans the blank line it was joined across;
        # every other paragraph must be a single blank-line-delimited block.
        if chunk.metadata.get("merged_parts"):
            continue
        assert "\n\n" not in chunk.text.strip(), (
            f"chunk {chunk.id} spans a paragraph break"
        )


def test_oversized_paragraph_is_split_on_sentences():
    cfg = ChunkingConfig(target_chunk_tokens=30, max_chunk_tokens=40, min_chunk_tokens=5)
    sentences = [
        f"The service processed {i} requests in region eu-west-{i % 3} without error."
        for i in range(40)
    ]
    source = " ".join(sentences)
    chunks = TextChunker(cfg).chunk(source, "long.txt")

    assert len(chunks) > 1
    assert_tiles(source, chunks)
    # No chunk grossly exceeds the ceiling, and none ends mid-sentence.
    for chunk in chunks:
        assert chunk.token_count <= cfg.max_chunk_tokens + 20
        assert chunk.text.strip().endswith(".")


def test_sentence_splitter_respects_abbreviations():
    source = (
        "Dr. Smith reviewed the incident at 09:12 UTC. "
        "The p99 latency was 8.4 s, i.e. well above target. "
        "No data was lost."
    )
    cfg = ChunkingConfig(target_chunk_tokens=8, max_chunk_tokens=10, min_chunk_tokens=1)
    chunks = TextChunker(cfg).chunk(source, "abbrev.txt")

    assert_tiles(source, chunks)
    for chunk in chunks:
        assert not chunk.text.strip().endswith("Dr.")
        assert not chunk.text.strip().endswith("i.e.")


# ---------------------------------------------------------------------------
# Log chunking
# ---------------------------------------------------------------------------
def test_every_log_record_is_captured(log_source):
    chunks = chunk_document(log_source, "checkout_service.log")
    # The generator emits exactly 2400 records.
    assert len(chunks) == 2400
    assert all(c.kind == ChunkKind.LOG_RECORD for c in chunks)


def test_multiline_traceback_is_one_record(log_source):
    """Regression: the exception line used to orphan into its own record."""
    chunks = chunk_document(log_source, "checkout_service.log")
    traces = [c for c in chunks if "Traceback (most recent call last)" in c.text]

    assert traces, "no traceback found in the sample log"
    for trace in traces:
        assert trace.metadata["level"] == "ERROR"
        assert trace.text.rstrip().endswith("(read timeout=8.0)"), (
            "the exception line was split off from its traceback"
        )
        assert "post_checkout" in trace.text


def test_log_levels_are_extracted(log_source):
    chunks = chunk_document(log_source, "checkout_service.log")
    levels = {c.metadata["level"] for c in chunks}
    assert levels == {"INFO", "DEBUG", "WARN", "ERROR"}
    assert all(c.metadata["level"] is not None for c in chunks)


def test_templates_collapse_volatile_fields_but_keep_signal():
    record = (
        "2024-03-14 09:07:52.869 ERROR checkout-api [eu-west-1] "
        "gateway timeout after 8000ms order=ORD-427039 attempt=3 "
        "idempotency_key=ORD-427039:3\n"
    )
    template = log_template(record)

    # Volatile fields are normalised...
    assert "<TS>" in template
    assert "ORD-427039" not in template
    assert "<ID>" in template
    # ...but the parts a judge can ask about survive.
    assert "eu-west-1" in template, "region must not be collapsed into a placeholder"
    assert "checkout-api" in template, "service name must survive"
    assert "gateway timeout after" in template
    assert "ERROR" in template


def test_template_collapses_repeated_records(log_source):
    chunks = chunk_document(log_source, "checkout_service.log")
    templates = {c.metadata["template"] for c in chunks}
    # Heavy redundancy is the whole premise of the engine; assert it is real.
    assert len(templates) < len(chunks) * 0.2


def test_looks_like_log_heuristic(log_source, markdown_source, python_source):
    assert looks_like_log(log_source) is True
    assert looks_like_log(markdown_source) is False
    assert looks_like_log(python_source) is False


def test_log_without_recognisable_format_still_chunks():
    source = "alpha beta\ngamma delta\nepsilon zeta\n"
    chunks = LogChunker().chunk(source, "odd.log")
    assert chunks
    assert_tiles(source, chunks)


# ---------------------------------------------------------------------------
# Size discipline and edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "relative",
    [
        "code/auth_service.py",
        "code/payment_client.js",
        "docs/incident_postmortem.md",
        "logs/checkout_service.log",
    ],
)
def test_chunks_respect_the_size_ceiling(corpus, relative):
    cfg = get_config().chunking
    chunks = chunk_file(corpus / relative)
    for chunk in chunks:
        if chunk.token_count > cfg.max_chunk_tokens:
            # Only an indivisible unit (a single very long line) may exceed it.
            assert chunk.text.strip().count("\n") == 0, (
                f"{chunk.id} is {chunk.token_count} tokens and was not split"
            )


@pytest.mark.parametrize("source", ["", "   ", "\n\n\t\n"])
def test_empty_input_produces_no_chunks(source):
    assert chunk_document(source, "empty.txt") == []


def test_single_line_input():
    chunks = chunk_document("just one line of prose", "tiny.txt")
    assert len(chunks) == 1
    assert chunks[0].token_count > 0


def test_unicode_offsets_are_correct():
    """Byte offsets from tree-sitter must be mapped to character offsets."""
    source = (
        '"""Résumé parser — handés naïve input."""\n\n'
        "def parse_café(x):\n"
        "    # ☕ mind the emoji\n"
        "    return x\n\n"
        "def other_ünïcode(y):\n"
        "    return y\n"
    )
    chunks = chunk_document(source, "unicode.py")
    assert_tiles(source, chunks)
    symbols = {c.symbol for c in chunks}
    assert "parse_café" in symbols
    assert "other_ünïcode" in symbols


def test_forcing_a_backend_overrides_detection(log_source):
    forced = chunk_document(log_source[:5000], "checkout_service.log", kind="text")
    assert all(c.kind != ChunkKind.LOG_RECORD for c in forced)


def test_chunk_serialisation_round_trips(python_source):
    chunks = chunk_document(python_source, "auth_service.py")
    payload = chunks[0].to_dict()
    assert payload["text"] == chunks[0].text
    assert payload["token_count"] == chunks[0].token_count
    compact = chunks[0].to_dict(include_text=False)
    assert "text" not in compact and "preview" in compact
