from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Collapse both provider chains to their local entries for the whole suite.
#
# Set before any engine import so no module-level chain is built against a
# hosted provider. A unit test must never depend on a network, someone's API
# key, or a free tier's remaining quota - a suite that turns red because Groq
# is rate-limited is worse than no suite. Tests that exercise the cloud
# providers do so against scripted transports (tests/test_providers.py); tests
# that hit real APIs are opt-in and marked `integration`.
os.environ.setdefault("CCE_OFFLINE", "1")

CORPUS = PROJECT_ROOT / "data" / "sample_corpus"


@pytest.fixture(scope="session")
def corpus() -> Path:
    return CORPUS


@pytest.fixture()
def config():
    """The project's real config.yaml, for tests that exercise provider wiring."""
    from engine.config import get_config

    return get_config()


@pytest.fixture(scope="session")
def python_source() -> str:
    return (CORPUS / "code" / "auth_service.py").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def js_source() -> str:
    return (CORPUS / "code" / "payment_client.js").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def markdown_source() -> str:
    return (CORPUS / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def log_source() -> str:
    return (CORPUS / "logs" / "checkout_service.log").read_text(encoding="utf-8")


def assert_tiles(source: str, chunks) -> None:
    """Chunks must cover the source in order, with no overlap and no loss.

    This is the single most important invariant in stage 2: if the chunker
    silently drops text, every downstream compression number is a lie.
    """
    cursor = 0
    for chunk in chunks:
        assert chunk.start_char >= cursor, (
            f"chunk {chunk.id} overlaps the previous chunk "
            f"({chunk.start_char} < {cursor})"
        )
        gap = source[cursor : chunk.start_char]
        assert not gap.strip(), f"content dropped before {chunk.id}: {gap.strip()[:80]!r}"
        assert source[chunk.start_char : chunk.end_char] == chunk.text, (
            f"chunk {chunk.id} text does not match its recorded char span"
        )
        assert chunk.text.strip(), f"chunk {chunk.id} is whitespace only"
        cursor = chunk.end_char
    tail = source[cursor:]
    assert not tail.strip(), f"content dropped at end of document: {tail.strip()[:80]!r}"


def assert_ordered(chunks) -> None:
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids must be unique"
    for chunk in chunks:
        assert chunk.start_line <= chunk.end_line
        assert chunk.token_count > 0
