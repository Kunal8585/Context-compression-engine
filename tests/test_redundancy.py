"""Stage 3 tests: what gets collapsed, what must not, and honest accounting."""

from __future__ import annotations

import numpy as np
import pytest

from engine.chunker import chunk_document, chunk_file
from engine.config import get_config
from engine.embeddings import EmbeddingModel
from engine.redundancy import RedundancyDetector, dedup_key, detect_redundancy
from engine.structural import structural_signature, tokenize
from engine.types import Chunk, ChunkKind, StageStatus


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder - loading MiniLM costs ~9s, so do it once per module."""
    return EmbeddingModel(get_config().redundancy)


@pytest.fixture(scope="module")
def log_result(corpus, embedder):
    chunks = chunk_file(corpus / "logs" / "checkout_service.log")
    return detect_redundancy(chunks, embedder=embedder)


@pytest.fixture(scope="module")
def code_result(corpus, embedder):
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    return detect_redundancy(chunks, embedder=embedder)


def _make_chunk(text: str, order: int, kind: str = ChunkKind.PARAGRAPH, **kw) -> Chunk:
    return Chunk(
        text=text,
        kind=kind,
        source="test",
        order=order,
        start_line=order + 1,
        end_line=order + 1,
        token_count=kw.pop("token_count", max(1, len(text) // 4)),
        id=kw.pop("id", f"test#{order:04d}:{kind}"),
        **kw,
    )


# ---------------------------------------------------------------------------
# Accounting: the numbers must reconcile, or the compression claim is unfounded
# ---------------------------------------------------------------------------
def test_every_removed_token_is_attributed_to_a_cluster(log_result):
    """Regression: transitive absorption used to orphan members from clusters."""
    metrics = log_result.metrics
    removed = metrics.tokens_in - metrics.tokens_out
    attributed = sum(c.absorbed_tokens for c in log_result.clusters.values())

    assert attributed == removed, (
        f"{removed} tokens removed but only {attributed} attributed to clusters"
    )
    assert metrics.details["accounting_ok"] is True


def test_chunk_counts_reconcile(log_result):
    metrics = log_result.metrics
    details = metrics.details
    collapsed = (
        details["exact_collapsed"]
        + details["structural_collapsed"]
        + details["embedding_collapsed"]
    )
    assert metrics.chunks_in - metrics.chunks_out == collapsed
    assert len(log_result.chunks) == metrics.chunks_out


def test_cluster_membership_is_complete_and_disjoint(log_result):
    """Each original chunk belongs to exactly one cluster, or to none."""
    seen: set[str] = set()
    for cluster in log_result.clusters.values():
        assert cluster.member_ids[0] == cluster.representative_id
        for member_id in cluster.member_ids:
            assert member_id not in seen, f"{member_id} is in two clusters"
            seen.add(member_id)

    kept_ids = {c.id for c in log_result.chunks}
    for cluster in log_result.clusters.values():
        assert cluster.representative_id in kept_ids
        # Absorbed members must NOT be in the surviving set.
        for member_id in cluster.member_ids[1:]:
            assert member_id not in kept_ids


def test_representative_keeps_its_full_original_text(log_result, corpus):
    source = (corpus / "logs" / "checkout_service.log").read_text(encoding="utf-8")
    for chunk in log_result.chunks:
        assert chunk.text in source, "a representative's text was mutated"
        assert chunk.compressed_text is None, "stage 3 must not rewrite text"


def test_duplicate_count_matches_cluster_size(log_result):
    for chunk in log_result.chunks:
        cluster = log_result.cluster_for(chunk)
        if cluster is None:
            assert chunk.duplicate_count == 1
        else:
            assert chunk.duplicate_count == cluster.size
            assert len(chunk.absorbed_ids) == cluster.absorbed_count


def test_survivors_stay_in_document_order(log_result):
    orders = [c.order for c in log_result.chunks]
    assert orders == sorted(orders)


# ---------------------------------------------------------------------------
# Logs: the redundancy the engine exists to exploit
# ---------------------------------------------------------------------------
def test_log_redundancy_is_substantial(log_result):
    assert log_result.metrics.reduction_pct > 90, (
        "the sample log is deliberately repetitive; stage 3 should collapse it"
    )


def test_rare_events_survive_collapse(log_result):
    """The whole point: high-frequency noise goes, the incident evidence stays."""
    surviving = "\n".join(c.text for c in log_result.chunks)

    for evidence in [
        "connection pool wait exceeded threshold",
        "PAYMENT_POOL_SIZE resolved to 8",
        "rollback initiated",
        "Traceback (most recent call last)",
        "gateway timeout after 8000ms",
    ]:
        assert evidence in surviving, f"stage 3 destroyed the evidence: {evidence!r}"


def test_traceback_is_never_collapsed_into_a_one_liner(log_result):
    traces = [c for c in log_result.chunks if "Traceback" in c.text]
    assert traces
    assert "requests.exceptions.ReadTimeout" in traces[0].text


# ---------------------------------------------------------------------------
# Structural dedup for code
# ---------------------------------------------------------------------------
def test_identical_validators_collapse(code_result):
    kept_symbols = {c.symbol for c in code_result.chunks}
    collapsed = {
        symbol
        for cluster in code_result.clusters.values()
        for symbol in cluster.absorbed_symbols
    }
    assert "validate_tenant_id" in collapsed
    assert "validate_device_id" in collapsed
    assert "validate_username" in kept_symbols


def test_validator_with_a_different_limit_is_not_collapsed(code_result):
    """128 must never merge with 64 - numbers are preserved by design."""
    kept_symbols = {c.symbol for c in code_result.chunks}
    assert "validate_password" in kept_symbols, (
        "validate_password uses a 128-char limit and must survive on its own"
    )


def test_structural_signature_preserves_numbers():
    a = "def f(x):\n    if len(x) > 64:\n        raise ValueError('too long')\n"
    b = "def g(y):\n    if len(y) > 64:\n        raise ValueError('nope')\n"
    c = "def h(z):\n    if len(z) > 128:\n        raise ValueError('nope')\n"

    assert structural_signature(a, "function") == structural_signature(b, "function")
    assert structural_signature(a, "function") != structural_signature(c, "function")


def test_structural_signature_separates_kinds():
    text = "def f(x):\n    return x\n"
    assert structural_signature(text, "function") != structural_signature(text, "method")


def test_structural_signature_ignores_docstring_wording():
    """Two validators differing only in their docstring text are one shape."""
    a = 'def f(x):\n    """Validate a username."""\n    return x + 1\n'
    b = 'def g(y):\n    """Validate a tenant id, which is a different thing."""\n    return y + 1\n'
    assert structural_signature(a, "function") == structural_signature(b, "function")


def test_structural_signature_ignores_comment_wording():
    a = "def f(x):\n    # one note\n    return x + 1\n"
    b = "def g(y):\n    # a totally different note\n    return y + 1\n"
    assert structural_signature(a, "function") == structural_signature(b, "function")


def test_docstring_presence_is_itself_structural():
    """Conservative by design: having a docstring is a structural difference."""
    with_doc = 'def f(x):\n    """Doc."""\n    return x + 1\n'
    without = "def f(x):\n    return x + 1\n"
    assert structural_signature(with_doc, "function") != structural_signature(
        without, "function"
    )


def test_structural_signature_respects_operators():
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return x * 1\n"
    assert structural_signature(a, "function") != structural_signature(b, "function")


def test_tokenizer_keeps_keywords_and_blanks_identifiers():
    tokens = tokenize("if not username: raise ValueError('nope')")
    assert "if" in tokens and "not" in tokens and "raise" in tokens
    assert "username" not in tokens and "ValueError" not in tokens
    assert tokens.count("<ID>") >= 2
    assert "<STR>" in tokens


def test_short_chunks_are_exempt_from_structural_dedup(corpus, embedder):
    """Four one-line exception classes share a shape; they must not collapse."""
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    result = detect_redundancy(chunks, embedder=embedder)
    kept = {c.symbol for c in result.chunks}

    for name in ["AuthError", "TokenExpiredError", "TokenRevokedError", "AccountLockedError"]:
        assert name in kept, f"{name} was collapsed despite being below min_tokens"


def test_structural_dedup_can_be_disabled(corpus, embedder):
    cfg = get_config().with_overrides(
        {"redundancy": {"structural": {"enabled": False}}}
    )
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    result = RedundancyDetector(cfg, embedder=embedder).run(chunks)
    assert result.metrics.details["structural_collapsed"] == 0


def test_prose_is_not_structurally_deduped(corpus, embedder):
    """Structural dedup must never fire on paragraphs - shape means nothing there."""
    chunks = chunk_file(corpus / "docs" / "incident_postmortem.md")
    result = detect_redundancy(chunks, embedder=embedder)
    assert result.metrics.details["structural_collapsed"] == 0


# ---------------------------------------------------------------------------
# Semantic dedup on prose (the case embeddings exist for)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ticket_result(corpus, embedder):
    chunks = chunk_file(corpus / "support" / "tickets.txt")
    return detect_redundancy(chunks, embedder=embedder)


def test_semantically_identical_tickets_merge_despite_different_wording(ticket_result):
    """No shared phrasing to hash - this is the job only embeddings can do."""
    assert ticket_result.metrics.details["embedding_collapsed"] > 0
    assert ticket_result.metrics.reduction_pct > 10


def test_distinct_customer_issues_survive_at_the_default_threshold(ticket_result):
    """A refund request must never be folded into a checkout-timeout cluster.

    Verified empirically by a threshold sweep: at 0.88 every merge is
    within-category; by 0.74 a duplicate-charge complaint merges into a
    damaged-item refund. The default sits well inside the safe zone.
    """
    surviving = "\n".join(c.text for c in ticket_result.chunks)
    for ticket, issue in [
        ("TICKET-8803", "damaged item refund"),
        ("TICKET-8805", "email address change"),
        ("TICKET-8811", "password reset"),
        ("TICKET-8814", "duplicate charge"),
        ("TICKET-8817", "loyalty points"),
        ("TICKET-8819", "auto-cancelled order"),
        ("TICKET-8825", "invoice download"),
        ("TICKET-8826", "resolution confirmation"),
    ]:
        assert ticket in surviving, f"{issue} ({ticket}) was wrongly collapsed"


def test_absorbed_tickets_remain_identifiable_in_the_audit_trail(ticket_result):
    """Collapsing is only acceptable because the marker names what went."""
    for cluster in ticket_result.clusters.values():
        assert cluster.absorbed_symbols
        for symbol in cluster.absorbed_symbols:
            assert "TICKET-" in symbol


# ---------------------------------------------------------------------------
# Precise keys beat fuzzy matching
# ---------------------------------------------------------------------------
def test_templated_chunks_are_exempt_from_fuzzy_clustering(log_result):
    """Regression: fuzzy matching used to undo stage 2's retention decision.

    Stage 2 deliberately keeps the service and region in a log template so that
    records from different services never merge. Embeddings score those pairs at
    0.998 and merged them anyway, collapsing four services and three regions
    into a single representative.
    """
    assert log_result.metrics.details["embedding_collapsed"] == 0
    assert "exempt from fuzzy clustering" in (log_result.metrics.note or "")

    surviving_templates = {c.metadata["template"] for c in log_result.chunks}
    services = {
        service
        for template in surviving_templates
        for service in ["checkout-api", "payment-client", "cart-store", "inventory"]
        if service in template
    }
    regions = {
        region
        for template in surviving_templates
        for region in ["eu-west-1", "us-east-1", "ap-south-1"]
        if region in template
    }
    assert len(services) == 4, "a service was lost to fuzzy clustering"
    assert len(regions) == 3, "a region was lost to fuzzy clustering"


# ---------------------------------------------------------------------------
# Exact pass and dedup keys
# ---------------------------------------------------------------------------
def test_log_records_key_on_their_template():
    a = _make_chunk("2024-01-01 00:00:01 INFO served order=ORD-111111", 0,
                    ChunkKind.LOG_RECORD)
    a.metadata["template"] = "<TS> INFO served order=<ID>"
    b = _make_chunk("2024-01-01 00:00:09 INFO served order=ORD-222222", 1,
                    ChunkKind.LOG_RECORD)
    b.metadata["template"] = "<TS> INFO served order=<ID>"
    assert dedup_key(a) == dedup_key(b)


def test_text_keys_ignore_whitespace_but_not_content():
    a = _make_chunk("the  service   failed", 0)
    b = _make_chunk("the service failed", 1)
    c = _make_chunk("the service recovered", 2)
    assert dedup_key(a) == dedup_key(b)
    assert dedup_key(a) != dedup_key(c)


def test_exact_pass_keeps_the_first_occurrence(embedder):
    chunks = [
        _make_chunk("identical content here", 0, id="first"),
        _make_chunk("identical content here", 1, id="second"),
        _make_chunk("identical content here", 2, id="third"),
    ]
    result = detect_redundancy(chunks, embedder=embedder)
    assert [c.id for c in result.chunks] == ["first"]
    assert result.chunks[0].duplicate_count == 3


# ---------------------------------------------------------------------------
# Protected chunks
# ---------------------------------------------------------------------------
def test_protected_chunks_are_never_collapsed(embedder):
    """The user's question must survive even if it duplicates the context."""
    text = "What caused the checkout latency spike?"
    chunks = [
        _make_chunk(text, 0, ChunkKind.PARAGRAPH, id="context-copy"),
        _make_chunk(text, 1, ChunkKind.QUERY, id="the-query"),
        _make_chunk(text, 2, ChunkKind.INSTRUCTION, id="the-instruction"),
    ]
    result = detect_redundancy(chunks, embedder=embedder)
    kept = {c.id for c in result.chunks}

    assert "the-query" in kept
    assert "the-instruction" in kept
    assert result.metrics.details["protected_chunks"] == 2


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------
class _DeadEmbedder(EmbeddingModel):
    """Simulates sentence-transformers being unavailable."""

    def __init__(self):
        super().__init__(get_config().redundancy)
        self._loaded = True
        self._model = None
        self._load_error = "simulated: no model available"

    def encode(self, texts):
        return None


def test_pipeline_survives_a_dead_embedding_backend(corpus):
    """Exact + structural passes must still run and report the degradation."""
    chunks = chunk_file(corpus / "logs" / "checkout_service.log")
    result = detect_redundancy(chunks, embedder=_DeadEmbedder())

    assert result.metrics.status == StageStatus.OK
    assert result.metrics.details["embedding_collapsed"] == 0
    assert result.metrics.details["exact_collapsed"] > 2000
    assert result.embeddings is None
    assert "unavailable" in (result.metrics.note or "")
    # Still a large win from the exact pass alone.
    assert result.metrics.reduction_pct > 85


def test_embeddings_can_be_disabled_by_config(corpus, embedder):
    cfg = get_config().with_overrides({"redundancy": {"enabled": False}})
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    result = RedundancyDetector(cfg, embedder=embedder).run(chunks)

    assert result.metrics.details["embedding_collapsed"] == 0
    assert result.embeddings is None
    # Structural dedup is independent of the embedding model and still runs.
    assert result.metrics.details["structural_collapsed"] == 2


def test_empty_input(embedder):
    result = detect_redundancy([], embedder=embedder)
    assert result.chunks == []
    assert result.metrics.status == StageStatus.SKIPPED


def test_single_chunk_input(embedder):
    result = detect_redundancy([_make_chunk("only one", 0)], embedder=embedder)
    assert len(result.chunks) == 1
    assert result.clusters == {}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_embeddings_align_with_surviving_chunks(log_result):
    assert log_result.embeddings is not None
    assert log_result.embeddings.shape[0] == len(log_result.chunks)
    assert log_result.embeddings.shape[1] == 384


def test_embeddings_are_l2_normalised(log_result):
    norms = np.linalg.norm(log_result.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_threshold_is_respected(corpus, embedder):
    """A stricter threshold must never collapse more than a looser one."""
    chunks = chunk_file(corpus / "logs" / "checkout_service.log")
    loose = RedundancyDetector(
        get_config().with_overrides({"redundancy": {"similarity_threshold": 0.85}}),
        embedder=embedder,
    ).run(chunks)
    strict = RedundancyDetector(
        get_config().with_overrides({"redundancy": {"similarity_threshold": 0.99}}),
        embedder=embedder,
    ).run(chunks)

    assert strict.metrics.chunks_out >= loose.metrics.chunks_out


def test_run_is_idempotent(corpus, embedder):
    """Re-running on the same chunk objects must not double-count."""
    chunks = chunk_file(corpus / "code" / "auth_service.py")
    detector = RedundancyDetector(embedder=embedder)
    first = detector.run(chunks)
    second = detector.run(chunks)

    assert first.metrics.chunks_out == second.metrics.chunks_out
    assert first.metrics.tokens_out == second.metrics.tokens_out
