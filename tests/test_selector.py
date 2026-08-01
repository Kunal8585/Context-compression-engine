"""Stage 5 tests: budget discipline, protection, and dependency preservation."""

from __future__ import annotations

import pytest

from engine.config import get_config
from engine.dependencies import (
    build_symbol_index,
    code_identifiers,
    defined_names,
    find_broken,
    referenced_names,
)
from engine.selector import (
    DROPPED_BUDGET,
    KEPT_DENSITY,
    KEPT_DEPENDENCY,
    KEPT_PROTECTED,
    BudgetSelector,
    select_within_budget,
)
from engine.types import Chunk, ChunkKind, StageStatus


def _chunk(
    text: str,
    order: int,
    tokens: int,
    density: float,
    kind: str = ChunkKind.PARAGRAPH,
    symbol: str | None = None,
) -> Chunk:
    chunk = Chunk(
        text=text,
        kind=kind,
        source="test",
        order=order,
        start_line=order + 1,
        end_line=order + 1,
        token_count=tokens,
        id=f"test#{order:04d}:{symbol or kind}",
        symbol=symbol,
    )
    chunk.density = density
    return chunk


# ---------------------------------------------------------------------------
# Budget discipline
# ---------------------------------------------------------------------------
def test_selection_respects_the_budget():
    chunks = [_chunk(f"chunk {i}", i, 100, 1.0 - i / 20) for i in range(20)]
    result = select_within_budget(chunks, budget_tokens=500)

    assert result.used_tokens <= 500
    assert sum(c.token_count for c in result.kept) == result.used_tokens


def test_highest_density_chunks_are_kept_first():
    chunks = [_chunk(f"chunk {i}", i, 100, i / 10) for i in range(10)]
    result = select_within_budget(chunks, budget_tokens=300)

    kept_densities = sorted((c.density for c in result.kept), reverse=True)
    dropped_densities = [c.density for c in result.dropped]
    assert min(kept_densities) >= max(dropped_densities)


def test_budget_is_measured_against_the_original_input():
    """Stage 3 may already have removed most of the input.

    Measuring the budget against what survived would make a log file that
    deduped 107k -> 6.5k tokens look like it needed no compression at all *and*
    report a ratio against the wrong denominator.
    """
    chunks = [_chunk(f"chunk {i}", i, 100, 0.5) for i in range(10)]  # 1000 tokens
    result = select_within_budget(chunks, original_tokens=10_000, budget_ratio=0.30)

    assert result.budget_tokens == 3000
    assert result.used_tokens == 1000, "nothing needed dropping"
    assert result.original_tokens == 10_000


def test_output_preserves_original_document_order():
    chunks = [_chunk(f"chunk {i}", i, 50, (i * 7) % 10 / 10) for i in range(15)]
    result = select_within_budget(chunks, budget_tokens=300)

    orders = [c.order for c in result.kept]
    assert orders == sorted(orders)


def test_empty_input():
    result = select_within_budget([])
    assert result.kept == []
    assert result.metrics.status == StageStatus.SKIPPED


# ---------------------------------------------------------------------------
# Protection - the hard rule
# ---------------------------------------------------------------------------
def test_the_user_query_is_never_dropped():
    """Hard rule: compressing away the question is a bug, not a trade-off."""
    query = _chunk("What caused the outage?", 0, 60, 0.0, ChunkKind.QUERY)
    instruction = _chunk("Answer from the context only.", 1, 40, 0.0, ChunkKind.INSTRUCTION)
    filler = [_chunk(f"filler {i}", i + 2, 100, 0.99) for i in range(20)]

    result = select_within_budget([query, instruction, *filler], budget_tokens=200)
    kept_ids = {c.id for c in result.kept}

    assert query.id in kept_ids
    assert instruction.id in kept_ids
    assert query.drop_reason == KEPT_PROTECTED
    assert query.selected is True


def test_protected_chunks_over_budget_are_still_kept():
    """Better to exceed the budget than to answer a question we deleted."""
    query = _chunk("A very long question " * 40, 0, 400, 0.0, ChunkKind.QUERY)
    result = select_within_budget([query], budget_tokens=50)

    assert result.kept == [query]
    assert result.used_tokens > result.budget_tokens
    assert "over budget" in (result.metrics.note or "")


# ---------------------------------------------------------------------------
# Dependency preservation
# ---------------------------------------------------------------------------
def _dependency_fixture():
    """A caller the selector wants, and a helper it does not."""
    caller = _chunk(
        "def handler(payload):\n    return compute_checksum(payload)\n",
        0, 40, 0.95, ChunkKind.FUNCTION, symbol="handler",
    )
    helper = _chunk(
        "def compute_checksum(payload):\n    return sum(payload) % 65521\n",
        1, 40, 0.01, ChunkKind.FUNCTION, symbol="compute_checksum",
    )
    filler = [
        _chunk(f"def unrelated_{i}(x):\n    return x\n", i + 2, 40, 0.5,
               ChunkKind.FUNCTION, symbol=f"unrelated_{i}")
        for i in range(6)
    ]
    return caller, helper, filler


def test_dropped_definitions_are_re_admitted():
    caller, helper, filler = _dependency_fixture()
    result = select_within_budget([caller, helper, *filler], budget_tokens=200)
    kept_ids = {c.id for c in result.kept}

    assert caller.id in kept_ids
    assert helper.id in kept_ids, "a kept call lost its definition"
    assert helper.drop_reason == KEPT_DEPENDENCY
    assert any(d.symbol == "compute_checksum" for d in result.repaired_dependencies)
    assert result.broken_dependencies == []


def test_dependency_repair_can_be_disabled():
    caller, helper, filler = _dependency_fixture()
    cfg = get_config().with_overrides(
        {"selection": {"preserve_code_dependencies": False}}
    )
    result = BudgetSelector(cfg).run([caller, helper, *filler], budget_tokens=200)

    assert helper.id not in {c.id for c in result.kept}
    assert result.repaired_dependencies == []


def test_unrepairable_dependencies_are_reported_not_hidden():
    """At an extreme budget a break may be unavoidable - it must be visible."""
    caller, helper, _ = _dependency_fixture()
    result = select_within_budget([caller, helper], budget_tokens=45)

    assert caller.id in {c.id for c in result.kept}
    assert helper.id not in {c.id for c in result.kept}
    assert any(d.symbol == "compute_checksum" for d in result.broken_dependencies)
    assert result.metrics.details["dependencies_broken"] >= 1


def test_most_referenced_definition_is_repaired_first():
    popular = _chunk("def shared_helper(x):\n    return x\n", 0, 40, 0.01,
                     ChunkKind.FUNCTION, symbol="shared_helper")
    rare = _chunk("def lonely_helper(x):\n    return x\n", 1, 40, 0.01,
                  ChunkKind.FUNCTION, symbol="lonely_helper")
    callers = [
        _chunk(f"def caller_{i}(x):\n    return shared_helper(x)\n", i + 2, 40, 0.9,
               ChunkKind.FUNCTION, symbol=f"caller_{i}")
        for i in range(3)
    ]
    solo = _chunk("def solo(x):\n    return lonely_helper(x)\n", 5, 40, 0.9,
                  ChunkKind.FUNCTION, symbol="solo")

    # Room for the callers plus exactly one helper.
    result = select_within_budget([popular, rare, *callers, solo], budget_tokens=225)
    kept_ids = {c.id for c in result.kept}
    assert popular.id in kept_ids, "the helper three chunks depend on was not prioritised"


# ---------------------------------------------------------------------------
# Symbol analysis
# ---------------------------------------------------------------------------
def test_docstring_prose_is_not_read_as_a_code_reference():
    """Regression: "access/refresh token pair" registered as a call to refresh()."""
    text = (
        'def issue(self):\n'
        '    """Issue an access/refresh token pair for a subject."""\n'
        '    return build_claims()\n'
    )
    names = code_identifiers(text)
    assert "refresh" not in names, "docstring prose leaked into the symbol graph"
    assert "build_claims" in names


def test_comments_are_not_read_as_code_references():
    text = "def f(x):\n    # calls legacy_helper when x is negative\n    return x\n"
    assert "legacy_helper" not in code_identifiers(text)


def test_defined_names_cover_methods_and_their_class():
    chunk = _chunk("def issue(self): pass", 0, 10, 0.5, ChunkKind.METHOD,
                   symbol="TokenService.issue")
    names = defined_names(chunk)
    assert {"TokenService.issue", "issue", "TokenService"} <= names


def test_prose_chunks_have_no_symbol_graph():
    chunk = _chunk("The service called validate_username on startup.", 0, 20, 0.5)
    assert defined_names(chunk) == set()
    assert referenced_names(chunk) == set()


def test_self_reference_is_not_a_dependency():
    recursive = _chunk(
        "def walk(node):\n    return [walk(c) for c in node.children]\n",
        0, 40, 0.9, ChunkKind.FUNCTION, symbol="walk",
    )
    index = build_symbol_index([recursive])
    assert find_broken([recursive], index) == []


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
def test_every_dropped_chunk_is_auditable():
    chunks = [_chunk(f"chunk number {i}", i, 100, 1.0 - i / 30) for i in range(20)]
    result = select_within_budget(chunks, budget_tokens=500)

    trail = result.audit_trail()
    assert len(trail) == len(result.dropped)
    for entry in trail:
        assert entry["reason"] == DROPPED_BUDGET
        assert entry["tokens"] > 0
        assert entry["density"] is not None
        assert entry["preview"]
        assert entry["lines"]


def test_kept_chunks_record_why_they_were_kept():
    chunks = [_chunk(f"chunk {i}", i, 100, 1.0 - i / 30) for i in range(10)]
    result = select_within_budget(chunks, budget_tokens=350)
    for chunk in result.kept:
        assert chunk.drop_reason in {KEPT_DENSITY, KEPT_PROTECTED, KEPT_DEPENDENCY}
        assert chunk.selected is True
    for chunk in result.dropped:
        assert chunk.selected is False


def test_metrics_reconcile():
    chunks = [_chunk(f"chunk {i}", i, 100, 1.0 - i / 30) for i in range(20)]
    result = select_within_budget(chunks, budget_tokens=700)
    details = result.metrics.details

    assert len(result.kept) + len(result.dropped) == len(chunks)
    assert details["dropped_chunks"] == len(result.dropped)
    assert (
        details["dropped_tokens"] + result.used_tokens
        == sum(c.token_count for c in chunks)
    )
