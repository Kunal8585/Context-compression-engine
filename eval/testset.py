"""Test set loading and validation.

A test set whose answers are not actually present in its contexts would
understate retention - the compressed run would fail items the original run
could never have passed either. :func:`validate` asserts every key fact appears
in its context before a single model call is made.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from engine.config import PROJECT_ROOT

DEFAULT_TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.json"
CORPUS_ROOT = PROJECT_ROOT / "data" / "sample_corpus"

#: A key fact is either a literal, or a list of acceptable alternatives
#: ("600,000" / "600000" / "600_000" are the same fact).
Fact = str | list[str]


@dataclass
class EvalContext:
    key: str
    source: str
    text: str
    description: str = ""
    lines: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return Path(self.source).name


@dataclass
class TestItem:
    id: str
    context_key: str
    question: str
    expected_answer: str
    key_facts: list[Fact] = field(default_factory=list)


@dataclass
class TestSet:
    name: str
    description: str
    contexts: dict[str, EvalContext]
    items: list[TestItem]

    def context_for(self, item: TestItem) -> EvalContext:
        return self.contexts[item.context_key]


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop thousands separators."""
    return re.sub(r"\s+", " ", text.lower()).replace(",", "").replace("_", "")


def fact_present(fact: Fact, text: str) -> bool:
    """Is this fact stated in `text`?

    Short or numeric facts are matched on word boundaries: a bare ``8`` must not
    be satisfied by the ``8`` inside ``86,000``.
    """
    alternatives = fact if isinstance(fact, list) else [fact]
    haystack = _normalise(text)
    for alternative in alternatives:
        needle = _normalise(alternative)
        if not needle:
            continue
        if needle.isdigit():
            # Guard against *digit* neighbours only, so `8` is not satisfied by
            # the 8 in `86000`, while `7912` is still satisfied by `7912ms`.
            # A stricter \w boundary rejected every value written with a unit,
            # which under-counted correct answers in both conditions.
            if re.search(rf"(?<![\d.]){re.escape(needle)}(?!\d)", haystack):
                return True
        elif len(needle) <= 4:
            if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
                return True
        elif needle in haystack:
            return True
    return False


def load_testset(path: str | Path | None = None) -> TestSet:
    resolved = Path(path or DEFAULT_TESTSET)
    raw = json.loads(resolved.read_text(encoding="utf-8"))

    contexts: dict[str, EvalContext] = {}
    for key, spec in raw["contexts"].items():
        source_path = CORPUS_ROOT / spec["source"]
        text = source_path.read_text(encoding="utf-8")
        line_range = spec.get("lines")
        if line_range:
            lines = text.splitlines(keepends=True)
            text = "".join(lines[line_range[0] : line_range[1]])
        contexts[key] = EvalContext(
            key=key,
            source=spec["source"],
            text=text,
            description=spec.get("description", ""),
            lines=tuple(line_range) if line_range else None,
        )

    items = [
        TestItem(
            id=entry["id"],
            context_key=entry["context"],
            question=entry["question"],
            expected_answer=entry["expected_answer"],
            key_facts=entry.get("key_facts", []),
        )
        for entry in raw["items"]
    ]
    return TestSet(
        name=raw.get("name", resolved.stem),
        description=raw.get("description", ""),
        contexts=contexts,
        items=items,
    )


def validate(testset: TestSet) -> list[str]:
    """Return a list of problems. Empty means the test set is answerable."""
    problems: list[str] = []
    seen: set[str] = set()

    for item in testset.items:
        if item.id in seen:
            problems.append(f"{item.id}: duplicate id")
        seen.add(item.id)

        context = testset.contexts.get(item.context_key)
        if context is None:
            problems.append(f"{item.id}: unknown context {item.context_key!r}")
            continue
        if not item.key_facts:
            problems.append(f"{item.id}: no key_facts, so it cannot be scored")
        for fact in item.key_facts:
            if not fact_present(fact, context.text):
                problems.append(
                    f"{item.id}: key fact {fact!r} does not appear in context "
                    f"{item.context_key!r} - the question is unanswerable"
                )
    return problems
