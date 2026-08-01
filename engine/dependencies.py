"""Symbol dependency tracking for code chunks.

At an aggressive budget the greedy selector will happily keep a function that
calls ``verify_password`` while dropping the chunk that defines it. The kept
code then reads as if it references something that does not exist, and a model
asked "what does authenticate() do?" has to guess. That is a silent correctness
failure the compression ratio will never show.

This module builds a lightweight symbol graph - which chunk *defines* each name,
which chunks *reference* it - so the selector can repair those breaks with its
leftover budget and report whatever it could not fix.

It is deliberately lexical rather than a real resolver. A full import graph
would need cross-file analysis and per-language scope rules; matching referenced
identifiers against the set of names actually defined in this context catches
the case that matters (a call whose definition was cut) at a fraction of the
cost. False positives are cheap - we keep a chunk we might not have needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .structural import _KEYWORDS, _TOKEN_RE
from .types import Chunk, ChunkKind

_ATTRIBUTE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)")

# Names too common to treat as evidence of a dependency. Matching `get` or
# `run` against a chunk that happens to define a method of that name produces
# noise, not dependencies.
_BUILTINS = frozenset(
    """
    abs all any bool bytes callable chr dict dir enumerate filter float format
    frozenset getattr hasattr hash hex id input int isinstance issubclass items
    iter keys len list map max min next object open ord print range repr
    reversed round set setattr sorted str sum super tuple type vars zip values
    append add copy extend get index insert join pop push remove replace reverse
    sort split strip update write read close send test main self cls args kwargs
    error warn info debug log name value data result response request config
    """.split()
)

_MIN_NAME_LENGTH = 3


@dataclass(frozen=True)
class BrokenDependency:
    """A kept chunk references a name whose definition was dropped."""

    referrer_id: str
    symbol: str
    definition_id: str
    definition_tokens: int

    def to_dict(self) -> dict:
        return {
            "referrer_id": self.referrer_id,
            "symbol": self.symbol,
            "definition_id": self.definition_id,
            "definition_tokens": self.definition_tokens,
        }


def is_code(chunk: Chunk) -> bool:
    return chunk.kind in ChunkKind.CODE_KINDS


def defined_names(chunk: Chunk) -> set[str]:
    """Names this chunk defines and that others could reference."""
    if not is_code(chunk) or not chunk.symbol:
        return set()
    names = {chunk.symbol}
    # `TokenService.issue` is referenced as `.issue(...)` or via the class name.
    if "." in chunk.symbol:
        owner, _, member = chunk.symbol.rpartition(".")
        names.update({member, owner})
    return {n for n in names if len(n) >= _MIN_NAME_LENGTH}


def code_identifiers(text: str) -> set[str]:
    """Identifiers in *code*, excluding comments and string literals.

    Reuses the structural lexer so prose never counts as a reference. Without
    this, `TokenService.issue`'s docstring - "Issue an access/refresh token
    pair" - reads as a call to `refresh()`, because `/` is a word boundary.
    """
    names: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        if match.lastgroup == "name":
            names.add(match.group())
    return names


def referenced_names(chunk: Chunk) -> set[str]:
    """Names this chunk mentions, minus keywords and generic vocabulary."""
    if not is_code(chunk):
        return set()
    names = code_identifiers(chunk.text)
    names.update(_ATTRIBUTE.findall(chunk.text))
    return {
        name
        for name in names
        if len(name) >= _MIN_NAME_LENGTH
        and name not in _KEYWORDS
        and name not in _BUILTINS
    }


def build_symbol_index(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """name -> chunks defining it (a name can be defined more than once)."""
    index: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        for name in defined_names(chunk):
            index.setdefault(name, []).append(chunk)
    return index


def find_broken(
    kept: list[Chunk],
    index: dict[str, list[Chunk]],
    kept_ids: set[str] | None = None,
) -> list[BrokenDependency]:
    """Dependencies of `kept` whose defining chunk is not itself kept."""
    kept_ids = kept_ids if kept_ids is not None else {c.id for c in kept}
    broken: list[BrokenDependency] = []
    seen: set[tuple[str, str]] = set()

    for chunk in kept:
        if not is_code(chunk):
            continue
        own = defined_names(chunk)
        for name in referenced_names(chunk):
            if name in own:
                continue  # self-reference, e.g. recursion
            definitions = index.get(name)
            if not definitions:
                continue  # defined elsewhere, or not a real symbol
            if any(definition.id in kept_ids for definition in definitions):
                continue  # at least one definition survived
            definition = min(definitions, key=lambda c: c.token_count)
            key = (chunk.id, name)
            if key in seen:
                continue
            seen.add(key)
            broken.append(
                BrokenDependency(
                    referrer_id=chunk.id,
                    symbol=name,
                    definition_id=definition.id,
                    definition_tokens=definition.token_count,
                )
            )
    return broken
