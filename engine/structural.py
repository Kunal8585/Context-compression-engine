"""Structural signatures for code chunks.

Sentence embeddings are trained on prose and are weak at code: on our sample
corpus, four validators that differ *only* in an identifier name score 0.62-0.77
cosine - nowhere near a safe near-duplicate threshold. Raising the threshold to
catch them would start merging genuinely different functions.

So code gets the same treatment logs get. A log record is deduped on its
template - the message with volatile fields blanked. A code chunk is deduped on
its *structural signature* - the token stream with identifiers and string
literals blanked, keywords, operators and **numbers preserved**.

Preserving numbers is the load-bearing decision::

    def validate_username(username):   len(username) > 64    ->  signature A
    def validate_tenant_id(tenant_id): len(tenant_id) > 64   ->  signature A   (collapses)
    def validate_device_id(device_id): len(device_id) > 64   ->  signature A   (collapses)
    def validate_password(password):   len(password) > 128   ->  signature B   (kept)

The three genuinely interchangeable validators collapse to one; the password
validator survives on its own because its limit differs. Blanking numbers too
would have silently merged a 64-character limit with a 128-character one, which
is exactly the kind of loss the reasoning-retention metric punishes.

The ``min_tokens`` guard is the other safety rail. Short chunks have little
structure to distinguish them - four one-line exception classes all reduce to
``class <ID> ( <ID> ) : <STR>`` - and collapsing them saves almost no budget.
Below the threshold we simply do not try.
"""

from __future__ import annotations

import hashlib
import re

from .types import ChunkKind

# Keywords are kept verbatim; everything else that looks like a name is blanked.
# Python and JavaScript sets are merged - a Python file will not contain `const`,
# so there is no ambiguity in practice.
_KEYWORDS = frozenset(
    """
    and as assert async await break case catch class const continue def default
    del delete do elif else except export extends finally for from function
    global if import in instanceof is lambda let new nonlocal not of or pass
    raise return static super switch this throw try typeof var void while with
    yield null undefined true false None True False
    """.split()
)

_TOKEN_RE = re.compile(
    r"""
      (?P<comment>  \#[^\n]* | //[^\n]* | /\*.*?\*/ )
    | (?P<string>   [rRbBuUfF]{0,2}
                    (?: \"\"\".*?\"\"\"
                      | '''.*?'''
                      | "(?:\\.|[^"\\\n])*"
                      | '(?:\\.|[^'\\\n])*'
                      | `(?:\\.|[^`\\])*`
                    ) )
    | (?P<number>   0[xXbBoO][0-9a-fA-F_]+ | \d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)? )
    | (?P<name>     [A-Za-z_$][A-Za-z0-9_$]* )
    | (?P<op>       \S )
    """,
    re.VERBOSE | re.DOTALL,
)

#: Chunk kinds eligible for structural dedup. Prose has no meaningful "shape",
#: and module_level blocks are imports and constants whose content *is* the
#: information.
DEFAULT_KINDS = frozenset({ChunkKind.FUNCTION, ChunkKind.METHOD, ChunkKind.CLASS})


def tokenize(text: str) -> list[str]:
    """Normalise code into a comparable token stream."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind == "comment":
            continue  # prose; two bodies differing only in comments are the same shape
        if kind == "string":
            tokens.append("<STR>")
        elif kind == "number":
            tokens.append(value.replace("_", ""))  # 600_000 == 600000
        elif kind == "name":
            tokens.append(value if value in _KEYWORDS else "<ID>")
        else:
            tokens.append(value)
    return tokens


def structural_signature(text: str, kind: str) -> str | None:
    """Return a hashable signature for a code chunk, or None if not applicable."""
    tokens = tokenize(text)
    if not tokens:
        return None
    digest = hashlib.blake2b(
        " ".join(tokens).encode("utf-8"), digest_size=16
    ).hexdigest()
    # The kind is part of the key: a class and a function that happen to reduce
    # to the same token stream are still different things.
    return f"str:{kind}:{digest}"
