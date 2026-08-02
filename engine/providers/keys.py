"""API key access. Environment only, and one-way.

Two rules this module exists to enforce:

**Keys come from the environment, never from config.** ``config.yaml`` is
committed and shown to judges; it names providers and models, never secrets. A
local ``.env`` is loaded once as a convenience for development, and never
overrides a variable the real environment already set.

**Nothing here hands a key back to anything that can print it.** The only
public question you may ask about a key is whether it is present. ``get`` is
deliberately the single accessor, used exactly at the point of building an HTTP
header, so there is one place to audit. :func:`redact` is the belt-and-braces
pass applied to every error string that leaves a provider, because a stray
traceback is the realistic way a key escapes - not a deliberate log line.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Supported keys -> where a reader gets one. Drives ``.env.example`` and the
#: ``/health`` provider report. Every entry here is optional: a provider whose
#: key is absent is skipped in a fallback chain, never an error on its own.
KEY_REGISTRY: dict[str, str] = {
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
    "GROQ_API_KEY": "https://console.groq.com/keys",
    "GOOGLE_API_KEY": "https://aistudio.google.com/apikey",
    "COHERE_API_KEY": "https://dashboard.cohere.com/api-keys",
    "OPENROUTER_API_KEY": "https://openrouter.ai/keys",
}

#: Values that mean "the reader copied .env.example and never filled it in".
#: Treating these as configured produces a confusing 401 instead of an honest
#: "no key set", so they are filtered at the source.
_PLACEHOLDER = re.compile(r"your-key-here|your_key_here|sk-your|changeme|xxx+", re.I)

_LOADED = False


class MissingKey(RuntimeError):
    """Raised when a provider is asked to run without its key configured."""


def load_dotenv(path: str | Path | None = None, *, force: bool = False) -> int:
    """Merge a local ``.env`` into ``os.environ``. Returns how many were set.

    Never overrides an existing environment variable: a deployment that injects
    real secrets must always win over a stale file left in the working tree.
    """
    global _LOADED
    if _LOADED and path is None and not force:
        return 0
    resolved = Path(path) if path else PROJECT_ROOT / ".env"
    if path is None:
        _LOADED = True
    if not resolved.is_file():
        return 0
    applied = 0
    for line in resolved.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name or not value or name in os.environ:
            continue
        os.environ[name] = value
        applied += 1
    if applied:
        # Names only. Never the values.
        log.debug("loaded %d variable(s) from %s", applied, resolved.name)
    return applied


def has(name: str) -> bool:
    """True when ``name`` is set to something that is not a placeholder."""
    load_dotenv()
    value = (os.environ.get(name) or "").strip()
    return bool(value) and not _PLACEHOLDER.search(value)


def get(name: str) -> str:
    """Return a key for immediate use in a request header. Never log the result."""
    load_dotenv()
    value = (os.environ.get(name) or "").strip()
    if not value or _PLACEHOLDER.search(value):
        raise MissingKey(
            f"{name} is not configured. See .env.example for where to get one "
            f"({KEY_REGISTRY.get(name, 'provider dashboard')})."
        )
    return value


def configured_keys() -> dict[str, bool]:
    """Presence map over every supported key. Safe to serialise anywhere."""
    return {name: has(name) for name in KEY_REGISTRY}


#: Long opaque tokens: OpenAI/Groq ``sk-``/``gsk_``, Google ``AIza``, and the
#: generic 32+ char base64-ish blob every other vendor issues.
_SECRET_SHAPES = (
    re.compile(r"\b(?:sk|gsk|xai|sk-or|sk-proj)[-_][A-Za-z0-9\-_]{12,}", re.I),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{12,}=*", re.I),
    re.compile(r"(?<=[?&]key=)[A-Za-z0-9\-._~]{12,}"),
)


def redact(text: str) -> str:
    """Strip anything key-shaped from a string bound for a log or a response.

    Applied to every provider error before it propagates. The providers already
    avoid putting keys in URLs (Google's key goes in a header, not a query
    string, precisely so a ``requests`` exception cannot carry it) - this is the
    second line of defence for the error text a vendor sends *back*.
    """
    if not text:
        return text
    for pattern in _SECRET_SHAPES:
        text = pattern.sub("[redacted]", text)
    # A configured key echoed verbatim by a vendor error is the one shape a
    # generic pattern can miss, so substitute the real values too.
    for name in KEY_REGISTRY:
        value = (os.environ.get(name) or "").strip()
        if len(value) >= 12 and value in text:
            text = text.replace(value, "[redacted]")
    return text
