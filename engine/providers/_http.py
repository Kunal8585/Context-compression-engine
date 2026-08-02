"""One place that knows how to POST to a model vendor and fail informatively.

Every provider funnels through :func:`post_json` so that timeouts, rate limits
and 5xx responses turn into the same two exception types no matter whose API
produced them - which is what lets the fallback chain stay vendor-agnostic.

The other job here is keeping the key out of the failure path. Vendor error
bodies are truncated (a 4 KB HTML error page helps nobody) and redacted, and
Google's key is passed as a header by the caller rather than the ``?key=``
query parameter its quickstart suggests, because ``requests`` puts the full URL
into every exception it raises.
"""

from __future__ import annotations

import json
from typing import Any

from .base import ProviderFailed, ProviderUnavailable
from .keys import redact

#: Vendor error bodies are for diagnosis, not for reading in full.
_MAX_ERROR_CHARS = 320

#: Status codes worth saying something specific about, because the fix differs.
_STATUS_HINTS = {
    401: "rejected the key (401)",
    403: "forbidden (403) - key lacks access to this model",
    404: "model or endpoint not found (404)",
    413: "request too large (413) - lower the batch size",
    429: "rate limited or out of quota (429)",
}


def post_json(
    provider: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    """POST JSON and return the decoded body, or raise a normalised error."""
    import requests

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    except requests.exceptions.Timeout:
        raise ProviderFailed(
            provider, f"timed out after {timeout_s:.0f}s", kind="timeout"
        ) from None
    except requests.exceptions.ConnectionError as exc:
        # Unreachable is "skip me" rather than "I failed": a local Ollama that
        # is simply not running should read the same as an unconfigured key.
        raise ProviderUnavailable(provider, f"unreachable ({_brief(exc)})") from None
    except Exception as exc:  # noqa: BLE001
        raise ProviderFailed(provider, _brief(exc)) from None

    if response.status_code >= 400:
        hint = _STATUS_HINTS.get(response.status_code, f"HTTP {response.status_code}")
        raise ProviderFailed(provider, f"{hint}: {_body_excerpt(response)}")

    try:
        return response.json()
    except Exception:  # noqa: BLE001 - a 200 that is not JSON is still a failure
        raise ProviderFailed(
            provider, f"non-JSON response: {_clip(response.text)}"
        ) from None


def get_json(
    provider: str, url: str, headers: dict[str, str], timeout_s: float
) -> dict[str, Any]:
    """GET JSON, used for the local-provider reachability probe."""
    import requests

    try:
        response = requests.get(url, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise ProviderFailed(provider, f"timed out after {timeout_s:.0f}s") from None
    except Exception as exc:  # noqa: BLE001
        raise ProviderUnavailable(provider, _brief(exc)) from None


def dig(provider: str, payload: dict[str, Any], *path: Any) -> Any:
    """Walk a response shape, failing with the shape rather than a KeyError.

    Vendors change response envelopes and free tiers return partial ones (a
    Gemini safety block yields ``candidates: []``). A raw ``KeyError: 0`` in a
    traceback is unreadable; this says which vendor returned what instead.
    """
    cursor: Any = payload
    for step in path:
        try:
            cursor = cursor[step]
        except (KeyError, IndexError, TypeError):
            raise ProviderFailed(
                provider,
                f"unexpected response shape at {'.'.join(map(str, path))}: "
                f"{_clip(json.dumps(payload, default=str), 220)}",
            ) from None
    return cursor


def _body_excerpt(response: Any) -> str:
    """The useful sentence out of a vendor error, redacted and clipped."""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return _clip(response.text)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return _clip(str(error["message"]))
        if isinstance(error, str):
            return _clip(error)
        for key in ("message", "detail"):
            if body.get(key):
                return _clip(str(body[key]))
    return _clip(json.dumps(body, default=str))


def _brief(exc: Exception) -> str:
    return _clip(f"{type(exc).__name__}: {exc}")


def _clip(text: str, limit: int = _MAX_ERROR_CHARS) -> str:
    text = redact(" ".join((text or "").split()))
    return text if len(text) <= limit else text[:limit] + "..."
