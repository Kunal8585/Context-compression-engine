"""Generation providers for stage 6 and the eval harness.

=============  ==============================  ==========  ======================
provider       default model                   tier        note
=============  ==============================  ==========  ======================
groq           llama-3.3-70b-versatile         free        fastest by a wide margin
gemini         gemini-2.0-flash                free        generous daily quota
openrouter     configurable (:free models)     free        model set changes often
openai         gpt-4o-mini                     paid        quality baseline
local          ollama llama3.2:3b              free        no network, no key
=============  ==============================  ==========  ======================

Four of the five speak the OpenAI chat-completions shape, so they share one
implementation and differ only in endpoint, model and key. Gemini and Ollama
have their own envelopes and get their own classes.

OpenRouter's model is read from ``OPENROUTER_MODEL`` (falling back to config)
on purpose: which models carry the ``:free`` tag changes month to month, and
pinning one in code guarantees a 404 at some point after this is written.
"""

from __future__ import annotations

import logging

from . import keys
from ._http import dig, get_json, post_json
from .base import GenerationProvider, ProviderFailed

log = logging.getLogger(__name__)


class _OpenAICompatible(GenerationProvider):
    """The chat-completions shape, shared by OpenAI, Groq and OpenRouter."""

    endpoint = ""

    def __init__(
        self,
        name: str,
        endpoint: str,
        model: str,
        env_key: str,
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.endpoint = endpoint
        self.model = model
        self.env_key = env_key
        self.default_timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {keys.get(self.env_key)}",
            "Content-Type": "application/json",
        }

    def _generate(
        self, prompt: str, max_tokens: int, timeout_s: float, system: str | None
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = post_json(
            self.name,
            self.endpoint,
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            self._headers(),
            timeout_s,
        )
        # A provider that returns an error inside a 200 (OpenRouter does this
        # for an unavailable free model) must fail, not return empty text.
        if isinstance(body.get("error"), dict):
            raise ProviderFailed(self.name, body["error"].get("message", "error"))
        message = dig(self.name, body, "choices", 0, "message")
        text = (message.get("content") or "").strip()
        if not text:
            # An empty completion is a failure, not an answer. Reasoning models
            # served through OpenRouter routinely return 200 with everything in
            # a `reasoning` field and `content` empty; a caller that accepted
            # that would silently treat "no answer" as "the answer is nothing" -
            # and stage 6 would keep the original while the chain never learned
            # to fall through to a provider that actually replies.
            finish = (body.get("choices") or [{}])[0].get("finish_reason", "unknown")
            detail = "reasoning-only response" if message.get("reasoning") else "no content"
            raise ProviderFailed(
                self.name, f"empty completion ({detail}, finish_reason={finish})"
            )
        return text


class GroqProvider(_OpenAICompatible):
    """Groq: free tier, and by far the fastest of the hosted options."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", timeout_s: float = 25.0):
        super().__init__(
            "groq",
            "https://api.groq.com/openai/v1/chat/completions",
            model,
            "GROQ_API_KEY",
            timeout_s,
        )


class OpenRouterProvider(_OpenAICompatible):
    """OpenRouter, pointed at whichever model is currently free-tagged."""

    def __init__(self, model: str | None = None, timeout_s: float = 40.0):
        import os

        resolved = (
            os.environ.get("OPENROUTER_MODEL")
            or model
            or "meta-llama/llama-3.3-70b-instruct:free"
        )
        super().__init__(
            "openrouter",
            "https://openrouter.ai/api/v1/chat/completions",
            resolved,
            "OPENROUTER_API_KEY",
            timeout_s,
        )


class OpenAIGenerationProvider(_OpenAICompatible):
    """gpt-4o-mini: paid, cheap, and the quality baseline everything else is
    compared against."""

    def __init__(self, model: str = "gpt-4o-mini", timeout_s: float = 30.0):
        super().__init__(
            "openai",
            "https://api.openai.com/v1/chat/completions",
            model,
            "OPENAI_API_KEY",
            timeout_s,
        )


class GeminiGenerationProvider(GenerationProvider):
    """gemini-2.0-flash via generateContent. Free tier, own response envelope."""

    name = "gemini"
    env_key = "GOOGLE_API_KEY"
    base = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str = "gemini-2.0-flash", timeout_s: float = 30.0):
        super().__init__()
        self.model = model
        self.default_timeout_s = timeout_s

    def _generate(
        self, prompt: str, max_tokens: int, timeout_s: float, system: str | None
    ) -> str:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        body = post_json(
            self.name,
            f"{self.base}/{model_path}:generateContent",
            payload,
            # Header rather than ?key= so the secret cannot ride along in the
            # URL attached to a requests exception.
            {
                "x-goog-api-key": keys.get(self.env_key),
                "Content-Type": "application/json",
            },
            timeout_s,
        )
        candidates = body.get("candidates") or []
        if not candidates:
            # An empty candidate list is a safety block or a quota refusal, not
            # an empty answer; say which so the chain's reason is useful.
            reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderFailed(self.name, f"no completion returned ({reason})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            finish = candidates[0].get("finishReason", "unknown")
            raise ProviderFailed(self.name, f"empty completion (finishReason={finish})")
        return text


class OllamaProvider(GenerationProvider):
    """Local Ollama. No key, no network, no cost - the tail of every chain.

    Kept as a peer of the hosted providers rather than deleted: it is the only
    one that still answers with the wifi off, and it is what makes "the chain
    never runs out" true on a laptop.
    """

    name = "local"
    env_key = None

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        timeout_s: float = 45.0,
        num_ctx: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.host = host.rstrip("/")
        self.default_timeout_s = timeout_s
        self.num_ctx = num_ctx
        self._probe: tuple[bool, str] | None = None

    def configured(self, refresh: bool = False) -> tuple[bool, str]:
        """Reachable *and* holding the model. Cached; a probe per call would
        double the request count of a 12-chunk stage."""
        if self._probe is not None and not refresh:
            return self._probe
        try:
            body = get_json(self.name, f"{self.host}/api/tags", {}, 3.0)
        except Exception as exc:  # noqa: BLE001
            self._probe = (False, f"ollama unreachable at {self.host}: {exc}")
            return self._probe
        names = [entry.get("name", "") for entry in body.get("models", [])]
        wanted = self.model
        if any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names):
            self._probe = (True, "")
        else:
            self._probe = (
                False,
                f"model {wanted!r} not pulled (have: {names or 'none'})",
            )
        return self._probe

    def describe(self) -> dict:
        return {**super().describe(), "host": self.host}

    def _generate(
        self, prompt: str, max_tokens: int, timeout_s: float, system: str | None
    ) -> str:
        options: dict = {"temperature": 0, "num_predict": max_tokens}
        if self.num_ctx:
            # Ollama defaults to a 2048-token window and SILENTLY TRUNCATES
            # past it. Every "original context" measurement in the harness
            # would be fiction without this being set explicitly.
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        body = post_json(
            self.name, f"{self.host}/api/generate", payload, {}, timeout_s
        )
        self.last_prompt_tokens = int(body.get("prompt_eval_count") or 0)
        return (body.get("response") or "").strip()


#: Name used in config.yaml -> constructor.
GENERATION_PROVIDERS = {
    "groq": GroqProvider,
    "gemini": GeminiGenerationProvider,
    "openrouter": OpenRouterProvider,
    "openai": OpenAIGenerationProvider,
    "local": OllamaProvider,
}
