"""Standalone provider smoke test - run this before trusting the pipeline.

    python -m engine.providers.check              # every configured provider
    python -m engine.providers.check --chains     # what the chains resolve to
    python -m engine.providers.check --embedding  # embeddings only

Hits each provider that has a key configured with the smallest real request it
supports, and prints what came back. This is deliberately separate from the
test suite: it costs real quota and needs a network, so it is a thing you run
on purpose, not on every commit.

Providers without a key are reported as `skip`, which is the correct state and
not a failure. The exit code is non-zero only if a provider that *is* configured
failed to answer - i.e. something you believed was working is not.
"""

from __future__ import annotations

import argparse
import sys
import time

from ..config import Config, get_config
from . import keys
from .base import ProviderError
from .chain import EmbeddingChain, GenerationChain
from .embedding import EMBEDDING_PROVIDERS
from .generation import GENERATION_PROVIDERS

PROBE_TEXTS = ["payment pool exhausted after 8000ms", "checkout service timeout"]
PROBE_PROMPT = "Reply with exactly one word: ok"

GREEN, RED, DIM, YELLOW, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[33m", "\033[0m"


def _status(ok: bool | None) -> str:
    if ok is None:
        return f"{YELLOW}skip{RESET}"
    return f"{GREEN} ok {RESET}" if ok else f"{RED}fail{RESET}"


def check_embeddings(cfg: Config) -> list[tuple[str, bool | None, str]]:
    from . import _embedding_provider

    rows: list[tuple[str, bool | None, str]] = []
    for name in EMBEDDING_PROVIDERS:
        provider = _embedding_provider(name, cfg)
        ready, reason = provider.configured()
        if not ready:
            rows.append((name, None, reason))
            continue
        started = time.perf_counter()
        try:
            vectors = provider.embed(PROBE_TEXTS)
        except ProviderError as exc:
            rows.append((name, False, exc.reason))
            continue
        except Exception as exc:  # noqa: BLE001
            rows.append((name, False, keys.redact(f"{type(exc).__name__}: {exc}")))
            continue
        elapsed = (time.perf_counter() - started) * 1000
        detail = (
            f"{len(vectors)} vectors, {len(vectors[0])}-d, "
            f"{elapsed:.0f}ms, {provider.stats.calls} call(s), "
            f"batch limit {provider.batch_limit}"
        )
        rows.append((name, True, detail))
    return rows


def check_generation(cfg: Config) -> list[tuple[str, bool | None, str]]:
    from . import _generation_provider

    rows: list[tuple[str, bool | None, str]] = []
    for name in GENERATION_PROVIDERS:
        provider = _generation_provider(name, cfg, cfg.providers.timeout_s, None)
        ready, reason = provider.configured()
        if not ready:
            rows.append((name, None, reason))
            continue
        started = time.perf_counter()
        try:
            answer = provider.generate(PROBE_PROMPT, max_tokens=16, timeout_s=30)
        except ProviderError as exc:
            rows.append((name, False, exc.reason))
            continue
        except Exception as exc:  # noqa: BLE001
            rows.append((name, False, keys.redact(f"{type(exc).__name__}: {exc}")))
            continue
        elapsed = (time.perf_counter() - started) * 1000
        preview = " ".join(answer.split())[:60] or "(empty)"
        rows.append((name, True, f"{elapsed:.0f}ms, {provider.model} -> {preview!r}"))
    return rows


def check_chains(cfg: Config) -> None:
    """Resolve both chains for real and report which provider actually served."""
    from . import build_embedding_chain, build_generation_chain

    print(f"\n{DIM}chains as configured in config.yaml{RESET}")
    embedding: EmbeddingChain = build_embedding_chain(cfg)
    generation: GenerationChain = build_generation_chain(cfg)
    print(f"  embedding  : {' -> '.join(embedding.names)}")
    print(f"  generation : {' -> '.join(generation.names)}")

    print(f"\n{DIM}resolving each chain against a live call{RESET}")
    try:
        vectors = embedding.embed(PROBE_TEXTS)
        print(f"  embedding  {_status(True)} served by "
              f"{GREEN}{embedding.last_provider}{RESET} ({len(vectors[0])}-d)")
    except Exception as exc:  # noqa: BLE001
        print(f"  embedding  {_status(False)} {keys.redact(str(exc))}")
    for attempt in embedding.last_attempts:
        mark = "skipped" if attempt.skipped else ("used" if attempt.ok else "failed")
        print(f"      {DIM}{attempt.provider:<12} {mark:<8} {attempt.reason}{RESET}")

    try:
        answer = generation.generate(PROBE_PROMPT, max_tokens=16, timeout_s=30)
        print(f"  generation {_status(True)} served by "
              f"{GREEN}{generation.last_provider}{RESET} -> "
              f"{' '.join(answer.split())[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  generation {_status(False)} {keys.redact(str(exc))}")
    for attempt in generation.last_attempts:
        mark = "skipped" if attempt.skipped else ("used" if attempt.ok else "failed")
        print(f"      {DIM}{attempt.provider:<12} {mark:<8} {attempt.reason}{RESET}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine.providers.check", description=__doc__)
    parser.add_argument("--embedding", action="store_true", help="embeddings only")
    parser.add_argument("--generation", action="store_true", help="generation only")
    parser.add_argument("--chains", action="store_true", help="resolve the chains too")
    parser.add_argument("--config", help="alternate config.yaml")
    args = parser.parse_args(argv)

    keys.load_dotenv()
    cfg = get_config(args.config)

    present = keys.configured_keys()
    print(f"{DIM}keys configured (presence only - values are never printed){RESET}")
    for name, ok in present.items():
        print(f"  {'*' if ok else ' '} {name:<22} "
              f"{GREEN + 'set' + RESET if ok else DIM + 'not set' + RESET}")

    both = not (args.embedding or args.generation)
    failures = 0

    if args.embedding or both:
        print(f"\n{DIM}embedding providers{RESET}")
        for name, ok, detail in check_embeddings(cfg):
            print(f"  [{_status(ok)}] {name:<12} {detail}")
            failures += ok is False

    if args.generation or both:
        print(f"\n{DIM}generation providers{RESET}")
        for name, ok, detail in check_generation(cfg):
            print(f"  [{_status(ok)}] {name:<12} {detail}")
            failures += ok is False

    if args.chains:
        check_chains(cfg)

    print()
    if failures:
        print(f"{RED}{failures} configured provider(s) failed{RESET} - "
              f"the chain will fall through them at runtime.")
    else:
        print(f"{GREEN}every configured provider answered.{RESET}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
