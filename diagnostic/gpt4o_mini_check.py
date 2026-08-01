"""Bounded diagnostic: is the local 3B model the retention bottleneck?

    export OPENAI_API_KEY=sk-...
    python -m diagnostic.gpt4o_mini_check

**This is diagnostic-only and is deliberately not wired into the pipeline.**
Nothing here is imported by ``/compress`` or ``/evaluate``, and the shipped
answering model stays local regardless of what this reports. Running entirely
on-device with no API key is the project's differentiator; this script exists to
answer one question and then get out of the way.

The question: stage 8 showed 80.8% of key facts survive compression but the
local 3B model retrieves only ~57% of them, and on 4 items it answered
"NOT FOUND" while the fact was verifiably present in the compressed text. Is
that a model-capability limit, or is something about the compressed context's
structure at fault?

Method: take those exact items from the existing eval report, give GPT-4o-mini
the *same* compressed context and the *same* question, and score with the *same*
fact-matching function the harness uses. Only the answering model changes, so
the comparison isolates retrieval capability.

Four API calls. No retries, no backoff - a one-shot diagnostic, not production
code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from engine.config import PROJECT_ROOT, get_config
from engine.pipeline import CompressionPipeline
from eval.harness import SYSTEM_PROMPT
from eval.scoring import key_fact_recall
from eval.testset import fact_present, load_testset

REPORT_IN = PROJECT_ROOT / "reports" / "latest.json"
REPORT_OUT = PROJECT_ROOT / "eval" / "diagnostic_report.json"
MODEL = "gpt-4o-mini"


def die(message: str) -> None:
    sys.stdout.flush()  # keep stderr from interleaving ahead of stdout
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def explain_openai_error(exc: Exception) -> str:
    """Turn an SDK error into something actionable."""
    text = str(exc)
    if "insufficient_quota" in text or "credit balance" in text.lower():
        return (
            "the API key is valid but the OpenAI account has no credits.\n"
            "  Add credits at https://platform.openai.com/settings/organization/billing\n"
            "  This diagnostic is optional - the shipped pipeline needs no key, and\n"
            "  the fact-survival decomposition stands without it."
        )
    if "invalid_api_key" in text or "Incorrect API key" in text:
        return "the OPENAI_API_KEY in .env is not a valid key."
    if "rate_limit" in text:
        return "rate limited by OpenAI; wait a moment and re-run."
    return text


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Read KEY=value lines from a project-local .env into the environment.

    Only used by this diagnostic - the shipped pipeline reads no secrets at all.
    A real environment variable always wins, so exporting overrides the file.
    `.env` is gitignored; keep the key out of the repository.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def select_candidates(report: dict, testset, compressed: dict[str, str]) -> list[dict]:
    """Items the 3B model got wrong *while the fact was present in its context*.

    Read out of the existing eval report rather than re-derived by re-running
    the harness. An item the compressor genuinely stripped is not evidence about
    retrieval, so those are excluded.
    """
    candidates = []
    for entry in report["items"]:
        # Deliberately NOT filtered on verdict == "degraded". An item where the
        # model failed on *both* the full and the compressed context (log-05,
        # tickets-02) is scored "retained" because compression cost nothing -
        # but it is still a case of a present fact going unretrieved, which is
        # exactly what this diagnostic is testing.
        item = next(i for i in testset.items if i.id == entry["id"])
        context = compressed[item.context_key]
        present = [f for f in item.key_facts if fact_present(f, context)]
        if len(present) <= entry["compressed"]["recall"] * len(item.key_facts):
            continue  # compression dropped it; not a retrieval failure
        candidates.append(
            {
                "item": item,
                "local_recall": entry["compressed"]["recall"],
                "local_answer": entry["compressed"]["answer"],
                "facts_present": len(present),
                "facts_total": len(item.key_facts),
            }
        )
    return candidates


def ask_openai(client, context: str, question: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=120,
    )
    return (
        (response.choices[0].message.content or "").strip(),
        (time.perf_counter() - started) * 1000,
    )


def main() -> int:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        die(
            "OPENAI_API_KEY is not set.\n\n"
            "  Either put it in a project-local .env (gitignored):\n"
            "      echo 'OPENAI_API_KEY=sk-...' >> .env\n\n"
            "  ...or export it for one shell session:\n"
            "      export OPENAI_API_KEY=sk-...\n\n"
            "  Then: python -m diagnostic.gpt4o_mini_check\n\n"
            "  Diagnostic-only. The shipped pipeline never calls OpenAI and "
            "needs no key."
        )
    if not REPORT_IN.exists():
        die(f"no eval report at {REPORT_IN}. Run `python -m eval.harness` first.")

    try:
        from openai import OpenAI
    except ImportError:
        die("the `openai` package is not installed (pip install openai)")

    report = json.loads(REPORT_IN.read_text(encoding="utf-8"))
    testset = load_testset()
    budget = report["config"]["budget_ratio"]

    # The pipeline is deterministic (asserted by test_pipeline_is_deterministic),
    # so recompressing reproduces byte-identical context to the eval run.
    pipeline = CompressionPipeline(
        get_config().with_overrides({"abstractive": {"enabled": False}})
    )
    compressed: dict[str, str] = {
        key: pipeline.compress(ctx.text, ctx.name, budget_ratio=budget).compressed_text
        for key, ctx in testset.contexts.items()
    }

    candidates = select_candidates(report, testset, compressed)
    if not candidates:
        print("no retrieval-failure items in the current report; nothing to diagnose")
        return 0

    print(f"diagnostic: {len(candidates)} item(s) where the fact was present in the "
          f"compressed context but the local model missed it")
    print(f"answering model under test: {MODEL} (local baseline: "
          f"{report['config']['downstream_model']})\n")

    client = OpenAI()
    results = []
    for candidate in candidates:
        item = candidate["item"]
        context = compressed[item.context_key]
        try:
            answer, latency_ms = ask_openai(client, context, item.question)
        except Exception as exc:
            die(f"OpenAI call failed for {item.id}: {explain_openai_error(exc)}")
        recall, missing = key_fact_recall(answer, item)

        print(f"--- {item.id}  ({candidate['facts_present']}/{candidate['facts_total']} "
              f"facts present in a {len(context):,}-char compressed context)")
        print(f"    Q  : {item.question}")
        print(f"    3B : recall={candidate['local_recall']:.0%}  "
              f"{candidate['local_answer'][:90]!r}")
        print(f"    4om: recall={recall:.0%}  {answer[:90]!r}")
        if missing:
            print(f"         still missing: {', '.join(missing)}")
        print()

        results.append(
            {
                "id": item.id,
                "context": item.context_key,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "compressed_context_chars": len(context),
                "facts_present_in_context": candidate["facts_present"],
                "facts_total": candidate["facts_total"],
                "local_3b": {
                    "model": report["config"]["downstream_model"],
                    "recall": candidate["local_recall"],
                    "answer": candidate["local_answer"],
                },
                "gpt4o_mini": {
                    "model": MODEL,
                    "recall": recall,
                    "answer": answer,
                    "latency_ms": round(latency_ms, 1),
                    "missing_facts": missing,
                },
                "resolved": recall > candidate["local_recall"],
            }
        )

    local_mean = sum(r["local_3b"]["recall"] for r in results) / len(results)
    remote_mean = sum(r["gpt4o_mini"]["recall"] for r in results) / len(results)
    resolved = sum(1 for r in results if r["resolved"])

    print("=" * 70)
    print(f"  {'item':<14}{'local 3B':>10}{'gpt-4o-mini':>14}   resolved")
    for r in results:
        print(f"  {r['id']:<14}{r['local_3b']['recall']:>9.0%}"
              f"{r['gpt4o_mini']['recall']:>13.0%}   "
              f"{'yes' if r['resolved'] else 'no'}")
    print(f"  {'MEAN':<14}{local_mean:>9.0%}{remote_mean:>13.0%}   "
          f"{resolved}/{len(results)}")
    print(f"\n  retention delta on these items: {remote_mean - local_mean:+.0%}")
    if resolved >= max(1, len(results) // 2):
        verdict = (
            "RETRIEVAL-BOUND - a frontier model recovers facts the 3B model "
            "missed from identical context, so the limiter is the local model, "
            "not the compressor"
        )
    elif resolved:
        verdict = (
            "MIXED - a frontier model recovers some but not most; context "
            "structure may also contribute"
        )
    else:
        verdict = (
            "NOT retrieval-bound - a frontier model fails on the same items, so "
            "the compressed context's structure (marker placement, chunk "
            "ordering) is the more likely cause"
        )
    print(f"  verdict: {verdict}")
    print("=" * 70)

    payload = {
        "purpose": (
            "Diagnostic only. Determines whether the accuracy-retention gap is "
            "local-model retrieval capability or compressed-context structure. "
            "The shipped pipeline's answering model is unchanged and remains "
            "fully local."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_report": report["report_id"],
        "budget_ratio": budget,
        "baseline_model": report["config"]["downstream_model"],
        "diagnostic_model": MODEL,
        "accuracy_metric": "key_fact_recall",
        "items_tested": len(results),
        "aggregate": {
            "local_mean_recall": round(local_mean, 4),
            "diagnostic_mean_recall": round(remote_mean, 4),
            "retention_delta": round(remote_mean - local_mean, 4),
            "items_resolved": resolved,
        },
        "verdict": verdict,
        "items": results,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
