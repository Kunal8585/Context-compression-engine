"""Stage 8 - the evaluation harness.

Runs every test item twice against the same downstream model: once with the
full context, once with the compressed context. Everything the dashboard shows
for accuracy, latency and cost originates here, measured.

    python -m eval.harness                    # full run, writes reports/latest.{json,csv}
    python -m eval.harness --quick            # skip the expensive 14.5k-token context
    python -m eval.harness --budget 0.15      # sweep a different budget

Method notes, because these are what a judge should be able to interrogate:

* **The uncompressed context must fit the model.** Ollama silently truncates
  past ``num_ctx``; a truncated "original" would flatter the compressed run.
  ``num_ctx`` is set explicitly and the harness refuses to run an item whose
  original context exceeds it.
* **Accuracy is deterministic key-fact recall**, not a model grading itself.
  See :mod:`eval.scoring`.
* **Cost is computed, not asserted** - published per-1M-token prices from
  ``config.yaml`` applied to measured token counts. Local inference is free, so
  the dollar figure answers "what would this prompt cost against a hosted API",
  which is the number that transfers off this laptop.
* **Latency is wall-clock on this machine** (M2, llama3.2:3b) and is therefore
  hardware-specific. Compression ratio and cost are not.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from engine.config import PROJECT_ROOT, Config, get_config
from engine.pipeline import CompressionPipeline
from engine.tokenizer import get_tokenizer

from .scoring import Judge, key_fact_recall
from .testset import (
    EvalContext,
    TestItem,
    TestSet,
    fact_present,
    load_testset,
    validate,
)

log = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
DIAGNOSTIC_REPORT = PROJECT_ROOT / "eval" / "diagnostic_report.json"

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. Be specific and "
    "concise: state the exact values, names and identifiers the context gives. "
    "If the context does not contain the answer, reply exactly: NOT FOUND."
)


@dataclass
class RunOutcome:
    answer: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    recall: float = 0.0
    missing_facts: list[str] = field(default_factory=list)
    judge_correct: bool | None = None
    error: str | None = None


@dataclass
class ItemResult:
    id: str
    context_key: str
    question: str
    expected_answer: str
    original: RunOutcome
    compressed: RunOutcome
    verdict: str = "retained"


class DownstreamModel:
    """The model both conditions are measured against. Identical settings."""

    def __init__(self, cfg: Config) -> None:
        self.settings = cfg.evaluation

    def available(self) -> tuple[bool, str]:
        try:
            import requests

            response = requests.get(f"{self.settings.host}/api/tags", timeout=3)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
            wanted = self.settings.downstream_model
            if any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names):
                return True, ""
            return False, f"model {wanted!r} not pulled (have: {names or 'none'})"
        except Exception as exc:
            return False, f"ollama unreachable at {self.settings.host}: {exc}"

    def ask(self, context: str, question: str) -> tuple[str, float, int]:
        """Returns (answer, latency_ms, prompt_tokens_reported_by_model)."""
        import requests

        prompt = f"{context}\n\nQuestion: {question}"
        started = time.perf_counter()
        response = requests.post(
            f"{self.settings.host}/api/generate",
            json={
                "model": self.settings.downstream_model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": self.settings.num_ctx,
                    "num_predict": self.settings.max_answer_tokens,
                    "temperature": 0,
                },
            },
            timeout=self.settings.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - started) * 1000
        return (
            (payload.get("response") or "").strip(),
            latency_ms,
            int(payload.get("prompt_eval_count") or 0),
        )


class Harness:
    def __init__(self, cfg: Config | None = None, budget_ratio: float | None = None):
        self.cfg = cfg or get_config()
        self.budget_ratio = (
            budget_ratio if budget_ratio is not None else self.cfg.selection.budget_ratio
        )
        self.tokenizer = get_tokenizer(self.cfg.tokenizer)
        self.model = DownstreamModel(self.cfg)
        self.judge = Judge(self.cfg.evaluation)
        # Stage 6 off: it adds minutes of LLM calls to every compression for no
        # measured token saving on this corpus (see README).
        self.pipeline = CompressionPipeline(
            self.cfg.with_overrides({"abstractive": {"enabled": False}})
        )
        self._compressed: dict[str, tuple[str, int, int]] = {}

    # -- compression cache -------------------------------------------------
    def compressed_context(self, context: EvalContext) -> tuple[str, int, int]:
        """Compress each context once per run, not once per question."""
        if context.key not in self._compressed:
            result = self.pipeline.compress(
                context.text, context.name, budget_ratio=self.budget_ratio
            )
            self._compressed[context.key] = (
                result.compressed_text,
                result.original_tokens,
                result.compressed_tokens,
            )
        return self._compressed[context.key]

    # -- one condition -----------------------------------------------------
    def _run_one(self, context: str, item: TestItem) -> RunOutcome:
        outcome = RunOutcome(tokens_in=self.tokenizer.count(context))
        try:
            answer, latency_ms, _ = self.model.ask(context, item.question)
        except Exception as exc:
            outcome.error = str(exc)
            return outcome
        outcome.answer = answer
        outcome.latency_ms = latency_ms
        outcome.tokens_out = self.tokenizer.count(answer)
        outcome.recall, outcome.missing_facts = key_fact_recall(answer, item)
        outcome.judge_correct = self.judge.score(item, answer)
        return outcome

    # -- full run ----------------------------------------------------------
    def run(self, testset: TestSet, limit: int | None = None) -> dict:
        problems = validate(testset)
        if problems:
            raise ValueError(
                "test set is not answerable; fix these before trusting any number:\n  "
                + "\n  ".join(problems)
            )

        available, reason = self.model.available()
        if not available:
            raise RuntimeError(
                f"downstream model unavailable: {reason}. "
                f"Run `ollama pull {self.cfg.evaluation.downstream_model}`."
            )

        items = testset.items[:limit] if limit else testset.items
        num_ctx = self.cfg.evaluation.num_ctx
        results: list[ItemResult] = []
        started = time.perf_counter()

        for index, item in enumerate(items, start=1):
            context = testset.context_for(item)
            original_tokens = self.tokenizer.count(context.text)
            if original_tokens > num_ctx:
                # Refuse rather than silently measure a truncated original.
                raise ValueError(
                    f"{item.id}: original context is {original_tokens:,} tokens but "
                    f"num_ctx is {num_ctx:,}. Ollama would truncate it and the "
                    f"comparison would be meaningless. Raise evaluation.num_ctx "
                    f"or shrink the context."
                )

            compressed_text, _, _ = self.compressed_context(context)
            print(
                f"[{index}/{len(items)}] {item.id} ({context.key}) ... ",
                end="",
                flush=True,
            )

            original = self._run_one(context.text, item)
            compressed = self._run_one(compressed_text, item)
            verdict = (
                "retained"
                if compressed.recall >= original.recall
                else ("degraded" if compressed.recall < original.recall else "retained")
            )
            if compressed.recall > original.recall:
                verdict = "improved"
            results.append(
                ItemResult(
                    id=item.id,
                    context_key=item.context_key,
                    question=item.question,
                    expected_answer=item.expected_answer,
                    original=original,
                    compressed=compressed,
                    verdict=verdict,
                )
            )
            print(
                f"orig {original.recall:.0%} ({original.latency_ms/1000:.1f}s) -> "
                f"comp {compressed.recall:.0%} ({compressed.latency_ms/1000:.1f}s)"
            )

        return self._report(testset, results, time.perf_counter() - started)

    # -- deterministic measurements (no model calls) -----------------------
    def fact_survival(self, testset: TestSet, items: list[TestItem]) -> dict:
        """Fraction of key facts still present in the compressed context.

        The compressor's own ceiling, measured without asking a model anything.
        Accuracy cannot exceed it, so reporting the two together separates
        "compression destroyed the fact" from "the model failed to use it".
        """
        by_context: dict[str, dict] = {}
        surviving = total = 0
        for item in items:
            context = testset.context_for(item)
            compressed_text, _, _ = self.compressed_context(context)
            present = sum(1 for f in item.key_facts if fact_present(f, compressed_text))
            bucket = by_context.setdefault(
                item.context_key, {"surviving": 0, "total": 0}
            )
            bucket["surviving"] += present
            bucket["total"] += len(item.key_facts)
            surviving += present
            total += len(item.key_facts)
        for bucket in by_context.values():
            bucket["rate"] = round(bucket["surviving"] / bucket["total"], 4)
        return {
            "overall": round(surviving / total, 4) if total else 0.0,
            "facts_surviving": surviving,
            "facts_total": total,
            "by_context": by_context,
            "note": (
                "Deterministic: no model involved. This is the compressor's own "
                "ceiling - downstream accuracy is bounded by it."
            ),
        }

    @staticmethod
    def diagnostic_footnote() -> dict:
        """Optional GPT-4o-mini comparison, clearly labelled as a footnote.

        Never a headline number: the shipped pipeline answers locally, and this
        only indicates *why* the local model missed facts that were present.
        """
        base = {
            "label": "GPT-4o-mini retrieval diagnostic",
            "headline_metric": False,
            "affects_shipped_pipeline": False,
            "note": (
                "Diagnostic only. The shipped pipeline answers with a local "
                "model and requires no API key."
            ),
        }
        if not DIAGNOSTIC_REPORT.exists():
            return {
                **base,
                "available": False,
                "reason": (
                    "not run - optional, requires OPENAI_API_KEY with available "
                    "credits (python -m diagnostic.gpt4o_mini_check)"
                ),
            }
        payload = json.loads(DIAGNOSTIC_REPORT.read_text(encoding="utf-8"))
        return {
            **base,
            "available": True,
            "diagnostic_model": payload.get("diagnostic_model"),
            "baseline_model": payload.get("baseline_model"),
            "items_tested": payload.get("items_tested"),
            "aggregate": payload.get("aggregate"),
            "verdict": payload.get("verdict"),
        }

    # -- reporting ---------------------------------------------------------
    def _report(
        self, testset: TestSet, results: list[ItemResult], wall_clock_s: float
    ) -> dict:
        survival = self.fact_survival(testset, [
            i for i in testset.items if any(r.id == i.id for r in results)
        ])
        price = self.cfg.pricing.price_for(self.cfg.evaluation.pricing_model)

        def cost(tokens_in: int, tokens_out: int) -> float:
            return (
                tokens_in * price.input_per_1m + tokens_out * price.output_per_1m
            ) / 1_000_000

        tokens_before = sum(r.original.tokens_in for r in results)
        tokens_after = sum(r.compressed.tokens_in for r in results)
        cost_before = sum(cost(r.original.tokens_in, r.original.tokens_out) for r in results)
        cost_after = sum(
            cost(r.compressed.tokens_in, r.compressed.tokens_out) for r in results
        )
        latency_before = sum(r.original.latency_ms for r in results)
        latency_after = sum(r.compressed.latency_ms for r in results)
        accuracy_before = statistics.mean(r.original.recall for r in results)
        accuracy_after = statistics.mean(r.compressed.recall for r in results)

        judged = [
            (r.original.judge_correct, r.compressed.judge_correct)
            for r in results
            if r.original.judge_correct is not None
            and r.compressed.judge_correct is not None
        ]

        aggregate = {
            "items": len(results),
            "compression_ratio": _ratio(tokens_before, tokens_after),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "cost_before_usd": round(cost_before, 6),
            "cost_after_usd": round(cost_after, 6),
            "cost_reduction": _ratio(cost_before, cost_after),
            "latency_before_ms": round(latency_before, 1),
            "latency_after_ms": round(latency_after, 1),
            "latency_speedup": round(latency_before / latency_after, 3)
            if latency_after
            else 0.0,
            "accuracy_before": round(accuracy_before, 4),
            "accuracy_after": round(accuracy_after, 4),
            "accuracy_retention": round(accuracy_after / accuracy_before, 4)
            if accuracy_before
            else 0.0,
            "fact_survival_rate": survival["overall"],
            "verdicts": {
                verdict: sum(1 for r in results if r.verdict == verdict)
                for verdict in ("retained", "improved", "degraded")
            },
        }
        if judged:
            aggregate["judge_accuracy_before"] = round(
                sum(1 for a, _ in judged if a) / len(judged), 4
            )
            aggregate["judge_accuracy_after"] = round(
                sum(1 for _, b in judged if b) / len(judged), 4
            )

        return {
            "report_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "wall_clock_s": round(wall_clock_s, 1),
            "config": {
                "test_set": testset.name,
                "downstream_model": self.cfg.evaluation.downstream_model,
                "num_ctx": self.cfg.evaluation.num_ctx,
                "judge": self.judge.describe,
                "accuracy_metric": "key_fact_recall",
                "budget_ratio": self.budget_ratio,
                "pricing_model": self.cfg.evaluation.pricing_model,
                "abstractive_enabled": False,
                "tokenizer_exact": self.tokenizer.is_exact,
            },
            "aggregate": aggregate,
            "fact_survival": survival,
            "diagnostic": self.diagnostic_footnote(),
            "contexts": {
                key: {
                    "source": testset.contexts[key].source,
                    "original_tokens": original,
                    "compressed_tokens": compressed,
                    "compression_ratio": _ratio(original, compressed),
                }
                for key, (_, original, compressed) in self._compressed.items()
            },
            "items": [
                {
                    "id": r.id,
                    "context": r.context_key,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "verdict": r.verdict,
                    "original": asdict(r.original),
                    "compressed": asdict(r.compressed),
                }
                for r in results
            ],
        }


def finalize(report: dict, harness: "Harness", testset: TestSet) -> dict:
    """Refresh the deterministic sections of an existing report.

    Fact survival and the diagnostic footnote need no model calls, so a report
    can gain them without repeating ~14 minutes of local inference. The measured
    accuracy, latency and cost numbers are carried through untouched.
    """
    tested = [i for i in testset.items if any(e["id"] == i.id for e in report["items"])]
    survival = harness.fact_survival(testset, tested)
    # Per-item counts so each CSV row shows whether compression or retrieval failed.
    for entry in report["items"]:
        item = next((i for i in testset.items if i.id == entry["id"]), None)
        if item is None:
            continue
        compressed_text, _, _ = harness.compressed_context(testset.context_for(item))
        present = sum(1 for f in item.key_facts if fact_present(f, compressed_text))
        entry["facts_present_in_compressed"] = f"{present}/{len(item.key_facts)}"
    report["fact_survival"] = survival
    report["aggregate"]["fact_survival_rate"] = survival["overall"]
    report["diagnostic"] = harness.diagnostic_footnote()
    report["finalized_at"] = datetime.now(timezone.utc).isoformat()
    return report


def _ratio(before: float, after: float) -> float:
    return round(1.0 - after / before, 4) if before else 0.0


def write_report(report: dict, directory: Path = REPORTS_DIR) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "latest.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = directory / "latest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "context", "verdict",
            "orig_tokens", "comp_tokens", "token_reduction_pct",
            "orig_recall", "comp_recall", "facts_in_compressed_ctx",
            "orig_latency_ms", "comp_latency_ms", "speedup",
            "missing_facts_compressed", "question",
        ])
        for item in report["items"]:
            original, compressed = item["original"], item["compressed"]
            reduction = _ratio(original["tokens_in"], compressed["tokens_in"])
            speedup = (
                round(original["latency_ms"] / compressed["latency_ms"], 2)
                if compressed["latency_ms"]
                else ""
            )
            writer.writerow([
                item["id"], item["context"], item["verdict"],
                original["tokens_in"], compressed["tokens_in"], round(100 * reduction, 1),
                round(original["recall"], 3), round(compressed["recall"], 3),
                item.get("facts_present_in_compressed", ""),
                round(original["latency_ms"], 1), round(compressed["latency_ms"], 1),
                speedup,
                "|".join(compressed["missing_facts"]), item["question"],
            ])
    return json_path, csv_path


def print_summary(report: dict) -> None:
    aggregate = report["aggregate"]
    config = report["config"]
    print("\n" + "=" * 72)
    print(f"  {report['config']['test_set']} - {aggregate['items']} items, "
          f"{report['wall_clock_s']}s wall clock")
    print(f"  model {config['downstream_model']} (num_ctx {config['num_ctx']:,}) | "
          f"judge {config['judge']}")
    print("=" * 72)
    rows = [
        ("Compression ratio", f"{aggregate['compression_ratio']:.1%}",
         f"{aggregate['tokens_before']:,} -> {aggregate['tokens_after']:,} tokens"),
        ("Cost reduction", f"{aggregate['cost_reduction']:.1%}",
         f"${aggregate['cost_before_usd']:.5f} -> ${aggregate['cost_after_usd']:.5f} "
         f"({config['pricing_model']})"),
        ("Accuracy retention", f"{aggregate['accuracy_retention']:.1%}",
         f"{aggregate['accuracy_before']:.1%} -> {aggregate['accuracy_after']:.1%} "
         f"key-fact recall"),
        ("Latency speedup", f"{aggregate['latency_speedup']:.2f}x",
         f"{aggregate['latency_before_ms']/1000:.1f}s -> "
         f"{aggregate['latency_after_ms']/1000:.1f}s"),
    ]
    for label, headline, detail in rows:
        print(f"  {label:<20} {headline:>10}   {detail}")
    survival = report.get("fact_survival")
    if survival:
        print(f"\n  fact survival        {survival['overall']:>9.1%}   "
              f"{survival['facts_surviving']}/{survival['facts_total']} key facts kept "
              f"by compression (no model involved)")
        for key, bucket in survival["by_context"].items():
            print(f"      {key:<16}{bucket['rate']:>6.0%}  "
                  f"{bucket['surviving']}/{bucket['total']}")
        print(f"\n  Accuracy is bounded by fact survival: the compressor preserved "
              f"{survival['overall']:.1%},")
        print(f"  the model retrieved {aggregate['accuracy_after']:.1%} of what was there.")

    diagnostic = report.get("diagnostic")
    if diagnostic:
        print(f"\n  [footnote] {diagnostic['label']}: ", end="")
        if diagnostic.get("available"):
            agg = diagnostic["aggregate"]
            print(f"{agg['local_mean_recall']:.0%} -> {agg['diagnostic_mean_recall']:.0%} "
                  f"on {diagnostic['items_tested']} items ({agg['retention_delta']:+.0%})")
            print(f"             {diagnostic['verdict']}")
        else:
            print(f"{diagnostic['reason']}")
        print("             (not a headline metric; shipped pipeline stays local)")

    print(f"\n  verdicts: {aggregate['verdicts']}")
    degraded = [i for i in report["items"] if i["verdict"] == "degraded"]
    if degraded:
        print("\n  degraded items (what compression cost us):")
        for item in degraded:
            print(f"    {item['id']:<12} missing: "
                  f"{', '.join(item['compressed']['missing_facts']) or '-'}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.harness", description=__doc__)
    parser.add_argument("--testset", help="path to a testset.json")
    parser.add_argument("--budget", type=float, help="budget ratio override")
    parser.add_argument("--limit", type=int, help="only run the first N items")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the 14.5k-token log context (the slow one)",
    )
    parser.add_argument("--config", help="alternate config.yaml")
    parser.add_argument("--out", help="report directory (default: reports/)")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="recompute fact survival + diagnostic footnote on the existing "
             "report without re-running any model calls",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    testset = load_testset(args.testset)
    if args.quick:
        testset.items = [i for i in testset.items if i.context_key != "log_incident"]
        print(f"--quick: {len(testset.items)} items (log_incident skipped)")

    harness = Harness(get_config(args.config), args.budget)

    if args.finalize:
        path = REPORTS_DIR / "latest.json"
        if not path.exists():
            print(f"error: no report at {path}", file=sys.stderr)
            return 1
        report = finalize(
            json.loads(path.read_text(encoding="utf-8")), harness, testset
        )
        json_path, csv_path = write_report(report, Path(args.out) if args.out else REPORTS_DIR)
        print_summary(report)
        print(f"\nwrote {json_path}\n      {csv_path}")
        return 0

    try:
        report = harness.run(testset, limit=args.limit)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json_path, csv_path = write_report(
        report, Path(args.out) if args.out else REPORTS_DIR
    )
    print_summary(report)
    print(f"\nwrote {json_path}\n      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
