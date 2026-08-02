"""Stage 8 - the evaluation harness.

Runs every test item twice against the same downstream model: once with the
full context, once with the compressed context. Everything the dashboard shows
for accuracy, latency and cost originates here, measured.

    python -m eval.harness                    # full run, writes reports/latest.{json,csv}
    python -m eval.harness --quick            # skip the expensive 14.5k-token context
    python -m eval.harness --budget 0.15      # sweep a different budget

Method notes, because these are what a judge should be able to interrogate:

* **The model that answers is the one the provider chain resolves to.** The
  harness shares :class:`~engine.providers.chain.GenerationChain` with stage 6,
  and every report records which entry actually served the run - not which one
  config.yaml lists first. Both conditions go through the same object, so a
  mid-run fallback applies equally to the compressed and uncompressed arms.
* **The uncompressed context must fit the model.** Ollama silently truncates
  past ``num_ctx``; a truncated "original" would flatter the compressed run.
  ``num_ctx`` is set explicitly and the harness refuses to run such an item.
  The guard applies to the local provider only - hosted models have much larger
  windows and reject an oversized prompt rather than quietly shortening it.
* **Accuracy is deterministic key-fact recall**, not a model grading itself.
  See :mod:`eval.scoring`.
* **Cost is computed, not asserted** - published per-1M-token prices from
  ``config.yaml`` applied to measured token counts.
* **Latency is wall-clock, and now includes the network.** It is a property of
  the provider that served the run and the connection it ran over, so it is the
  least transferable number here. Compression ratio and token counts are not:
  they are properties of the compressor alone and do not move when the provider
  does.
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
from engine.providers import (
    GenerationChain,
    NoProviderAvailable,
    build_generation_chain,
    normalise_mode,
)
from engine.tokenizer import get_tokenizer

from .scoring import Judge, key_fact_recall
from .providers import available_providers
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
BASELINE_REPORT = REPORTS_DIR / "baseline_comparison.json"

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. Be specific and "
    "concise: state the exact values, names and identifiers the context gives. "
    "If the context does not contain the answer, reply exactly: NOT FOUND."
)

#: LLMLingua-2's smallest published checkpoint. The paper's LLMLingua-1 default
#: is llama-7b, which is not a fair ask of a laptop; -2 is also the newer and
#: stronger method, so comparing against it is the harder test of the two.
LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_LLMLINGUA: object | None = None
_LLMLINGUA_TRIED = False


def _load_llmlingua():
    """Load LLMLingua-2 once, or return None if it cannot be had.

    Imported from a neutral working directory on purpose. nltk (an llmlingua
    dependency) refuses to import any module that resolves inside the current
    directory, and this project's virtualenv lives *inside* the project - so
    every site-package looks like a CWD import to that check and the import
    dies. Restoring the cwd afterwards keeps the rest of the harness, which
    reads relative paths, unaffected.
    """
    global _LLMLINGUA, _LLMLINGUA_TRIED
    if _LLMLINGUA_TRIED:
        return _LLMLINGUA
    _LLMLINGUA_TRIED = True

    import os
    import tempfile

    previous = os.getcwd()
    try:
        os.chdir(tempfile.gettempdir())
        from llmlingua import PromptCompressor

        _LLMLINGUA = PromptCompressor(
            model_name=LLMLINGUA_MODEL, use_llmlingua2=True, device_map="cpu"
        )
    except Exception as exc:  # noqa: BLE001 - an optional baseline, never fatal
        log.warning("llmlingua unavailable (%s); its rows will be skipped", exc)
        _LLMLINGUA = None
    finally:
        os.chdir(previous)
    return _LLMLINGUA


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
    """The model both conditions are measured against. Identical settings.

    Since the provider migration this is the *same* ``GenerationChain`` stage 6
    uses - same ordering, same fallback, same timeouts - asked a different
    question with a different system prompt. Sharing it is the point: there is
    now one answer to "which model produced these numbers", and the report
    records whichever chain entry actually served the run rather than whichever
    one was configured first.

    The comparison stays fair because both conditions go through this one
    object: if the chain falls back mid-run, it falls back for the compressed
    and uncompressed arms alike.
    """

    def __init__(
        self,
        cfg: Config,
        chain: GenerationChain | None = None,
        mode: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.settings = cfg.evaluation
        self.mode = normalise_mode(mode)
        self.chain = chain or build_generation_chain(
            cfg,
            timeout_s=self.settings.timeout_s,
            num_ctx=self.settings.num_ctx,
            mode=self.mode,
        )

    def available(self) -> tuple[bool, str]:
        return self.chain.available()

    @property
    def label(self) -> str:
        """``provider:model`` of whoever last answered - for the report header."""
        name = self.chain.last_provider or self.chain.active_provider_name()
        provider = next((p for p in self.chain.providers if p.name == name), None)
        return f"{name}:{provider.model}" if provider else "none"

    @property
    def is_local(self) -> bool:
        """Whether the live provider is the local Ollama one.

        Decides whether the ``num_ctx`` truncation guard applies: Ollama
        silently truncates past its window, hosted models have their own much
        larger ones and reject rather than truncate.
        """
        return (self.chain.active_provider_name() or "") == "local"

    def ask(self, context: str, question: str) -> tuple[str, float, int]:
        """Returns (answer, latency_ms, prompt_tokens_reported_by_model)."""
        prompt = f"{context}\n\nQuestion: {question}"
        started = time.perf_counter()
        answer = self.chain.generate(
            prompt,
            max_tokens=self.settings.max_answer_tokens,
            timeout_s=self.settings.timeout_s,
            system=SYSTEM_PROMPT,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        # Only the local provider reports its own prompt token count; the
        # harness counts tokens itself anyway, so 0 is a fine "not reported".
        active = next(
            (p for p in self.chain.providers if p.name == self.chain.last_provider),
            None,
        )
        reported = int(getattr(active, "last_prompt_tokens", 0) or 0)
        return answer.strip(), latency_ms, reported


class Harness:
    def __init__(
        self,
        cfg: Config | None = None,
        budget_ratio: float | None = None,
        mode: str | None = None,
        query_aware: bool = True,
    ):
        self.cfg = cfg or get_config()
        #: When False the question is withheld from the compressor, which is
        #: the pre-query-aware behaviour and the control condition for
        #: measuring what the signal is worth.
        self.query_aware = query_aware
        self.budget_ratio = (
            budget_ratio if budget_ratio is not None else self.cfg.selection.budget_ratio
        )
        # One mode for the whole run: the compressor's embeddings and the
        # answering model must come from the same side of the local/cloud line,
        # or the latency and cost figures describe a hybrid nobody chose.
        self.mode = normalise_mode(mode)
        self.tokenizer = get_tokenizer(self.cfg.tokenizer)
        self.model = DownstreamModel(self.cfg, mode=self.mode)
        self.judge = Judge(self.cfg.evaluation, self.cfg.providers)
        # Stage 6 off: it adds minutes of LLM calls to every compression for no
        # measured token saving on this corpus (see README).
        self.pipeline = CompressionPipeline(
            self.cfg.with_overrides({"abstractive": {"enabled": False}}),
            mode=self.mode,
        )
        self._compressed: dict[tuple[str, str], tuple[str, int, int]] = {}

    # -- compression cache -------------------------------------------------
    def compressed_context(
        self, context: EvalContext, question: str | None = None
    ) -> tuple[str, int, int]:
        """Compress a context, cached.

        The cache key includes the question, because with query-aware scoring
        the compression genuinely differs per question - that is the whole
        point of the signal. Keying on the context alone would have quietly
        served one question's compression to all of them and reported a number
        that no configuration produces.

        With ``query_aware=False`` the question is not passed down, the key
        collapses back to the context, and this compresses once per context as
        it did before.
        """
        query = (question or "").strip() if self.query_aware else ""
        key = (context.key, query)
        if key not in self._compressed:
            result = self.pipeline.compress(
                context.text,
                context.name,
                budget_ratio=self.budget_ratio,
                query=query or None,
            )
            self._compressed[key] = (
                result.compressed_text,
                result.original_tokens,
                result.compressed_tokens,
            )
        return self._compressed[key]

    # -- one condition -----------------------------------------------------
    def _run_one(self, context: str, item: TestItem) -> RunOutcome:
        outcome = RunOutcome(tokens_in=self.tokenizer.count(context))
        try:
            answer, latency_ms, _ = self.model.ask(context, item.question)
        except NoProviderAvailable:
            # The whole chain went down mid-run. Every remaining item would
            # record a 0% recall that reads as "compression destroyed the
            # facts" rather than "nothing answered", so stop loudly instead of
            # finishing a run whose numbers mean nothing.
            raise
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
                f"no generation provider available: {reason}. Configure a key in "
                f".env (see .env.example) or start a local Ollama, then re-run. "
                f"Chain as configured: "
                f"{' -> '.join(self.cfg.providers.generation_providers)}."
            )

        items = testset.items[:limit] if limit else testset.items
        num_ctx = self.cfg.evaluation.num_ctx
        # The truncation guard exists because Ollama silently truncates past
        # num_ctx, which would make every "original context" measurement
        # fiction. Hosted models have their own, much larger windows and reject
        # an oversized prompt rather than quietly shortening it - so the guard
        # applies to the local provider only, and applying it regardless would
        # refuse runs that are perfectly valid on Groq or Gemini.
        guard_truncation = self.model.is_local
        results: list[ItemResult] = []
        started = time.perf_counter()

        for index, item in enumerate(items, start=1):
            context = testset.context_for(item)
            original_tokens = self.tokenizer.count(context.text)
            if guard_truncation and original_tokens > num_ctx:
                # Refuse rather than silently measure a truncated original.
                raise ValueError(
                    f"{item.id}: original context is {original_tokens:,} tokens but "
                    f"num_ctx is {num_ctx:,}. Ollama would truncate it and the "
                    f"comparison would be meaningless. Raise evaluation.num_ctx "
                    f"or shrink the context."
                )

            compressed_text, _, _ = self.compressed_context(context, item.question)
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
            compressed_text, _, _ = self.compressed_context(context, item.question)
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

    def compare_baselines(self, testset: TestSet, budgets: list[float],
                          limit: int | None = None) -> dict:
        """Measured, model-free selection comparison for the deck/demo.

        This intentionally calls the real chunking, redundancy and density
        stages for every strategy. Only stage 5 is swapped.
        """
        items = testset.items[:limit] if limit else testset.items
        rows: list[dict] = []
        labels = {"truncate": "naive_truncation", "random": "random_sampling",
                  "density": "density_based"}
        for budget in budgets:
            for strategy, label in labels.items():
                pipe = CompressionPipeline(self.cfg.with_overrides({"abstractive": {"enabled": False}}))
                facts = total = 0
                by_context: dict[str, list[int]] = {}
                for item in items:
                    context = testset.context_for(item)
                    result = pipe.compress(context.text, context.name, budget_ratio=budget,
                                           fast_mode=True, selection_strategy=strategy)
                    present = sum(fact_present(f, result.compressed_text) for f in item.key_facts)
                    facts += present
                    total += len(item.key_facts)
                    bucket = by_context.setdefault(context.key, [0, 0])
                    bucket[0] += present
                    bucket[1] += len(item.key_facts)
                rows.append({"strategy": label, "budget_ratio": budget,
                             "fact_survival_rate": round(facts / total, 4) if total else 0.0,
                             "facts_surviving": facts, "facts_total": total,
                             "by_input_type": {
                                 key: round(v[0] / v[1], 4) if v[1] else 0.0
                                 for key, v in by_context.items()
                             },
                             "accuracy_retention": None,
                             "accuracy_note": "not run; baseline mode is deterministic by default"})
        return {"generated_at": datetime.now(timezone.utc).isoformat(),
                "metric": "deterministic key-fact survival", "seed": 1337,
                "rows": rows,
                "note": "Accuracy retention is omitted unless model calls are explicitly added; fact survival isolates selection from retrieval noise."}

    def compare_llmlingua(
        self, testset: TestSet, budgets: list[float], limit: int | None = None
    ) -> dict:
        """Measure LLMLingua-2 on the same corpus, with the same fact metric.

        Truncation and random sampling are strawmen - nobody ships them.
        LLMLingua is the actual prior art in prompt compression, so "how does
        this compare to LLMLingua?" is a question worth answering with numbers
        rather than for the first time on stage.

        Same key-fact survival check, same budgets, same inputs. The only thing
        that changes is the compressor.
        """
        compressor = _load_llmlingua()
        if compressor is None:
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "available": False,
                "reason": (
                    "llmlingua is not installed (pip install llmlingua), or its "
                    "model could not be loaded"
                ),
                "rows": [],
            }

        items = testset.items[:limit] if limit else testset.items
        rows: list[dict] = []
        for budget in budgets:
            facts = total = 0
            tokens_before = tokens_after = 0
            elapsed = 0.0
            by_context: dict[str, list[int]] = {}
            cache: dict[str, str] = {}
            for item in items:
                context = testset.context_for(item)
                if context.key not in cache:
                    started = time.perf_counter()
                    out = compressor.compress_prompt(
                        context.text, rate=budget, force_tokens=["\n", ".", ",", "?"]
                    )
                    elapsed += time.perf_counter() - started
                    cache[context.key] = out["compressed_prompt"]
                    tokens_before += self.tokenizer.count(context.text)
                    tokens_after += self.tokenizer.count(cache[context.key])
                compressed = cache[context.key]
                present = sum(fact_present(f, compressed) for f in item.key_facts)
                facts += present
                total += len(item.key_facts)
                bucket = by_context.setdefault(context.key, [0, 0])
                bucket[0] += present
                bucket[1] += len(item.key_facts)
            rows.append({
                "strategy": "llmlingua2",
                "budget_ratio": budget,
                "fact_survival_rate": round(facts / total, 4) if total else 0.0,
                "facts_surviving": facts,
                "facts_total": total,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "compression_ratio": _ratio(tokens_before, tokens_after),
                "wall_clock_s": round(elapsed, 1),
                "by_input_type": {
                    key: round(v[0] / v[1], 4) if v[1] else 0.0
                    for key, v in by_context.items()
                },
            })
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "available": True,
            "model": LLMLINGUA_MODEL,
            "metric": "deterministic key-fact survival (identical to our rows)",
            "rows": rows,
            "note": (
                "LLMLingua-2 is a token-level classifier: it drops individual "
                "tokens, so its output is not valid code, log or prose and it "
                "cannot report what it removed. Both properties matter for the "
                "fact metric and for auditability."
            ),
        }

    def compare_providers(
        self, testset: TestSet, names: list[str], limit: int | None = None
    ) -> dict:
        """Cross-provider accuracy report. Each provider runs alone, unchained.

        No fallback here on purpose: the point of the report is to attribute a
        retention number to a specific model, and a chain that silently
        substituted a different one would make two rows secretly identical. A
        provider with no key is reported as skipped, which is information.
        """
        items = testset.items[:limit] if limit else testset.items
        registry = available_providers(self.cfg)
        rows, skipped = [], []
        for name in names:
            provider = registry.get(name)
            if not provider:
                skipped.append({"provider": name, "reason": "unknown provider"})
                continue
            ok, reason = provider.configured()
            if not ok:
                log.warning("%s: %s", name, reason)
                skipped.append({"provider": name, "reason": reason})
                continue
            before = after = 0.0
            errors: list[str] = []
            for item in items:
                context = testset.context_for(item)
                compressed, _, _ = self.compressed_context(context, item.question)
                try:
                    before += key_fact_recall(
                        provider.generate(
                            f"{context.text}\n\nQuestion: {item.question}",
                            max_tokens=self.cfg.evaluation.max_answer_tokens,
                            system=SYSTEM_PROMPT,
                        ),
                        item,
                    )[0]
                    after += key_fact_recall(
                        provider.generate(
                            f"{compressed}\n\nQuestion: {item.question}",
                            max_tokens=self.cfg.evaluation.max_answer_tokens,
                            system=SYSTEM_PROMPT,
                        ),
                        item,
                    )[0]
                except Exception as exc:
                    errors.append(f"{item.id}: {exc}")
                    # A rate-limited or stalled provider has already consumed
                    # its bounded retry. Stop it rather than multiplying the
                    # timeout across the remainder of the test set.
                    break
            answered = len(items) - len(errors)
            rows.append({
                "provider": name,
                "model": provider.model,
                "items_answered": answered,
                "accuracy_before": round(before / answered, 4) if answered else 0,
                "accuracy_after": round(after / answered, 4) if answered else 0,
                "accuracy_retention": round(after / before, 4) if before else 0,
                "errors": errors,
            })
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
            "skipped": skipped,
            "note": (
                "Each provider answers alone, with no fallback, so a retention "
                "number belongs to exactly the model named in its row."
            ),
        }

    @staticmethod
    def diagnostic_footnote() -> dict:
        """Optional GPT-4o-mini comparison, clearly labelled as a footnote.

        Never a headline number. It isolates *retrieval* from *compression*: if
        a stronger model finds a fact the run's provider missed, the fact
        survived compression and the answer model was the limit.
        """
        base = {
            "label": "GPT-4o-mini retrieval diagnostic",
            "headline_metric": False,
            "affects_shipped_pipeline": False,
            "note": (
                "Diagnostic only, and independent of the provider chain that "
                "served the run - it exists to separate 'compression lost the "
                "fact' from 'the answering model missed it'."
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

    def _context_summary(self, testset: TestSet) -> dict:
        """Per-context token figures, averaged over that context's questions.

        With query-aware scoring one context has as many compressions as it has
        questions, and they legitimately differ. Averaging is the honest
        summary; the per-item rows below carry the exact numbers.
        """
        grouped: dict[str, list[tuple[int, int]]] = {}
        for (context_key, _query), (_text, original, compressed) in self._compressed.items():
            grouped.setdefault(context_key, []).append((original, compressed))

        summary: dict[str, dict] = {}
        for context_key, pairs in grouped.items():
            original = round(statistics.mean(p[0] for p in pairs))
            compressed = round(statistics.mean(p[1] for p in pairs))
            entry = {
                "source": testset.contexts[context_key].source,
                "original_tokens": original,
                "compressed_tokens": compressed,
                "compression_ratio": _ratio(original, compressed),
            }
            if len(pairs) > 1:
                entry["compressions"] = len(pairs)
                entry["note"] = "mean over this context's per-question compressions"
            summary[context_key] = entry
        return summary

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
                # Whichever chain entry actually answered, not whichever was
                # configured first. With a fallback chain those differ, and a
                # report that names the wrong model is a report that lies.
                "downstream_model": self.model.label,
                "mode": self.mode,
                "generation_chain": self.cfg.providers.generation_providers,
                "generation_provider": self.model.chain.last_provider,
                "generation_attempts": [
                    a.to_dict() for a in self.model.chain.last_attempts
                ],
                "embedding_chain": [self.cfg.providers.embedding_provider]
                + list(self.cfg.providers.embedding_fallback),
                "embedding_provider": self.pipeline.embedder.stats.provider,
                "num_ctx": self.cfg.evaluation.num_ctx,
                "judge": self.judge.describe,
                "accuracy_metric": "key_fact_recall",
                "budget_ratio": self.budget_ratio,
                "pricing_model": self.cfg.evaluation.pricing_model,
                "query_aware": self.query_aware,
                "abstractive_enabled": False,
                "tokenizer_exact": self.tokenizer.is_exact,
            },
            "aggregate": aggregate,
            "fact_survival": survival,
            "diagnostic": self.diagnostic_footnote(),
            # The cache is keyed (context, question) since query-aware scoring
            # makes the compression question-dependent. Report per context,
            # averaging the per-question compressions of each one.
            "contexts": self._context_summary(testset),
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
        compressed_text, _, _ = harness.compressed_context(
            testset.context_for(item), item.question
        )
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
    parser.add_argument(
        "--no-query", action="store_true",
        help="withhold the question from the compressor (query-blind control "
             "condition, i.e. the behaviour before query-aware scoring)",
    )
    parser.add_argument("--compare-baselines", action="store_true",
                        help="write deterministic density vs truncation/random report")
    parser.add_argument("--with-llmlingua", action="store_true",
                        help="also measure LLMLingua-2 on the same corpus and "
                             "metric (needs `pip install llmlingua`; downloads "
                             "a ~700 MB model on first use)")
    parser.add_argument("--providers", help="comma-separated evaluation providers: local,groq,gemini,openrouter")
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

    harness = Harness(
        get_config(args.config), args.budget, query_aware=not args.no_query
    )

    if args.compare_baselines:
        report = harness.compare_baselines(testset, [0.15, 0.30, 0.50], args.limit)
        if args.with_llmlingua:
            report["llmlingua"] = harness.compare_llmlingua(
                testset, [0.15, 0.30, 0.50], args.limit
            )
        output = Path(args.out) if args.out else REPORTS_DIR
        output.mkdir(parents=True, exist_ok=True)
        path = output / BASELINE_REPORT.name
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["rows"], indent=2))
        print(f"wrote {path}")
        return 0

    if args.providers:
        report = harness.compare_providers(testset, [p.strip() for p in args.providers.split(",") if p.strip()], args.limit)
        output = Path(args.out) if args.out else REPORTS_DIR
        output.mkdir(parents=True, exist_ok=True)
        path = output / "provider_comparison.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)); print(f"wrote {path}")
        return 0

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
