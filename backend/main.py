"""Stage 9 - FastAPI backend.

    uvicorn backend.main:app --reload --port 8000

Serves the contract in ``docs/API_CONTRACT.md``. Three things this layer is
responsible for beyond routing:

* **Warming the models at boot.** Cold, the first request pays ~9 s loading
  MiniLM and the MPS backend; warm it is ~250 ms. Without startup warmup the
  first judge to click Compress sees what looks like a hang.
* **Never echoing the input back.** ``/compress`` returns character spans over
  the text the client already holds, not the text itself - 2.17 MB versus 70 KB
  on the largest sample input.
* **Degrading rather than failing.** A stage that cannot run reports
  ``status: "skipped"`` with a reason inside a 200 response. Only a malformed
  request or a genuinely broken pipeline produces an error envelope.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engine import __version__
from engine.config import PROJECT_ROOT, get_config
from engine.ingestion import (
    MAX_FILES,
    MAX_TOTAL_BYTES,
    SEPARATOR,
    combine,
    combined_name,
    extract_file,
)
from engine.pipeline import CompressionPipeline
from engine.providers import (
    DEFAULT_MODE,
    MODES,
    InvalidMode,
    InvalidProvider,
    NoProviderAvailable,
    mode_readiness,
    normalise_mode,
    provider_catalogue,
    provider_status,
    selection_presets,
)
from eval.harness import REPORTS_DIR, Harness, write_report
from eval.testset import load_testset

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Process-wide state
# --------------------------------------------------------------------------
STATE: dict[str, Any] = {"pipelines": {}, "warm": False, "warmup_ms": {}}
JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_PIPELINES_LOCK = threading.Lock()

#: Recoverable marker content, keyed by compression id, for `/expand`.
#:
#: Deliberately bounded and in-memory: this is an escape hatch for the run you
#: are looking at, not a document store. Holding every compression forever
#: would turn a stateless service into a memory leak with a 2 MB unit, and the
#: text is reproducible by re-compressing.
EXPANSIONS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
MAX_EXPANSIONS = 8
_EXPAND_LOCK = threading.Lock()


def _remember_expansions(result: Any) -> str:
    """Store what each marker hides and return the id that addresses it."""
    compression_id = uuid.uuid4().hex[:12]
    payload: dict[str, Any] = {}
    by_id = {c.id: c for c in result.chunks}
    for marker in result.markers:
        chunks = [by_id[cid] for cid in marker["chunk_ids"] if cid in by_id]
        payload[marker["id"]] = {
            "kind": marker["kind"],
            "sections": marker["sections"],
            "tokens": marker["tokens"],
            "text": "\n\n".join(c.text for c in chunks),
            "chunks": [
                {
                    "id": c.id,
                    "kind": c.kind,
                    "symbol": c.symbol,
                    "tokens": c.token_count,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                }
                for c in chunks
            ],
        }
    with _EXPAND_LOCK:
        EXPANSIONS[compression_id] = {"markers": payload, "at": time.time()}
        while len(EXPANSIONS) > MAX_EXPANSIONS:
            EXPANSIONS.popitem(last=False)
    return compression_id

#: Shared by both arms of /answer so the comparison is not confounded by
#: different instructions. Identical to the eval harness's prompt.
#: Vendor phrasings for "your prompt exceeds what this model accepts".
_TOO_LARGE = (
    "413", "request too large", "too large", "context length",
    "maximum context", "exceeds", "reduce the length", "too many tokens",
)


def _is_context_too_large(error: str) -> bool:
    """Whether an arm failed because the prompt did not fit, not because the
    provider was down. The distinction is the whole demo on a large input."""
    lowered = error.lower()
    return any(marker in lowered for marker in _TOO_LARGE)


ANSWER_SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. Be specific and "
    "concise: state the exact values, names and identifiers the context gives. "
    "If the context does not contain the answer, reply exactly: NOT FOUND."
)

MAX_INPUT_CHARS = 2_000_000
SAMPLE_ROOT = PROJECT_ROOT / "data" / "sample_corpus"

#: Deployment default. A hosted instance has no Ollama and no Metal, and stage 6
#: measurably saves 0 tokens on this corpus while costing 3-6s per chunk - so the
#: judge-facing build skips it by default. This is a *default*, not a lock: a
#: request may still pass fast_mode=false to watch stage 6 run and get correctly
#: skipped or rejected live.
FORCE_FAST_MODE = os.environ.get("CCE_FORCE_FAST_MODE", "").lower() in {"1", "true", "yes"}


def pipeline(
    mode: str | None = None,
    embedding_provider: str | None = None,
    generation_provider: str | None = None,
) -> CompressionPipeline:
    """One cached pipeline per execution mode.

    A pipeline's provider chains are fixed at construction, so mode cannot be a
    per-call argument without mutating shared state - and these endpoints run on
    a threadpool, where two concurrent requests swapping chains on one object
    would each report the other's provider. Three small objects instead: the
    expensive parts (the MiniLM weights, the spaCy pipeline, the tokenizer) are
    cached process-wide and shared between them regardless.
    """
    key = (normalise_mode(mode), embedding_provider, generation_provider)
    with _PIPELINES_LOCK:
        existing = STATE["pipelines"].get(key)
        if existing is None:
            existing = CompressionPipeline(
                mode=key[0],
                embedding_provider=embedding_provider,
                generation_provider=generation_provider,
            )
            STATE["pipelines"][key] = existing
        return existing


@asynccontextmanager
async def lifespan(app: FastAPI):
    def warm() -> None:
        try:
            started = time.perf_counter()
            STATE["warmup_ms"] = pipeline().warmup()
            STATE["warm"] = True
            log.info("models warm in %.0f ms", (time.perf_counter() - started) * 1000)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("warmup failed (%s); first request will be slow", exc)

    # Off-thread so the port binds immediately and /health can report warm=false.
    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(
    title="Context Compression Engine",
    version=__version__,
    description=(
        "Algorithmic prompt compression: chunk, deduplicate, score, select, "
        "reconstruct. Every number returned is measured, not asserted."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # vite dev
        "http://localhost:4173", "http://localhost:3000",
    ],
    allow_origin_regex=r"https?://.*\.(vercel\.app|netlify\.app|onrender\.com)",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Errors - one envelope so the frontend can toast anything
# --------------------------------------------------------------------------
def error_response(code: str, message: str, http_status: int, **detail) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={"error": {"code": code, "message": message, "detail": detail}},
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("unhandled error on %s", request.url.path)
    return error_response("pipeline_error", str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class CompressRequest(BaseModel):
    text: str = Field(..., description="Raw context to compress")
    name: str = Field("input.txt", description="Filename; drives chunker detection")
    budget_ratio: float = Field(0.30, ge=0.05, le=1.0)
    kind: Literal["auto", "code", "text", "log"] = "auto"
    query: str | None = Field(None, description="Protected: never dropped")
    instruction: str | None = Field(None, description="Protected: never dropped")
    fast_mode: bool | None = Field(
        None,
        description="Skip stage 6. Defaults to the CCE_FORCE_FAST_MODE env var "
                    "(on for deployed instances); pass false to force it to run.",
    )
    include_original: bool = Field(False, description="Echo the input back (debug)")
    include_chunks: bool = Field(False, description="Per-chunk metadata (debug)")
    mode: Literal["local", "cloud", "auto"] = Field(
        DEFAULT_MODE,
        description=(
            "Which providers may serve this request. 'local' makes no outbound "
            "API call even with keys configured. 'cloud' uses the configured "
            "cloud chain only and errors rather than silently degrading to "
            "local. 'auto' (default) tries cloud then falls back to local."
        ),
    )
    embedding_provider: str | None = Field(
        None,
        description=(
            "Pin stage 3 to one provider (see GET /providers). Overrides `mode` "
            "for embeddings and gets NO fallback - a pinned provider that fails "
            "is an error, because silently substituting a different model would "
            "misattribute its results."
        ),
    )
    generation_provider: str | None = Field(
        None,
        description="Pin stage 6 to one provider. Same no-fallback rule as above.",
    )


class EvaluateRequest(BaseModel):
    test_set: str = "default"
    budget_ratio: float | None = Field(None, ge=0.05, le=1.0)
    run: bool = Field(False, description="Run fresh instead of serving the last report")
    mode: Literal["local", "cloud", "auto"] = Field(
        DEFAULT_MODE, description="Providers the answering step may use; see /compress."
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Readiness, including which model providers are actually live.

    The ``providers`` block is the honest answer to "what is running this right
    now": for each provider it reports whether a key is configured and which
    entry each chain would use, and it does so *without* ever returning a key
    or any fragment of one. Presence is a boolean here and nothing else.

    Cheap enough for the dashboard to poll: no hosted provider is called, only
    asked whether it is configured. The local providers are the exception and
    get a 3 s reachability probe, because that is the only way "is Ollama
    running" can be answered truthfully.
    """
    cfg = get_config()
    pipe = pipeline()
    generation_ok = pipe.abstractive.client.available()
    report_exists = (REPORTS_DIR / "latest.json").exists()
    providers = provider_status(cfg)
    modes = providers["modes"]
    return {
        "status": "ok",
        "version": __version__,
        "warm": STATE["warm"],
        "warmup_ms": {k: round(v) for k, v in STATE["warmup_ms"].items()},
        "tokenizer": pipe.tokenizer.describe(),
        "embeddings": pipe.embedder.describe(),
        "entities": pipe.density.entity_scorer.describe(),
        "providers": providers,
        # Independent per-mode readiness, so the dashboard can grey out a
        # toggle option instead of letting someone pick one that will fail.
        # These two are answered separately on purpose: Ollama being down must
        # not make cloud look broken, and having no keys must not make local
        # look broken.
        "modes": {
            "available": list(MODES),
            "default": DEFAULT_MODE,
            "local": modes["local"],
            "cloud": modes["cloud"],
        },
        "generation": {
            "available": generation_ok,
            "model": pipe.abstractive.client.model,
            "chain": cfg.providers.generation_providers,
            "error": pipe.abstractive.client.error,
        },
        "force_fast_mode": FORCE_FAST_MODE,
        "capabilities": {
            "abstractive": (
                generation_ok and cfg.abstractive.enabled and not FORCE_FAST_MODE
            ),
            "evaluate_cached": report_exists,
            "evaluate_live": generation_ok,
            "mode_local": modes["local"]["ready"],
            "mode_cloud": modes["cloud"]["ready"],
        },
    }


def _compress_request(request: CompressRequest, source_files: list[dict] | None = None,
                      file_status: list[dict] | None = None) -> Any:
    if not request.text.strip():
        return error_response(
            "invalid_request", "text is empty", status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    if len(request.text) > MAX_INPUT_CHARS:
        return error_response(
            "payload_too_large",
            f"input is {len(request.text):,} characters; the limit is "
            f"{MAX_INPUT_CHARS:,}",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            characters=len(request.text),
            limit=MAX_INPUT_CHARS,
        )

    try:
        engine = pipeline(
            request.mode, request.embedding_provider, request.generation_provider
        )
    except (InvalidMode, InvalidProvider) as exc:
        return error_response(
            "invalid_request", str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except NoProviderAvailable as exc:
        # mode='cloud' with nothing cloud-shaped available. Refuse loudly: a
        # user who asked for cloud and silently received a 3B local model would
        # read the latency and accuracy numbers as cloud's and be wrong.
        return error_response(
            "no_provider_available", str(exc), status.HTTP_503_SERVICE_UNAVAILABLE,
            mode=request.mode,
        )

    result = engine.compress(
        source=request.text,
        name=request.name,
        budget_ratio=request.budget_ratio,
        kind=request.kind,
        query=request.query,
        instruction=request.instruction,
        fast_mode=(
            FORCE_FAST_MODE if request.fast_mode is None else request.fast_mode
        ),
        source_files=source_files,
    )
    payload = result.to_dict(
        include_original=request.include_original,
        include_chunks=request.include_chunks,
    )
    # The recovered text itself is kept server-side rather than inlined: it is
    # by definition the bulk of what compression just removed, and returning it
    # would undo the payload saving the spans[] design exists for.
    payload["compression_id"] = _remember_expansions(result)
    if file_status is not None:
        payload["files"] = file_status
    return payload


@app.post("/compress", tags=["compression"])
async def compress(request: Request) -> Any:
    """Accept the established JSON contract or an additive multipart upload."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        try:
            body = await request.json()
            parsed = CompressRequest.model_validate(body)
        except Exception as exc:
            return error_response("invalid_request", str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY)
        return _compress_request(parsed)

    form = await request.form()
    uploads = [item for item in form.getlist("files") if hasattr(item, "read")]

    # Reject an oversized batch before parsing a single PDF: the count is known
    # from the form, so an obviously-too-large upload should cost milliseconds,
    # not two minutes of extraction followed by a 413.
    if len(uploads) > MAX_FILES:
        return error_response(
            "payload_too_large",
            f"{len(uploads)} files uploaded; at most {MAX_FILES} are allowed",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            files=len(uploads),
            limit=MAX_FILES,
        )

    total = 0
    extracted = []
    for upload in uploads:
        data = await upload.read()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            return error_response(
                "payload_too_large",
                f"uploads total more than {MAX_TOTAL_BYTES:,} bytes",
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                bytes=total,
                limit=MAX_TOTAL_BYTES,
            )
        # A file that cannot be parsed is skipped with a reason, never fatal.
        extracted.append(extract_file(upload.filename or "upload.txt", data))

    text, source_files, statuses = combine(extracted)

    # Pasted text is appended after the files, in the same provenance scheme,
    # so the two input modes compose instead of being mutually exclusive.
    pasted = str(form.get("text") or "").strip()
    if pasted:
        if text:
            text += SEPARATOR
        start = len(text)
        text += pasted
        source_files.append({"name": "pasted.txt", "start": start, "end": len(text)})
        statuses.append({
            "name": "pasted.txt", "status": "done", "reason": None,
            "characters": len(pasted),
        })

    if not text.strip():
        # Every file was skipped. That is a real answer, and the per-file
        # reasons are the useful part of it - so return them rather than a
        # bare "text is empty".
        return error_response(
            "no_readable_input",
            "none of the uploaded files produced any text",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            files=statuses,
        )

    fast_value = form.get("fast_mode")
    try:
        parsed = CompressRequest(
            text=text,
            name=combined_name(statuses),
            budget_ratio=float(form.get("budget_ratio") or 0.30),
            fast_mode=(
                str(fast_value).lower() in {"1", "true", "yes"}
                if fast_value is not None
                else None
            ),
            mode=str(form.get("mode") or DEFAULT_MODE).lower(),
            query=(str(form.get("query")).strip() or None) if form.get("query") else None,
            embedding_provider=(str(form.get("embedding_provider")).lower()
                                if form.get("embedding_provider") else None),
            generation_provider=(str(form.get("generation_provider")).lower()
                                 if form.get("generation_provider") else None),
        )
    except Exception as exc:
        return error_response(
            "invalid_request", str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    return _compress_request(parsed, source_files, statuses)


@app.post("/evaluate", tags=["evaluation"])
def evaluate(request: EvaluateRequest) -> Any:
    """Serve the last measured report, or start a fresh run.

    A live run is minutes of local inference (the uncompressed contexts are the
    slow half), so the default serves the report on disk instantly and the UI
    stays responsive. `run: true` returns a job id to poll.
    """
    report_path = REPORTS_DIR / "latest.json"

    if not request.run:
        if not report_path.exists():
            return error_response(
                "not_found",
                "no report on disk yet; run `python -m eval.harness` or POST "
                "with run=true",
                status.HTTP_404_NOT_FOUND,
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["cached"] = True
        return report

    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "started_at": time.time(), "report": None}

    def worker() -> None:
        try:
            testset = load_testset()
            harness = Harness(get_config(), request.budget_ratio, mode=request.mode)
            report = harness.run(testset)
            write_report(report)
            with _JOBS_LOCK:
                JOBS[job_id].update(status="complete", report=report)
        except Exception as exc:
            log.exception("evaluation job %s failed", job_id)
            with _JOBS_LOCK:
                JOBS[job_id].update(status="failed", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "status": "running", "poll": f"/evaluate/{job_id}"},
    )


@app.get("/evaluate/{job_id}", tags=["evaluation"])
def evaluate_status(job_id: str) -> Any:
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return error_response("not_found", f"no job {job_id}", status.HTTP_404_NOT_FOUND)
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_s": round(time.time() - job["started_at"], 1),
    }
    if job["status"] == "complete":
        payload["report"] = {**job["report"], "cached": False}
    elif job["status"] == "failed":
        payload["error"] = job.get("error")
    return payload


@app.get("/config", tags=["meta"])
def config() -> dict[str, Any]:
    """The tunables behind every number, for a technical walkthrough."""
    cfg = get_config()
    return {
        "selection": cfg.selection.model_dump(),
        "density_weights": cfg.density.weights.model_dump(),
        "redundancy": {
            "similarity_threshold": cfg.redundancy.similarity_threshold,
            "structural": cfg.redundancy.structural.model_dump(),
        },
        # The provider chains as configured. Model names only - the key
        # presence map lives on /health and the keys themselves nowhere.
        "providers": {
            "embedding_provider": cfg.providers.embedding_provider,
            "embedding_fallback": cfg.providers.embedding_fallback,
            "generation_providers": cfg.providers.generation_providers,
            "models": cfg.providers.models.model_dump(),
            "cooldown_s": cfg.providers.cooldown_s,
        },
        "abstractive": {
            "enabled": cfg.abstractive.enabled,
            "entity_overlap_threshold": cfg.abstractive.entity_overlap_threshold,
        },
        "evaluation": {
            "num_ctx": cfg.evaluation.num_ctx,
            "accuracy_metric": "key_fact_recall",
        },
    }


class AnswerRequest(BaseModel):
    """Ask one question twice: against the full context, and the compressed one."""

    text: str = Field(..., description="The full, uncompressed context")
    question: str = Field(..., min_length=1)
    name: str = "input.txt"
    mode: Literal["local", "cloud", "auto"] = DEFAULT_MODE
    embedding_provider: str | None = None
    generation_provider: str | None = None
    max_tokens: int = Field(160, ge=16, le=1024)


@app.post("/answer", tags=["evaluation"])
def answer(request: AnswerRequest) -> Any:
    """The payoff, measured live: does the compressed prompt still answer?

    Compression ratios are abstract. What a reader actually wants to know is
    whether the small prompt gets the same answer, how much faster, and how
    much cheaper - so this runs the *same question* against the *same model*
    twice, once with the full context and once with the compressed one, and
    returns both with their measured cost and latency.

    Fairness rules, because a rigged comparison is worse than none:

    * **One model, one set of parameters.** Both arms go through the same
      ``GenerationChain`` with identical temperature and token limits. If the
      chain falls back mid-request it falls back for both.
    * **Both arms run concurrently**, so this is wall-clock under the same
      conditions rather than two runs minutes apart. It also means the two
      calls contend for the same provider, which is the realistic serving
      case - and it is stated in the response rather than hidden.
    * **Cost is computed from measured tokens** and published per-1M pricing,
      not asserted.
    * The authoritative benchmark remains ``python -m eval.harness``, which
      runs sequentially over a 15-item set. This endpoint is one question.
    """
    cfg = get_config()
    if not request.text.strip():
        return error_response(
            "invalid_request", "text is empty", status.HTTP_422_UNPROCESSABLE_ENTITY
        )

    try:
        engine = pipeline(
            request.mode, request.embedding_provider, request.generation_provider
        )
    except (InvalidMode, InvalidProvider) as exc:
        return error_response(
            "invalid_request", str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    except NoProviderAvailable as exc:
        return error_response(
            "no_provider_available", str(exc),
            status.HTTP_503_SERVICE_UNAVAILABLE, mode=request.mode,
        )

    # Query-aware: the question shapes what survives, which is the whole point.
    compressed = engine.compress(
        source=request.text, name=request.name, query=request.question,
        fast_mode=True,
    )

    chain = engine.abstractive.client.chain
    available, why = chain.available()
    if not available:
        return error_response(
            "no_provider_available", why, status.HTTP_503_SERVICE_UNAVAILABLE
        )

    tokenizer = engine.tokenizer
    price = cfg.pricing.price_for(cfg.evaluation.pricing_model)

    def ask(context: str) -> dict:
        started = time.perf_counter()
        try:
            text = chain.generate(
                f"{context}\n\nQuestion: {request.question}",
                max_tokens=request.max_tokens,
                timeout_s=cfg.providers.timeout_s,
                system=ANSWER_SYSTEM_PROMPT,
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - one arm failing is reportable
            text, error = "", str(exc)
        elapsed = (time.perf_counter() - started) * 1000
        tokens_in = tokenizer.count(context)
        tokens_out = tokenizer.count(text)
        return {
            "answer": text,
            "error": error,
            # A context the model refuses as too large is not the same kind of
            # failure as a network blip, and on a big input it is the single
            # most informative thing this endpoint can report: the uncompressed
            # prompt is not merely expensive, it is unusable. Naming it lets
            # the UI say "compression made this answerable" instead of showing
            # a red error next to a working result.
            "too_large": bool(error) and _is_context_too_large(error),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(elapsed, 1),
            "cost_usd": round(
                (tokens_in * price.input_per_1m + tokens_out * price.output_per_1m)
                / 1_000_000, 6
            ),
        }

    # Concurrent so the comparison is one moment, not two.
    results: dict[str, dict] = {}
    threads = [
        threading.Thread(target=lambda k, c: results.__setitem__(k, ask(c)),
                         args=(key, context))
        for key, context in (
            ("full", request.text),
            ("compressed", compressed.compressed_text),
        )
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    full, small = results["full"], results["compressed"]

    # The strongest outcome this endpoint can report, and it happens on any
    # input big enough to matter: the full prompt was REFUSED for size while
    # the compressed one answered. That is not "compression is cheaper", it is
    # "compression is the difference between a usable prompt and no answer".
    unlocked = bool(full.get("too_large")) and not small["error"]

    return {
        "unlocked": unlocked,
        "question": request.question,
        "model": engine.abstractive.client.model,
        "mode": compressed.mode,
        "providers_used": compressed.providers_used,
        "full": full,
        "compressed": small,
        "delta": {
            "tokens_saved": full["tokens_in"] - small["tokens_in"],
            "compression_pct": compressed.summary()["compression_pct"],
            "cost_saved_usd": round(full["cost_usd"] - small["cost_usd"], 6),
            "cost_reduction_pct": round(
                100 * (1 - small["cost_usd"] / full["cost_usd"]), 1
            ) if full["cost_usd"] else 0.0,
            "speedup": round(full["latency_ms"] / small["latency_ms"], 2)
            if small["latency_ms"] else 0.0,
            "pricing_model": cfg.evaluation.pricing_model,
            "full_context_rejected": bool(full.get("too_large")),
        },
        "confidence": compressed.confidence.to_dict() if compressed.confidence else None,
        "compressed_text": compressed.compressed_text,
        "note": (
            "Both arms ran concurrently against the same model and parameters, "
            "so they contend for one provider - realistic serving conditions, "
            "but not a controlled latency benchmark. For that, "
            "`python -m eval.harness` runs a 15-item set sequentially."
        ),
    }


@app.get("/expand/{compression_id}/{marker_id}", tags=["compression"])
def expand(compression_id: str, marker_id: str) -> Any:
    """Recover the content behind one marker. The escape hatch.

    Every other compressor in this space is one-way: it decides what to drop
    and the caller lives with it. Because stage 7 already annotates each
    omission, and each annotation now carries an id, a consumer that reads
    ``[... omitted #d3 4 section(s) ...]`` and decides it actually needs that
    material can ask for exactly it - instead of re-running the whole
    compression at a looser budget and hoping.

    That turns the marker from an apology into an address, and makes the
    compression a *lossy view over retained content* rather than destruction.
    """
    with _EXPAND_LOCK:
        entry = EXPANSIONS.get(compression_id)
    if entry is None:
        return error_response(
            "not_found",
            f"no compression {compression_id}; it may have aged out of the "
            f"cache (the last {MAX_EXPANSIONS} compressions are retained)",
            status.HTTP_404_NOT_FOUND,
        )
    recovered = entry["markers"].get(marker_id)
    if recovered is None:
        return error_response(
            "not_found",
            f"no marker {marker_id!r} in compression {compression_id}",
            status.HTTP_404_NOT_FOUND,
            available=sorted(entry["markers"]),
        )
    return {
        "compression_id": compression_id,
        "marker_id": marker_id,
        **recovered,
    }


@app.get("/providers", tags=["meta"])
def providers() -> dict[str, Any]:
    """Every selectable model, per role, with whether it can run right now.

    What a model picker is built from. Each entry carries the concrete model id
    and its configured state, so the UI never has to hardcode a model name and
    never offers one that would 503. Contains no keys - `configured` is a
    boolean and nothing more.
    """
    cfg = get_config()
    catalogue = provider_catalogue(cfg)
    return {
        **catalogue,
        # One entry per selectable model, each already resolved to a coherent
        # (embedding, generation) pair. This is what the dashboard's single
        # dropdown renders; the per-role fields below remain available for
        # callers that want an exotic combination.
        "selections": selection_presets(cfg),
        "modes": {"available": list(MODES), "default": DEFAULT_MODE},
        "note": (
            "Pin a provider with embedding_provider / generation_provider on "
            "/compress. A pinned provider gets no fallback: if it fails the "
            "request errors rather than silently using a different model."
        ),
    }


@app.get("/samples", tags=["meta"])
def samples() -> list[dict[str, Any]]:
    """Sample corpus files, so the dashboard can offer one-click demos."""
    root = SAMPLE_ROOT
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            text = path.read_text(encoding="utf-8", errors="replace")
            out.append({
                "name": path.name,
                "path": str(path.relative_to(root)),
                "kind": path.parent.name,
                "characters": len(text),
                "tokens": pipeline().tokenizer.count(text),
            })
    return out


@app.get("/samples/{kind}/{name}", tags=["meta"])
def sample(kind: str, name: str) -> Any:
    root = SAMPLE_ROOT / kind
    path = (root / name).resolve()
    if not path.is_file() or root.resolve() not in path.parents:
        return error_response("not_found", f"no sample {kind}/{name}", status.HTTP_404_NOT_FOUND)
    return {"name": name, "text": path.read_text(encoding="utf-8", errors="replace")}
