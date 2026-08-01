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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engine import __version__
from engine.config import PROJECT_ROOT, get_config
from engine.pipeline import CompressionPipeline
from eval.harness import REPORTS_DIR, Harness, write_report
from eval.testset import load_testset

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Process-wide state
# --------------------------------------------------------------------------
STATE: dict[str, Any] = {"pipeline": None, "warm": False, "warmup_ms": {}}
JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

MAX_INPUT_CHARS = 2_000_000
SAMPLE_ROOT = PROJECT_ROOT / "data" / "sample_corpus"

#: Deployment default. A hosted instance has no Ollama and no Metal, and stage 6
#: measurably saves 0 tokens on this corpus while costing 3-6s per chunk - so the
#: judge-facing build skips it by default. This is a *default*, not a lock: a
#: request may still pass fast_mode=false to watch stage 6 run and get correctly
#: skipped or rejected live.
FORCE_FAST_MODE = os.environ.get("CCE_FORCE_FAST_MODE", "").lower() in {"1", "true", "yes"}


def pipeline() -> CompressionPipeline:
    if STATE["pipeline"] is None:
        STATE["pipeline"] = CompressionPipeline()
    return STATE["pipeline"]


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


class EvaluateRequest(BaseModel):
    test_set: str = "default"
    budget_ratio: float | None = Field(None, ge=0.05, le=1.0)
    run: bool = Field(False, description="Run fresh instead of serving the last report")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Readiness, including whether the models are warm."""
    cfg = get_config()
    pipe = pipeline()
    ollama_ok = pipe.abstractive.client.available()
    report_exists = (REPORTS_DIR / "latest.json").exists()
    return {
        "status": "ok",
        "version": __version__,
        "warm": STATE["warm"],
        "warmup_ms": {k: round(v) for k, v in STATE["warmup_ms"].items()},
        "tokenizer": pipe.tokenizer.describe(),
        "embeddings": pipe.embedder.describe(),
        "entities": pipe.density.entity_scorer.describe(),
        "ollama": {
            "available": ollama_ok,
            "model": cfg.abstractive.model,
            "error": pipe.abstractive.client.error,
        },
        "force_fast_mode": FORCE_FAST_MODE,
        "capabilities": {
            "abstractive": ollama_ok and cfg.abstractive.enabled and not FORCE_FAST_MODE,
            "evaluate_cached": report_exists,
            "evaluate_live": ollama_ok,
        },
    }


@app.post("/compress", tags=["compression"])
def compress(request: CompressRequest) -> Any:
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

    result = pipeline().compress(
        source=request.text,
        name=request.name,
        budget_ratio=request.budget_ratio,
        kind=request.kind,
        query=request.query,
        instruction=request.instruction,
        fast_mode=(
            FORCE_FAST_MODE if request.fast_mode is None else request.fast_mode
        ),
    )
    return result.to_dict(
        include_original=request.include_original,
        include_chunks=request.include_chunks,
    )


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
            harness = Harness(get_config(), request.budget_ratio)
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
        "abstractive": {
            "enabled": cfg.abstractive.enabled,
            "model": cfg.abstractive.model,
            "entity_overlap_threshold": cfg.abstractive.entity_overlap_threshold,
        },
        "evaluation": {
            "downstream_model": cfg.evaluation.downstream_model,
            "num_ctx": cfg.evaluation.num_ctx,
            "accuracy_metric": "key_fact_recall",
        },
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
