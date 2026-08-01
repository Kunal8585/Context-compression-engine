"""Stage 9 tests: the HTTP contract the frontend is built against."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import MAX_INPUT_CHARS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_code(corpus):
    return (corpus / "code" / "auth_service.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_reports_readiness_and_capabilities(client):
    payload = client.get("/health").json()

    assert payload["status"] == "ok"
    assert isinstance(payload["warm"], bool)
    for section in ("tokenizer", "embeddings", "entities", "ollama", "capabilities"):
        assert section in payload
    # The UI disables features rather than letting a judge click a dead button.
    assert set(payload["capabilities"]) >= {"abstractive", "evaluate_cached"}


def test_health_flags_inexact_token_counts(client):
    """If tiktoken ever degrades, the UI must be able to say so."""
    assert "exact" in client.get("/health").json()["tokenizer"]


# ---------------------------------------------------------------------------
# /compress
# ---------------------------------------------------------------------------
def test_compress_returns_the_contract_shape(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py", "budget_ratio": 0.30,
              "fast_mode": True},
    ).json()

    assert {"summary", "stages", "audit_trail", "compressed_text", "spans"} <= set(payload)
    summary = payload["summary"]
    assert summary["compressed_tokens"] <= summary["budget_tokens"]
    assert summary["compression_ratio"] > 0.6
    assert payload["compressed_text"].strip()


def test_compress_always_reports_six_stages(client, sample_code):
    """The accordion is built against a fixed stage list."""
    stages = client.post(
        "/compress", json={"text": sample_code, "fast_mode": True}
    ).json()["stages"]

    assert [s["name"] for s in stages] == [
        "chunking", "redundancy", "density", "selection",
        "abstractive", "reconstruction",
    ]
    for stage in stages:
        if stage["status"] == "skipped":
            assert stage["note"], f"{stage['name']} skipped without a reason"


def test_compress_does_not_echo_the_input_back(client, sample_code):
    """2.17 MB vs 70 KB on the largest sample - the client already has the text."""
    payload = client.post(
        "/compress", json={"text": sample_code, "fast_mode": True}
    ).json()

    assert "original_text" not in payload
    assert "chunks" not in payload
    assert payload["spans"], "the diff view needs spans instead"
    for span in payload["spans"]:
        assert 0 <= span["start"] < span["end"] <= len(sample_code)
        assert isinstance(span["kept"], bool)


def test_compress_can_include_the_original_for_debugging(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "fast_mode": True, "include_original": True,
              "include_chunks": True},
    ).json()
    assert payload["original_text"] == sample_code
    assert payload["chunks"]


def test_compress_never_drops_the_query(client, sample_code):
    question = "Which hashing algorithm is used?"
    payload = client.post(
        "/compress",
        json={"text": sample_code, "budget_ratio": 0.05, "query": question,
              "instruction": "Answer from context only.", "fast_mode": True},
    ).json()

    assert question in payload["compressed_text"]
    assert "Answer from context only." in payload["compressed_text"]


def test_budget_ratio_is_honoured(client, sample_code):
    sizes = []
    for ratio in (0.50, 0.30, 0.15):
        payload = client.post(
            "/compress",
            json={"text": sample_code, "budget_ratio": ratio, "fast_mode": True},
        ).json()
        assert payload["summary"]["compressed_tokens"] <= payload["summary"]["budget_tokens"]
        sizes.append(payload["summary"]["compressed_tokens"])
    assert sizes == sorted(sizes, reverse=True)


def test_fast_mode_skips_the_abstractive_stage(client, sample_code):
    stages = client.post(
        "/compress", json={"text": sample_code, "fast_mode": True}
    ).json()["stages"]
    abstractive = next(s for s in stages if s["name"] == "abstractive")
    assert abstractive["status"] == "skipped"
    assert "fast_mode" in abstractive["note"]


# ---------------------------------------------------------------------------
# Errors - one envelope the frontend can toast
# ---------------------------------------------------------------------------
def test_empty_text_is_rejected_with_an_error_envelope(client):
    response = client.post("/compress", json={"text": "   "})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["message"]


def test_oversized_input_is_rejected(client):
    response = client.post("/compress", json={"text": "x" * (MAX_INPUT_CHARS + 1)})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_out_of_range_budget_is_rejected(client, sample_code):
    assert client.post(
        "/compress", json={"text": sample_code, "budget_ratio": 1.5}
    ).status_code == 422


def test_missing_text_field_is_rejected(client):
    assert client.post("/compress", json={}).status_code == 422


# ---------------------------------------------------------------------------
# /evaluate
# ---------------------------------------------------------------------------
def test_evaluate_serves_the_cached_report(client):
    response = client.post("/evaluate", json={"run": False})
    if response.status_code == 404:
        pytest.skip("no report on disk; run `python -m eval.harness` first")

    report = response.json()
    assert report["cached"] is True
    aggregate = report["aggregate"]
    for metric in (
        "compression_ratio", "cost_reduction", "latency_speedup", "accuracy_retention",
    ):
        assert metric in aggregate, f"{metric} drives a hero card"
    # Provenance: a judge must be able to ask what produced these numbers.
    assert report["config"]["accuracy_metric"] == "key_fact_recall"
    assert report["config"]["downstream_model"]


def test_unknown_evaluation_job_is_a_clean_404(client):
    response = client.get("/evaluate/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------
def test_config_exposes_the_tunables(client):
    payload = client.get("/config").json()
    assert payload["selection"]["budget_ratio"] > 0
    assert sum(payload["density_weights"].values()) == pytest.approx(1.0, abs=0.01)
    assert payload["evaluation"]["accuracy_metric"] == "key_fact_recall"


def test_samples_are_listed_with_token_counts(client):
    samples = client.get("/samples").json()
    assert samples
    names = {s["name"] for s in samples}
    assert "auth_service.py" in names and "checkout_service.log" in names
    for sample in samples:
        assert sample["tokens"] > 0


def test_a_sample_can_be_fetched(client):
    sample = client.get("/samples/code/auth_service.py").json()
    assert sample["text"].startswith('"""Authentication service')


def test_sample_path_traversal_is_refused(client):
    response = client.get("/samples/code/..%2F..%2F..%2Fconfig.yaml")
    assert response.status_code in {403, 404}


def test_openapi_docs_are_available(client):
    """Swagger is the technical judge walkthrough."""
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/compress" in schema["paths"]
