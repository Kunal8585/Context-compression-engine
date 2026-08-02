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
    for section in (
        "tokenizer", "embeddings", "entities", "providers", "generation",
        "capabilities",
    ):
        assert section in payload
    # The UI disables features rather than letting a judge click a dead button.
    assert set(payload["capabilities"]) >= {"abstractive", "evaluate_cached"}


def test_health_reports_provider_status_without_leaking_keys(client):
    """The frontend needs to show which providers are live; never the secrets.

    This is the endpoint most likely to leak a key by accident - it is the one
    whose whole job is talking about credentials - so it is asserted directly:
    every key is reported as a boolean, and nothing key-shaped appears anywhere
    in the serialised response.
    """
    import json
    import os

    from engine.providers.keys import KEY_REGISTRY

    payload = client.get("/health").json()
    providers = payload["providers"]

    assert set(providers["keys_configured"]) == set(KEY_REGISTRY)
    assert all(isinstance(v, bool) for v in providers["keys_configured"].values())
    for role in ("embedding", "generation"):
        assert providers[role]["chain"], f"{role} chain must name its providers"
        assert isinstance(providers[role]["providers"], list)

    body = json.dumps(payload)
    for name in KEY_REGISTRY:
        value = (os.environ.get(name) or "").strip()
        if len(value) >= 12:
            assert value not in body, f"{name} leaked into /health"
    assert "sk-" not in body and "Bearer " not in body


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


def test_compress_accepts_multiple_files_and_tags_spans(client):
    response = client.post("/compress", files=[
        ("files", ("notes.txt", b"Incident ID INC-9\n\nOwner: Ada", "text/plain")),
        ("files", ("service.py", b"def health():\n    return 'ok'\n", "text/plain")),
    ], data={"budget_ratio": "0.5"})
    assert response.status_code == 200
    payload = response.json()
    assert [file["status"] for file in payload["files"]] == ["done", "done"]
    assert {span["source_file"] for span in payload["spans"]} <= {"notes.txt", "service.py"}


# ---------------------------------------------------------------------------
# /compress?mode=
# ---------------------------------------------------------------------------
def test_compress_defaults_to_auto_mode(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py", "fast_mode": True},
    ).json()
    assert payload["summary"]["mode"] == "auto"


@pytest.mark.parametrize("mode", ["local", "auto"])
def test_compress_honours_the_requested_mode(client, sample_code, mode):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "fast_mode": True, "mode": mode},
    ).json()
    assert payload["summary"]["mode"] == mode
    # The suite runs offline, so both resolve to local providers - the point is
    # that the resolved mode and the serving provider are both reported.
    assert payload["summary"]["providers_used"]["redundancy"] == "local"


def test_stages_report_which_provider_served_them(client, sample_code):
    stages = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "fast_mode": True, "mode": "local"},
    ).json()["stages"]

    by_name = {s["name"]: s for s in stages}
    # provider_used is part of the contract on every stage; populated only for
    # the ones that actually call a model.
    assert all("provider_used" in s for s in stages)
    assert by_name["redundancy"]["provider_used"] == "local"
    assert by_name["chunking"]["provider_used"] is None


def test_an_unknown_mode_is_rejected(client, sample_code):
    response = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py", "mode": "gpu"},
    )
    assert response.status_code == 422


def test_cloud_mode_errors_rather_than_degrading_to_local(client, sample_code):
    """Offline, cloud cannot run - and must say so instead of using local."""
    response = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "fast_mode": True, "mode": "cloud"},
    )
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "no_provider_available"
    assert error["detail"]["mode"] == "cloud"


def test_upload_accepts_a_mode_field(client):
    payload = client.post(
        "/compress",
        files=[("files", ("notes.txt", b"Incident INC-9 pool size 8", "text/plain"))],
        data={"fast_mode": "true", "mode": "local"},
    ).json()
    assert payload["summary"]["mode"] == "local"


# ---------------------------------------------------------------------------
# /health mode readiness
# ---------------------------------------------------------------------------
def test_health_reports_local_and_cloud_readiness_independently(client):
    modes = client.get("/health").json()["modes"]

    assert set(modes["available"]) == {"local", "cloud", "auto"}
    assert modes["default"] == "auto"
    for key in ("local", "cloud"):
        assert isinstance(modes[key]["ready"], bool)
        assert "embedding" in modes[key] and "generation" in modes[key]
    # The suite runs offline: local is usable, cloud is not, and the two
    # answers are independent of each other.
    assert modes["local"]["ready"] is True
    assert modes["cloud"]["ready"] is False
    assert modes["cloud"]["reason"]


def test_health_capabilities_expose_mode_flags_for_the_toggle(client):
    capabilities = client.get("/health").json()["capabilities"]
    assert capabilities["mode_local"] is True
    assert capabilities["mode_cloud"] is False


# ---------------------------------------------------------------------------
# /answer — the live A/B, and the fairness rules that make it worth anything
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raced(client, corpus):
    source = (corpus / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")
    return client.post(
        "/answer",
        json={"text": source, "question": "What was PAYMENT_POOL_SIZE reduced to?",
              "name": "p.md", "max_tokens": 64},
    ).json()


def test_answer_returns_both_arms_with_measured_cost_and_latency(raced):
    for arm in ("full", "compressed"):
        entry = raced[arm]
        assert entry["tokens_in"] > 0
        assert entry["latency_ms"] > 0
        assert entry["cost_usd"] >= 0


def test_the_compressed_arm_really_is_smaller(raced):
    assert raced["compressed"]["tokens_in"] < raced["full"]["tokens_in"]
    assert raced["delta"]["tokens_saved"] > 0
    assert raced["delta"]["compression_pct"] > 0


def test_both_arms_are_answered_by_the_same_model(raced):
    """A comparison across two models would measure the models, not compression."""
    assert raced["model"]
    assert raced["providers_used"]


def test_cost_is_derived_from_measured_tokens_not_asserted(raced):
    """Recompute the arithmetic from the published price and the token counts."""
    from engine.config import get_config

    cfg = get_config()
    price = cfg.pricing.price_for(cfg.evaluation.pricing_model)
    for arm in ("full", "compressed"):
        entry = raced[arm]
        expected = (
            entry["tokens_in"] * price.input_per_1m
            + entry["tokens_out"] * price.output_per_1m
        ) / 1_000_000
        assert entry["cost_usd"] == pytest.approx(expected, abs=1e-6)


def test_the_response_discloses_that_the_arms_ran_concurrently(raced):
    """Concurrency is realistic but not a controlled benchmark - say so."""
    assert "concurrently" in raced["note"]
    assert "eval.harness" in raced["note"]


def test_an_empty_question_is_rejected(client, sample_code):
    response = client.post(
        "/answer", json={"text": sample_code, "question": ""}
    )
    assert response.status_code == 422


def test_an_empty_context_is_rejected(client):
    response = client.post("/answer", json={"text": "   ", "question": "what?"})
    assert response.status_code == 422


def test_answer_rejects_an_unknown_model(client, sample_code):
    response = client.post(
        "/answer",
        json={"text": sample_code, "question": "what?",
              "generation_provider": "not-a-model"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /expand — recovering what a marker hides
# ---------------------------------------------------------------------------
def test_compress_returns_addressable_markers(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "budget_ratio": 0.30, "fast_mode": True},
    ).json()

    assert payload["compression_id"]
    assert payload["markers"], "a 30% budget must drop something"
    for marker in payload["markers"]:
        assert f"#{marker['id']}" in payload["compressed_text"]


def test_expand_returns_the_content_behind_a_marker(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "budget_ratio": 0.30, "fast_mode": True},
    ).json()
    marker = payload["markers"][0]

    recovered = client.get(
        f"/expand/{payload['compression_id']}/{marker['id']}"
    ).json()

    assert recovered["marker_id"] == marker["id"]
    assert recovered["sections"] == marker["sections"]
    assert recovered["text"].strip(), "an omission must recover actual content"
    # And it really is content the compressed prompt does not contain.
    assert recovered["text"][:60] not in payload["compressed_text"]


def test_expanded_content_comes_from_the_original(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "budget_ratio": 0.30, "fast_mode": True},
    ).json()
    for marker in payload["markers"][:3]:
        recovered = client.get(
            f"/expand/{payload['compression_id']}/{marker['id']}"
        ).json()
        for chunk in recovered["chunks"]:
            assert chunk["text"] in sample_code


def test_expanding_an_unknown_marker_is_a_404_listing_what_exists(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py", "fast_mode": True},
    ).json()
    response = client.get(f"/expand/{payload['compression_id']}/nope")
    assert response.status_code == 404
    assert response.json()["error"]["detail"]["available"]


def test_expanding_an_unknown_compression_is_a_404(client):
    response = client.get("/expand/deadbeefcafe/d0")
    assert response.status_code == 404
    assert "aged out" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# /compress?query= — query-aware selection
# ---------------------------------------------------------------------------
def test_a_query_changes_what_survives(client, corpus):
    source = (corpus / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")

    def compress(query):
        body = {"text": source, "name": "p.md", "budget_ratio": 0.25,
                "fast_mode": True}
        if query:
            body["query"] = query
        return client.post("/compress", json=body).json()["compressed_text"]

    assert compress("What was the connection pool size?") != compress(
        "How long did the rollback take?"
    )


def test_query_relevance_appears_in_the_density_weights(client, corpus):
    source = (corpus / "docs" / "incident_postmortem.md").read_text(encoding="utf-8")
    payload = client.post(
        "/compress",
        json={"text": source, "name": "p.md", "fast_mode": True,
              "query": "what was the pool size?"},
    ).json()
    density = next(s for s in payload["stages"] if s["name"] == "density")
    assert "query_relevance" in density["details"]["weights"]


def test_upload_accepts_a_query(client):
    payload = client.post(
        "/compress",
        files=[("files", ("notes.md", b"# Incident\n\nPool size was 8.\n\n"
                          b"## Rollback\n\nTook 22 minutes across 3 regions.",
                          "text/markdown"))],
        data={"fast_mode": "true", "query": "how long was the rollback?"},
    ).json()
    assert payload["summary"]["compressed_tokens"] > 0


# ---------------------------------------------------------------------------
# /providers — the catalogue a model picker is built from
# ---------------------------------------------------------------------------
def test_providers_lists_selectable_models_for_both_roles(client):
    payload = client.get("/providers").json()

    assert payload["embedding"] and payload["generation"]
    for role in ("embedding", "generation"):
        for entry in payload[role]:
            assert entry["provider"] and entry["model"]
            assert entry["role"] == role
            assert isinstance(entry["configured"], bool)
            assert isinstance(entry["local"], bool)
    assert payload["default_chains"]["generation"]


def test_providers_never_returns_a_key(client):
    import json
    import os

    from engine.providers.keys import KEY_REGISTRY

    body = json.dumps(client.get("/providers").json())
    for name in KEY_REGISTRY:
        value = (os.environ.get(name) or "").strip()
        if len(value) >= 12:
            assert value not in body


def test_compress_accepts_a_pinned_provider(client, sample_code):
    payload = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py", "fast_mode": True,
              "embedding_provider": "local", "generation_provider": "local"},
    ).json()
    assert payload["summary"]["providers_used"]["redundancy"] == "local"


def test_an_unknown_pinned_provider_is_rejected(client, sample_code):
    response = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "embedding_provider": "not-a-real-model"},
    )
    assert response.status_code == 422
    assert "not-a-real-model" in response.json()["error"]["message"]


def test_a_pinned_cloud_provider_is_refused_offline_rather_than_substituted(
    client, sample_code
):
    """Offline, a pinned cloud model must error - never quietly become local."""
    response = client.post(
        "/compress",
        json={"text": sample_code, "name": "auth_service.py",
              "generation_provider": "groq"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_provider_available"


def test_upload_accepts_pinned_providers(client, sample_code):
    # Needs a real document: stage 3 skips the embedding pass entirely for a
    # single-chunk input, so a one-line file records no provider at all.
    payload = client.post(
        "/compress",
        files=[("files", ("auth_service.py", sample_code.encode(), "text/x-python"))],
        data={"fast_mode": "true", "embedding_provider": "local"},
    ).json()
    assert payload["summary"]["providers_used"]["redundancy"] == "local"


def test_upload_rejects_more_than_the_file_limit_before_parsing(client):
    """Ten files is the cap; an over-cap batch must 413 in milliseconds."""
    from engine.ingestion import MAX_FILES

    response = client.post("/compress", files=[
        ("files", (f"note{i}.txt", b"some content here", "text/plain"))
        for i in range(MAX_FILES + 1)
    ])
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_upload_where_every_file_is_unreadable_explains_why(client):
    """The per-file reasons are the useful part of this failure, so return them."""
    response = client.post("/compress", files=[
        ("files", ("broken.pdf", b"not a pdf at all", "application/pdf")),
        ("files", ("photo.heic", b"\x00\x01\x02binary", "image/heic")),
    ])
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "no_readable_input"
    names = {f["name"] for f in error["detail"]["files"]}
    assert names == {"broken.pdf", "photo.heic"}


def test_upload_spans_carry_the_file_each_range_came_from(client):
    """Provenance is what puts a source badge on every span in the diff view."""
    response = client.post("/compress", files=[
        ("files", ("alpha.txt", b"Incident INC-9 owner Ada Lovelace on call", "text/plain")),
        ("files", ("beta.log", b"2025-01-01 ERROR pool exhausted after 8000ms", "text/plain")),
    ], data={"fast_mode": "true"})

    payload = response.json()
    assert response.status_code == 200
    sources = {span["source_file"] for span in payload["spans"]}
    assert sources <= {"alpha.txt", "beta.log"}
    assert sources, "spans must be tagged with their source file"


def test_pasted_text_and_uploads_compose(client):
    """The two input modes are additive, not mutually exclusive."""
    response = client.post(
        "/compress",
        files=[("files", ("notes.txt", b"Uploaded content about INC-9", "text/plain"))],
        data={"text": "Pasted content about the rollback", "fast_mode": "true"},
    )
    payload = response.json()
    assert response.status_code == 200
    names = [f["name"] for f in payload["files"]]
    assert names == ["notes.txt", "pasted.txt"]


def test_bad_pdf_is_skipped_without_failing_other_files(client):
    response = client.post("/compress", files=[
        ("files", ("broken.pdf", b"not a pdf", "application/pdf")),
        ("files", ("good.log", b"2025-01-01 INFO started", "text/plain")),
    ])
    assert response.status_code == 200
    statuses = {item["name"]: item for item in response.json()["files"]}
    assert statuses["broken.pdf"]["status"] == "skipped"
    assert statuses["good.log"]["status"] == "done"


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


def test_a_too_large_context_is_reported_as_such_not_as_a_crash():
    """On a big input this is the most informative outcome, not a failure.

    A prompt the model refuses for size means the uncompressed context has no
    answer at any price - which is compression's strongest case, and it should
    read as a finding rather than a red error.
    """
    from backend.main import _is_context_too_large

    for refusal in (
        "request too large (413) - lower the batch size",
        "This model's maximum context length is 8192 tokens",
        "prompt exceeds the context window",
        "too many tokens in request",
    ):
        assert _is_context_too_large(refusal), refusal

    for outage in (
        "rejected the key (401): Invalid API Key",
        "timed out after 30s",
        "unreachable (ConnectionError)",
    ):
        assert not _is_context_too_large(outage), outage
