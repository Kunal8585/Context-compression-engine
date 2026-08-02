# Backend API contract

Status: `/compress` is **implemented and stable** (stages 2-7 run behind it).
`/health` and `/evaluate` are **proposed** - shapes below are for review before
stage 9 locks them in and the frontend is built against them.

All payloads below are real output from the running pipeline, not sketches.

---

## `GET /health`

Lets the dashboard show a "backend ready" state, and - importantly - whether
the models are warm. A cold pipeline takes ~9 s on its first request because
MiniLM and the MPS backend load lazily; a warm one takes ~400 ms. The frontend
should refuse to look ready until `warm` is true.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "warm": true,
  "tokenizer": { "backend": "tiktoken:cl100k_base", "exact": true },
  "embeddings": {
    "model": "text-embedding-3-small", "device": "api", "available": true,
    "provider": "openai", "chain": ["openai", "gemini", "cohere", "local"]
  },
  "entities":   { "backend": "spacy:en_core_web_sm", "available": true },
  "generation": {
    "available": true, "model": "groq:llama-3.3-70b-versatile",
    "chain": ["groq", "gemini", "openai", "local"], "error": null
  },
  "providers": {
    "offline_mode": false,
    "keys_configured": {
      "OPENAI_API_KEY": true,  "GROQ_API_KEY": true,   "GOOGLE_API_KEY": false,
      "COHERE_API_KEY": false, "OPENROUTER_API_KEY": false
    },
    "keys_missing": ["COHERE_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"],
    "embedding": {
      "role": "embedding",
      "chain": ["openai", "gemini", "cohere", "local"],
      "active": "openai",
      "last_used": "openai",
      "providers": [
        { "provider": "openai", "model": "text-embedding-3-small",
          "configured": true, "requires_key": "OPENAI_API_KEY", "reason": null },
        { "provider": "gemini", "model": "text-embedding-004",
          "configured": false, "requires_key": "GOOGLE_API_KEY",
          "reason": "GOOGLE_API_KEY is not configured" }
      ]
    },
    "generation": { "role": "generation", "chain": ["groq", "gemini", "openai", "local"],
                    "active": "groq", "last_used": "groq", "providers": [] },
    "any_embedding_available": true,
    "any_generation_available": true
  },
  "capabilities": { "abstractive": true, "evaluate_cached": true, "evaluate_live": true }
}
```

`capabilities` tells the UI which features to disable rather than letting a
judge click a button that will fail.

**`providers` never contains a key.** `keys_configured` is a presence map of
booleans and is the only thing this endpoint will say about a credential; no
value or fragment of one appears anywhere in the response. A backend test
asserts this directly against every configured key.

`active` is the first provider that *would* be tried — a configured key, not a
proven-working one, since verifying that would cost real quota on every poll.
`last_used` is the one that actually served the most recent call. They differ
exactly when the chain has fallen back, which is the interesting case.

---

## `POST /compress`

### Request

```json
{
  "text": "<raw context>",
  "name": "auth_service.py",
  "budget_ratio": 0.30,
  "kind": "auto",
  "query": "How are passwords hashed?",
  "instruction": "Use only the context.",
  "fast_mode": false,
  "include_text": true
}
```

| Field | Required | Notes |
|---|---|---|
| `text` | yes | raw context |
| `name` | no | drives chunker auto-detection via file extension |
| `budget_ratio` | no | 0.05-1.0, default from `config.yaml` (0.30) |
| `kind` | no | `auto` \| `code` \| `text` \| `log` |
| `query` | no | protected chunk - never dropped |
| `instruction` | no | protected chunk - never dropped |
| `fast_mode` | no | skips stage 6 (abstractive); guaranteed-fast demo path |
| `include_text` | no | see payload-size note below |
| `mode` | no | `local` \| `cloud` \| `auto` (default) — see below |

### `mode`: which providers may serve this request

| Mode | Behaviour |
|---|---|
| `local` | Local providers only. **No outbound API call is made**, even with every key configured. Asserted by a test that severs the network transport and confirms a full compression still succeeds. |
| `cloud` | Configured cloud providers only, in configured order. **Local is deliberately not a fallback.** An exhausted cloud chain returns `503 no_provider_available`. |
| `auto` | The full configured chain — cloud first, local last. Default, and the pre-existing behaviour. |

Why `cloud` refuses to degrade: a user who asked for cloud and silently received
a 3B local model would read its latency and accuracy as cloud's, and draw
exactly the wrong conclusion. A loud 503 is the honest answer.

Both model-calling stages (stage 3 embeddings, stage 6 generation) resolve
under the **same** mode within one request. Embedding in the cloud while
generating locally would produce a run whose reported numbers describe a
configuration nobody chose.

`summary` gains `mode` (the mode that actually executed) and `providers_used`
(stage name → the provider that served it):

```json
"summary": {
  "mode": "cloud",
  "providers_used": { "redundancy": "gemini", "abstractive": "groq" }
}
```

Every entry in `stages[]` carries `provider_used` — the chain entry that
actually **answered**, not the one tried first, so a fallback is visible rather
than implied. It is `null` for the stages that call no model.

```json
{ "name": "redundancy", "provider_used": "gemini", "...": "..." }
{ "name": "chunking",   "provider_used": null,     "...": "..." }
```

`/evaluate` accepts the same `mode` field, applied to its answering step.

### Multi-file upload (`multipart/form-data`)

The same endpoint also accepts a file upload, additively — the JSON contract
above is unchanged. Send `multipart/form-data` with repeated `files` parts:

| Part | Notes |
|---|---|
| `files` | repeated; `.txt` `.md` `.log` `.pdf` and common code extensions |
| `text` | optional; pasted text, appended after the files |
| `budget_ratio` | optional, as above |
| `fast_mode` | optional, as above |

Limits: **10 files**, **5 MB per file**, **10 MB total**. Exceeding any of them
returns `413` before a single file is parsed.

Files are concatenated in **upload order**, separated by a blank line. That is
deterministic: the same files in the same order always produce the same context
and therefore the same compression ratio.

The response gains a `files[]` array, and every entry in `spans[]` gains a
`source_file` naming which upload that range came from:

```json
"files": [
  { "name": "postmortem.pdf", "status": "done",    "reason": null, "characters": 4192, "pages": 3 },
  { "name": "auth_service.py","status": "done",    "reason": null, "characters": 8515 },
  { "name": "scan.pdf",       "status": "skipped",
    "reason": "no extractable text (a scanned or image-only PDF has no text layer)",
    "characters": 0 }
]
```

**A file that cannot be read is `skipped`, never fatal.** A batch of ten where
one is a scan returns nine results and one explanation with HTTP 200. Only when
*every* file yields no text does the endpoint return an error
(`no_readable_input`, 422) — and that error carries the same per-file reasons,
because the reasons are the useful part of it.

**`source_file` attributes a chunk by where it starts.** Files are separated by
a blank line, which the text chunker treats as a paragraph boundary, so chunks
normally sit wholly inside one file. The exception is the small-chunk merge
pass: two short fragments either side of a boundary can merge into one chunk,
and that chunk is attributed to the file its first character came from. This is
deliberate — the alternative is special-casing the chunker for uploads, and
chunk boundaries are load-bearing for dedup and density scoring everywhere
else. The badge marks where a file *begins*, which is what it is read as.

### Response

```json
{
  "summary": {
    "source_name": "auth_service.py",
    "detected_kind": "code",
    "original_tokens": 1952,
    "compressed_tokens": 587,
    "tokens_saved": 1365,
    "compression_ratio": 0.6993,
    "compression_pct": 69.93,
    "budget_tokens": 586,
    "budget_ratio": 0.30,
    "chunks_total": 24,
    "chunks_kept": 7,
    "chunks_dropped": 17,
    "broken_dependencies": 0,
    "repaired_dependencies": 0,
    "total_ms": 407.98,
    "tokenizer_exact": true
  },
  "stages": [ /* one per pipeline stage, see below */ ],
  "audit_trail": [ /* one per dropped chunk, worst-scoring first */ ],
  "broken_dependencies": [],
  "repaired_dependencies": [],
  "original_text": "...",
  "compressed_text": "...",
  "chunks": [ /* per-chunk metadata, no text */ ]
}
```

**`stages[]`** - drives the collapsible stage-by-stage accordion. Every entry
has the same envelope; `details` is stage-specific.

```json
{
  "name": "selection",
  "status": "ok",
  "duration_ms": 0.24,
  "chunks_in": 22, "chunks_out": 7,
  "tokens_in": 1786, "tokens_out": 475,
  "tokens_removed": 1311,
  "reduction_pct": 73.4,
  "note": null,
  "details": {
    "budget_tokens": 586,
    "target_tokens": 586,
    "budget_utilisation": 0.9736,
    "dependencies_repaired": 0,
    "dependencies_broken": 0,
    "dropped_chunks": 15,
    "dropped_tokens": 1334,
    "budget_attempts": 2
  }
}
```

Measured stage timings, warm, on `auth_service.py`:

| Stage | status | ms | tokens |
|---|---|---|---|
| `chunking` | ok | 29.2 | 1952 → 1955 |
| `redundancy` | ok | 193.9 | 1955 → 1786 |
| `density` | ok | 181.7 | 1786 → 1786 |
| `selection` | ok | 0.2 | 1786 → 475 |
| `reconstruction` | ok | 0.4 | 475 → 587 |

**`audit_trail[]`** - why each chunk is missing. This is the reasoning-retention
answer when a judge asks "what did you throw away?"

```json
{
  "id": "auth_service.py#0016:method:TokenService.issue",
  "kind": "method",
  "symbol": "TokenService.issue",
  "lines": "136-159",
  "tokens": 233,
  "density": 0.6363,
  "duplicate_count": 1,
  "reason": "budget_exhausted",
  "preview": "def issue(self, subject: str, tenant_id: str, ..."
}
```

`reason` is one of `budget_exhausted`, `evicted_for_dependency`. Kept chunks
carry `protected`, `density` or `dependency`.

---

## `POST /evaluate` (proposed - stage 8)

```json
{ "test_set": "default", "budget_ratio": 0.30, "run": false }
```

**Execution model (resolved):** the harness runs offline
(`python -m eval.harness`) and writes `reports/latest.json`. With `run: false`
the endpoint serves that report instantly so the hero cards populate
immediately; `run: true` starts a background job and returns a `job_id` to poll.
Numbers are always real and measured - just not necessarily computed during the
demo itself.

```json
{
  "report_id": "2026-08-01T12:04:11Z",
  "generated_at": "2026-08-01T12:04:11Z",
  "cached": true,
  "config": {
    "downstream_model": "groq:llama-3.3-70b-versatile",
    "generation_chain": ["groq", "gemini", "openai", "local"],
    "generation_provider": "groq",
    "generation_attempts": [
      { "provider": "groq", "ok": true, "skipped": false, "reason": "", "duration_ms": 812.4 }
    ],
    "embedding_chain": ["openai", "gemini", "cohere", "local"],
    "embedding_provider": "openai",
    "judge": "gpt-4o-mini | qwen2.5:7b-instruct | exact-match",
    "budget_ratio": 0.30,
    "items": 15
  },
  "aggregate": {
    "compression_ratio": 0.72,
    "tokens_before": 24310, "tokens_after": 6807,
    "cost_before_usd": 0.00365, "cost_after_usd": 0.00102,
    "cost_reduction": 0.72,
    "latency_before_ms": 4120, "latency_after_ms": 1890,
    "latency_speedup": 2.18,
    "accuracy_before": 0.93, "accuracy_after": 0.89,
    "accuracy_retention": 0.957
  },
  "items": [
    {
      "id": "log-01",
      "question": "Which configuration value caused the latency spike?",
      "expected_answer": "PAYMENT_POOL_SIZE was changed from 64 to 8",
      "original":   { "answer": "...", "tokens": 1952, "latency_ms": 410, "score": 1.0 },
      "compressed": { "answer": "...", "tokens": 587,  "latency_ms": 190, "score": 1.0 },
      "verdict": "retained"
    }
  ]
}
```

`verdict` ∈ `retained` | `degraded` | `improved`. Cost is computed from the
published per-1M-token prices in `config.yaml`, never asserted.

---

## Errors

Uniform shape so the frontend can toast any failure without special-casing:

```json
{ "error": { "code": "payload_too_large", "message": "...", "detail": {} } }
```

Codes: `invalid_request`, `payload_too_large`, `pipeline_error`,
`model_unavailable`, `timeout`.

A stage failing internally is **not** an error response - it degrades, reports
`status: "skipped"` with a `note`, and the request still returns 200 with a
usable compressed prompt.

---

## Resolved decisions

### 1. Payload size - RESOLVED: client keeps the original

The frontend already holds the input the judge pasted, so the response never
echoes it back. `/compress` returns `compressed_text` plus a `spans[]` array of
character offsets into the original; the diff view highlights the local copy.

```json
"spans": [
  { "start": 0, "end": 812, "kept": true,  "kind": "module_level",
    "density": 0.44, "reason": "density", "symbol": null, "duplicate_count": 1 },
  { "start": 812, "end": 1103, "kept": false, "kind": "function",
    "density": 0.21, "reason": "budget_exhausted", "symbol": "validate_device_id",
    "duplicate_count": 3 }
]
```

`original_text` is still available behind `include_original: true` for debugging.
Worst case drops from 2.17 MB to ~15 KB.

### 1b. Payload size - measured

Measured, on the 107k-token log at a 30% budget:

| Payload | Bytes |
|---|---|
| Full response | **2,272,189** (2.17 MB) |
| Without `original_text` + `chunks[]` | **2,887** (2.8 KB) |

99.9% of the response is text the frontend needs for the before/after diff
view. 2.17 MB over conference wifi conflicts with the "loads fast on conference
wifi" requirement. Options in the review notes.

### 2. Stage list stability - RESOLVED: always six

`stages[]` always contains all six stages in fixed order (`chunking`,
`redundancy`, `density`, `selection`, `abstractive`, `reconstruction`). A stage
that did not run reports `status: "skipped"` with a `note` explaining why, so
the accordion has a stable shape and skips are visible rather than silent.

Previously `stages[]` had 5 entries; with stage 6 it becomes 6 - but only when
Ollama is available, so the accordion would change shape between environments.
Proposal: **always emit all 6**, with `status: "skipped"` and a `note` when a
stage did not run. Fixed-shape accordion, and the skip is visible rather than
silent.

### 3. Hero metrics - RESOLVED: live where possible, labelled otherwise

Compression ratio and latency animate live from the judge's own snippet.
Accuracy retention and cost reduction display the benchmark figure with a
"from N-item benchmark" caption. All four cards animate; none of them lie.

### 3b. Why - only compression ratio is measurable live from `/compress`. Accuracy
retention, latency speedup and cost reduction require running a task against
both contexts - that is `/evaluate`, over a fixed test set. For a judge's own
pasted snippet there is no expected answer, so accuracy retention cannot
honestly be computed for it.
