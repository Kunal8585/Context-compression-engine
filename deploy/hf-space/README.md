---
title: Context Compression Engine API
emoji: 🗜️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Context Compression Engine — API

Algorithmic prompt compression: chunk → deduplicate → score → select →
reconstruct. Shrinks a large context by 70%+ before it reaches an LLM, and
reports exactly what it removed.

**Interactive docs: [`/docs`](./docs)** · health: [`/health`](./health)

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/compress` | Compress a context. Returns the prompt, per-stage metrics, and character spans over the input. |
| `POST` | `/evaluate` | The measured benchmark report (compression, cost, latency, accuracy retention, fact survival). |
| `GET` | `/health` | Readiness and model status. |
| `GET` | `/config` | The thresholds and weights behind every number. |

## Hosted configuration

This Space runs **extractive-only** (`CCE_FORCE_FAST_MODE=1`): stage 6, the
optional local-LLM paraphrase step, is skipped. That is a measured decision, not
a missing feature — on the project's corpus stage 6 saved **0 tokens** while
costing 3–6 s per chunk, because the selector drops the verbose chunks it could
safely compress before stage 6 ever sees them. A request may still pass
`fast_mode: false` to watch it run and be correctly skipped.

Inference here is CPU-only; the local build uses Apple Metal. **Compression
ratio and cost reduction transfer unchanged. Latency figures do not** — the
benchmark's 4.05× speedup was measured on an M2.

Every number returned is produced by a measured run. There is no mocked data in
the response path.
