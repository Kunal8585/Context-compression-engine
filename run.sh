#!/usr/bin/env bash
# One-command local run.
#
#   ./run.sh              backend + frontend (if built)
#   ./run.sh --backend    API only
#   ./run.sh --check      verify the environment and exit
#
# Everything runs locally. No API key is required: the OpenAI judge is optional
# and the harness falls back to deterministic key-fact recall without it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
MODE="${1:-all}"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$1"; }
fail()  { printf '\033[1;31m[x]\033[0m %s\n' "$1"; exit 1; }

# --- environment -------------------------------------------------------------
[ -x "$PY" ] || fail "no virtualenv at $VENV - run ./scripts/setup.sh first"

check_env() {
  info "Checking environment"
  "$PY" - <<'PY'
import sys
from engine.config import get_config
from engine.tokenizer import get_tokenizer
from engine.embeddings import EmbeddingModel
from engine.entities import EntityScorer

cfg = get_config()
tk = get_tokenizer(cfg.tokenizer)
print(f"  tokenizer   : {tk.backend}" + ("" if tk.is_exact else "  [ESTIMATE]"))

emb = EmbeddingModel(cfg.redundancy)
print(f"  embeddings  : {'ok' if emb.available else 'MISSING'} "
      f"({cfg.redundancy.model} on {emb.device})")

ent = EntityScorer(cfg.density.spacy_model)
print(f"  entities    : {'spacy' if ent.available else 'regex fallback'}")

try:
    import requests
    models = [m.get("name","") for m in
              requests.get(f"{cfg.abstractive.host}/api/tags", timeout=3).json().get("models", [])]
    print(f"  ollama      : {models or 'running, no models pulled'}")
except Exception as exc:
    print(f"  ollama      : unavailable ({exc}) - stages 6 and 8 will degrade")

from pathlib import Path
report = Path("reports/latest.json")
print(f"  eval report : {'present' if report.exists() else 'MISSING - run python -m eval.harness'}")
PY
}

check_env
[ "$MODE" = "--check" ] && { info "Environment OK"; exit 0; }

# --- shutdown ----------------------------------------------------------------
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  printf '\n'
  info "Stopped"
}
trap cleanup INT TERM EXIT

# --- backend -----------------------------------------------------------------
info "Starting API on http://localhost:$API_PORT  (docs at /docs)"
"$VENV/bin/uvicorn" backend.main:app --host 0.0.0.0 --port "$API_PORT" &
PIDS+=($!)

# Wait for the port, then for the models to finish warming. A judge's first
# request should measure compression, not a lazy model load.
for _ in $(seq 1 60); do
  curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1 \
  || fail "API failed to start on port $API_PORT"

info "Warming models (first load is ~10s; after this /compress is ~250ms)"
for _ in $(seq 1 120); do
  warm=$(curl -sf "http://localhost:$API_PORT/health" | "$PY" -c \
    'import json,sys; print(json.load(sys.stdin)["warm"])' 2>/dev/null || echo False)
  [ "$warm" = "True" ] && break
  sleep 1
done
info "Backend ready"

# --- frontend ----------------------------------------------------------------
if [ "$MODE" != "--backend" ] && [ -f "$ROOT/frontend/package.json" ]; then
  if command -v npm >/dev/null 2>&1; then
    [ -d "$ROOT/frontend/node_modules" ] || (info "Installing frontend deps"; cd frontend && npm install)
    info "Starting dashboard on http://localhost:$WEB_PORT"
    (cd frontend && VITE_API_URL="http://localhost:$API_PORT" npm run dev -- --port "$WEB_PORT") &
    PIDS+=($!)
  else
    warn "npm not found - skipping the dashboard"
  fi
elif [ "$MODE" != "--backend" ]; then
  warn "frontend/ not built yet - serving the API only"
fi

printf '\n'
info "API   http://localhost:$API_PORT/docs"
[ -f "$ROOT/frontend/package.json" ] && info "Web   http://localhost:$WEB_PORT"
info "Ctrl-C to stop"
wait
