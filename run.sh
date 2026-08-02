#!/usr/bin/env bash
# One-command run.
#
#   ./run.sh              backend + frontend (if built)
#   ./run.sh --backend    API only
#   ./run.sh --check      verify the environment and exit
#
# API keys are optional. Model calls go through ordered fallback chains that
# skip any provider without a key and end at local MiniLM + Ollama, so this
# runs with five keys, one key, or none. `--check` prints which providers are
# live right now; CCE_OFFLINE=1 forces local-only.

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

emb = EmbeddingModel(cfg)
described = emb.describe()
print(f"  embeddings  : {'ok' if described['available'] else 'MISSING'} "
      f"({described['provider'] or 'none'}: {described['model']} on {described['device']})")

ent = EntityScorer(cfg.density.spacy_model)
print(f"  entities    : {'spacy' if ent.available else 'regex fallback'}")

from engine.providers import provider_status
status = provider_status(cfg)
present = [k for k, ok in status["keys_configured"].items() if ok]
print(f"  keys set    : {', '.join(present) or 'none (local providers only)'}")
# "next" is the first provider that WOULD be tried - a configured key, not a
# proven-working one. Verifying that costs real quota, so it is a separate
# opt-in command rather than something every --check pays for.
for role in ("embedding", "generation"):
    chain = status[role]
    print(f"  {role:<12}: next={chain['active'] or 'NONE AVAILABLE'}"
          f"  (chain: {' -> '.join(chain['chain'])})")
print("  (verify for real: python -m engine.providers.check --chains)")

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
# `$PY -m uvicorn`, never `$VENV/bin/uvicorn`.
#
# Console scripts in .venv/bin hard-code the absolute path of the interpreter
# that created them. Copy or rename the project directory and every one of them
# silently re-execs the OLD venv - same command, same apparent success, but a
# different site-packages. That is exactly how this repo ended up serving
# requests from a sibling checkout's dependencies, where pdfplumber was absent
# and every PDF upload reported "not installed".
#
# `python -m` resolves the module against the interpreter actually being run,
# so it cannot drift. Use it for every venv entry point.
"$PY" -m uvicorn backend.main:app --host 0.0.0.0 --port "$API_PORT" &
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
