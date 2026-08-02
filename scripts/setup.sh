#!/usr/bin/env bash
# One-command environment setup. Idempotent - safe to re-run.
#
#   ./scripts/setup.sh
#
# API keys are optional at every step. Model calls go through ordered fallback
# chains (see config.yaml -> providers) that skip any provider without a key
# and end at local MiniLM + Ollama, so a clean checkout with no .env still runs
# end to end. Copy .env.example to .env to add hosted providers - four of the
# five have a free tier.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_BIN="${PYTHON:-python3.12}"
VENV="$ROOT/.venv"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$1"; }

# --- 1. virtualenv -----------------------------------------------------------
if [ ! -d "$VENV" ]; then
  if command -v uv >/dev/null 2>&1; then
    info "Creating virtualenv with uv"
    uv venv --python 3.12
  else
    info "Creating virtualenv with $PY_BIN"
    "$PY_BIN" -m venv "$VENV"
  fi
else
  info "Virtualenv already present"
fi

# --- 2. python dependencies --------------------------------------------------
info "Installing Python dependencies"
# Always target the interpreter by path. A console script such as
# "$VENV/bin/pip" hard-codes the interpreter that created it, so a copied or
# renamed project directory installs into the ORIGINAL venv while appearing to
# succeed - see the note in run.sh.
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" -r requirements.txt
else
  "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
  "$VENV/bin/python" -m pip install -r requirements.txt
fi

# --- 3. spaCy pipeline (stage 4: entity density) -----------------------------
if "$VENV/bin/python" -c "import en_core_web_sm" >/dev/null 2>&1; then
  info "spaCy model en_core_web_sm already installed"
else
  info "Downloading spaCy model en_core_web_sm (~12 MB)"
  "$VENV/bin/python" -m spacy download en_core_web_sm || \
    warn "spaCy model download failed - the density scorer will run without NER"
fi

# --- 4. sample corpus --------------------------------------------------------
if [ ! -f "data/sample_corpus/logs/checkout_service.log" ]; then
  info "Generating the sample log corpus"
  "$VENV/bin/python" scripts/make_sample_logs.py
fi

# --- 5. local generation provider (tail of the generation chain) -------------
# Optional. Every chain ends at a local provider so the engine still runs with
# no keys and no network; without Ollama the generation chain simply has one
# fewer entry and stage 6 reports a skip.
MODEL="$("$VENV/bin/python" -c "
from engine.config import get_config
print(get_config().providers.models.local_generation)
" 2>/dev/null || echo 'llama3.2:3b')"

if command -v ollama >/dev/null 2>&1; then
  if ollama list 2>/dev/null | grep -q "^${MODEL%%:*}"; then
    info "Ollama model $MODEL already pulled"
  else
    warn "Ollama is installed but $MODEL is not pulled."
    warn "It backs the last entry of the generation chain:"
    warn "    ollama pull $MODEL"
  fi
else
  warn "Ollama not found - the generation chain loses its local fallback."
  warn "Either install from https://ollama.com and: ollama pull $MODEL"
  warn "or configure a free hosted key (see .env.example)."
fi

# --- 5b. .env scaffold -------------------------------------------------------
if [ ! -f "$ROOT/.env" ] && [ -f "$ROOT/.env.example" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  info "Created .env from .env.example - fill in any keys you have (all optional)"
fi

# --- 6. smoke test -----------------------------------------------------------
info "Running the test suite"
"$VENV/bin/python" -m pytest -q

info "Setup complete. Try:"
echo "    .venv/bin/python -m engine.inspect_chunks data/sample_corpus/logs/checkout_service.log --verify"
