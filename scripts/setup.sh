#!/usr/bin/env bash
# One-command environment setup. Idempotent - safe to re-run.
#
#   ./scripts/setup.sh
#
# Everything here is local. No API keys are required for the core pipeline;
# the OpenAI judge in the eval harness is optional and falls back to a local
# model when OPENAI_API_KEY is absent.

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
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  "$VENV/bin/pip" install --upgrade pip >/dev/null
  "$VENV/bin/pip" install -r requirements.txt
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

# --- 5. local LLM (stage 6 + downstream eval model) --------------------------
# Optional: the pipeline skips abstractive compression when Ollama is absent.
MODEL="$("$VENV/bin/python" -c "
from engine.config import get_config
print(get_config().abstractive.model)
" 2>/dev/null || echo 'llama3.2:3b')"

if command -v ollama >/dev/null 2>&1; then
  if ollama list 2>/dev/null | grep -q "^${MODEL%%:*}"; then
    info "Ollama model $MODEL already pulled"
  else
    warn "Ollama is installed but $MODEL is not pulled."
    warn "Stage 6 (abstractive compression) and the local eval model need it:"
    warn "    ollama pull $MODEL"
  fi
else
  warn "Ollama not found. Stages 6 and 8 will degrade gracefully."
  warn "Install from https://ollama.com, then: ollama pull $MODEL"
fi

# --- 6. smoke test -----------------------------------------------------------
info "Running the test suite"
"$VENV/bin/python" -m pytest -q

info "Setup complete. Try:"
echo "    .venv/bin/python -m engine.inspect_chunks data/sample_corpus/logs/checkout_service.log --verify"
