# Backend image for Hugging Face Spaces (Docker SDK).
#
# Two decisions worth stating up front, because they are what make this fit a
# free tier at all:
#
#   1. CPU-only torch. The default wheel pulls ~2 GB of CUDA libraries that a
#      free Space can neither use nor store. The CPU index cuts the image by
#      roughly an order of magnitude and costs nothing here - there is no GPU
#      to lose.
#   2. Models are baked in at build time. Downloading MiniLM and the spaCy
#      pipeline on first request would make a judge's first click look like a
#      hang, and would fail outright if the Space had no egress.
#
# Ollama is deliberately absent. Stage 6 measured 0 tokens saved on this corpus
# while costing 3-6 s per chunk, so the hosted build runs extractive-only via
# CCE_FORCE_FAST_MODE. That is a measured trade-off, not a limitation - see the
# README's "Known limitations".

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Writable caches; HF Spaces runs as a non-root user.
    HF_HOME=/home/user/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/home/user/.cache/tiktoken \
    # Hosted default: extractive-only. Still overridable per request.
    CCE_FORCE_FAST_MODE=1

RUN useradd -m -u 1000 user
WORKDIR /app

COPY deploy/hf-space/requirements.txt ./requirements.txt

# CPU-only torch first so the resolver cannot pull the CUDA build afterwards.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install --no-cache-dir -r requirements.txt

# Bake the models in. Nothing is fetched at request time.
RUN python -m spacy download en_core_web_sm \
 && python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY --chown=user:user engine/    ./engine/
COPY --chown=user:user backend/   ./backend/
COPY --chown=user:user eval/      ./eval/
COPY --chown=user:user config.yaml ./config.yaml
COPY --chown=user:user data/      ./data/
# The measured benchmark. /evaluate serves this so the hero cards populate
# instantly instead of running minutes of inference during a demo.
COPY --chown=user:user reports/   ./reports/

RUN chown -R user:user /app && mkdir -p /home/user/.cache && chown -R user:user /home/user
USER user

EXPOSE 7860

# One worker: each holds its own copy of MiniLM and spaCy in memory.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
