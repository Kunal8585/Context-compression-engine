# Backend image for Render.
#
# Two builds from one file, selected by --build-arg BUILD=full|lite:
#
#   full  torch (CPU) + sentence-transformers + spaCy. All five density signals
#         and the semantic dedup pass. Measured 80.8% fact survival.
#         Peak RSS ~408 MB, so it fits 512 MB - but only just, and it needs real
#         CPU to encode with. Render's free tier gives 0.1 vCPU, which will make
#         MiniLM encoding painfully slow. Use Starter or above.
#
#   lite  No torch. The pipeline degrades exactly as designed: the embedding
#         pass and the novelty signal report unavailable, weights redistribute
#         over the remaining four, and exact + structural dedup still run.
#         Measured 69.2% fact survival - 11.6 points worse, and honest about it
#         via /health. Peak RSS ~150 MB, comfortable on free tier hardware.
#
# Render injects $PORT; binding to anything else yields a service that builds
# and then never passes a health check.

FROM python:3.12-slim AS base

ARG BUILD=full
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    CCE_FORCE_FAST_MODE=1 \
    CCE_BUILD=${BUILD}

WORKDIR /app
COPY deploy/render/requirements-full.txt deploy/render/requirements-lite.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$BUILD" = "lite" ]; then \
        pip install --no-cache-dir -r requirements-lite.txt ; \
    else \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 && \
        pip install --no-cache-dir -r requirements-full.txt ; \
    fi

# Bake models in so nothing is fetched on the first request.
RUN python -m spacy download en_core_web_sm && \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && \
    if [ "$BUILD" != "lite" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" ; \
    fi

COPY engine/     ./engine/
COPY backend/    ./backend/
COPY eval/       ./eval/
COPY config.yaml ./config.yaml
COPY data/       ./data/
# The measured benchmark - /evaluate serves this to fill the hero cards.
COPY reports/    ./reports/

# Render sets $PORT. Shell form so it expands.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
