# syntax=docker/dockerfile:1.5

# ---- Base Stage: Common setup for final image ----
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


# ---- Builder Stage: Build dependencies and cache models ----
# Use a full python image to have build tools available for C extensions
FROM python:3.11 as builder

# Build argument to choose CPU-only or full ML packages
ARG USE_CPU_ONLY=false

ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    XDG_CACHE_HOME=/app/.cache

WORKDIR /app

RUN pip install --no-cache-dir uv
RUN uv venv

ENV UV_HTTP_TIMEOUT=1800

# Copy requirements files for conditional installation
COPY requirements-base.txt requirements-ml.txt requirements-cpu.txt ./

# Install packages with uv. This is a single RUN command to optimize layer caching.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    bash -c ' \
    set -e && \
    . .venv/bin/activate && \
    export UV_CONCURRENT_DOWNLOADS=$(nproc) && \
    export UV_CONCURRENT_INSTALLS=$(nproc) && \
    \
    echo "🚀 Installing base packages..." && \
    uv pip install --no-cache-dir -r requirements-base.txt && \
    \
    if [ "$USE_CPU_ONLY" = "true" ] ; then \
        echo "⚡ Installing CPU-only ML packages..." && \
        uv pip install \
            --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch && \
        uv pip install \
            --no-cache-dir \
            -r requirements-cpu.txt; \
    else \
        echo "🎯 Installing full ML packages (with potential CUDA support)..." && \
        uv pip install \
            --no-cache-dir \
            -r requirements-ml.txt; \
    fi && \
    echo "✅ All packages installed successfully!" \
    '

# Models are NOT baked into the image (Engram option B): they live in the host
# HuggingFace cache, bind-mounted at /hf-cache (shared with Aleph) — see docker-compose.yml.
# This keeps the image lean and avoids re-downloading ~1.6 GB of models on every build.
# Tree-sitter grammars download on first use.
RUN echo "📦 Models load from bind-mounted /hf-cache at runtime (not baked)."

# ---- Final Stage: Create the lean final image ----
FROM base as final

# Copy the virtual environment with all dependencies from the builder stage
COPY --from=builder /app/.venv ./.venv

# Activate venv; models resolve from the bind-mounted host HF cache (/hf-cache), offline.
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/hf-cache \
    HF_HUB_OFFLINE=1

# Copy the application source code
COPY app ./app
COPY templates ./templates

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 