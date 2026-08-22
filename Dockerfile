# semcache -- a semantic cache in front of any LLM.
#
#   docker build -t semcache .
#   docker run -it -v semcache-data:/data -e ANTHROPIC_API_KEY=sk-ant-... semcache
#
# The cache lives in the /data volume, so answers survive `docker run` to
# `docker run`. That persistence is the whole point -- without the volume every
# container starts cold and the cache never pays for itself.
FROM python:3.12-slim

# PIP_NO_CACHE_DIR keeps the image small; the other two keep logs readable.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SEMCACHE_HOME=/data \
    SEMCACHE_MODEL_CACHE_DIR=/opt/semcache/models

WORKDIR /app

# Dependency metadata first so a source-only change does not reinstall the world.
COPY pyproject.toml requirements.txt README.md ./
COPY semcache ./semcache

# [all] = local ONNX embedder + all three provider SDKs, so the image works with
# whatever key the user brings. No torch anywhere: fastembed is onnxruntime.
RUN pip install ".[all]"

# Bake the embedding model into the image so the first question in a fresh
# container is fast and needs no network. It must live OUTSIDE /data -- a mounted
# volume there would mask it and force a re-download on every new volume.
ARG PREFETCH_MODEL=1
RUN if [ "$PREFETCH_MODEL" = "1" ]; then \
        python -c "from semcache.embedders import LocalEmbedder; \
LocalEmbedder(cache_dir='$SEMCACHE_MODEL_CACHE_DIR').warm(); print('model cached')"; \
    fi

# Run as a non-root user. /data is chowned so the volume is writable on first run.
RUN useradd --create-home --uid 10001 semcache \
    && mkdir -p /data "$SEMCACHE_MODEL_CACHE_DIR" \
    && chown -R semcache:semcache /data /opt/semcache /app
USER semcache

VOLUME ["/data"]

# Fails if the package or the faiss wheel is broken, without needing a key.
HEALTHCHECK --interval=1m --timeout=10s --start-period=5s --retries=2 \
    CMD ["semcache", "--version"]

ENTRYPOINT ["semcache"]
# No default subcommand: bare `docker run ... semcache` drops into the REPL,
# and `docker run ... semcache bench --offline` works the same way.
CMD []
