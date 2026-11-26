ARG ACCELERATOR=cpu
ARG IMAGIES_VARIANT=vision

FROM python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64 AS runtime-cpu-editing
ENV PATH=/app/.venv/bin:$PATH

FROM psyb0t/torchbase:py3.12-torch2.14-latest-cpu AS runtime-cpu-vision

FROM psyb0t/torchbase:py3.12-torch2.14-latest-cu126 AS runtime-cuda-vision

FROM runtime-$ACCELERATOR-$IMAGIES_VARIANT
USER root
COPY --from=ghcr.io/astral-sh/uv:0.11.15@sha256:e590846f4776907b254ac0f44b5b380347af5d90d668138ca7938d1b0c2f98d3 /uv /usr/local/bin/uv
WORKDIR /app
ARG IMAGIES_VARIANT
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml uv.lock ./
RUN if [ "$IMAGIES_VARIANT" = vision ]; then \
        uv export --locked --no-dev --extra vision --quiet --output-file /tmp/imagies-requirements.txt && \
        uv pip install --python "$VIRTUAL_ENV/bin/python" --require-hashes --no-deps \
            -r /tmp/imagies-requirements.txt && \
        uv pip check --python "$VIRTUAL_ENV/bin/python" && \
        rm /tmp/imagies-requirements.txt; \
    else \
        uv sync --locked --no-dev; \
    fi
COPY src ./src
RUN mkdir /data && chown 10001:10001 /data
USER 10001:10001
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=2).close()"]
CMD ["uvicorn", "imagies.server:app", "--host", "0.0.0.0", "--port", "8000"]
