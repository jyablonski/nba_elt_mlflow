FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/.venv/bin:/root/.cargo/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=3.12

# install OS-level deps once in base (so we don't repeat in each stage)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------
# Stage: Install Python deps
# -------------------------------
FROM base AS python-deps

ARG INSTALL_DEV_DEPENDENCIES=false

COPY pyproject.toml uv.lock ./

RUN pip install uv==0.6.12 && \
    if [ "$INSTALL_DEV_DEPENDENCIES" = "true" ]; then \
        uv sync --frozen; \
    else \
        uv sync --frozen --no-dev; \
    fi

# -------------------------------
# Stage: Runtime
# -------------------------------
FROM base AS runtime

COPY --from=python-deps ./.venv ./.venv

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

COPY src/ ./src

RUN mkdir logs && touch logs/example.log

CMD ["python3", "-m", "src.app"]