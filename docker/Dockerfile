FROM python:3.12-slim AS base

ENV POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

FROM base AS python-deps

COPY docker/install_packages.sh . \
    pyproject.toml . \
    poetry.lock ./

RUN ./install_packages.sh && \
    pip install poetry && \
    poetry install --no-root

FROM base AS runtime

COPY --from=python-deps ./.venv ./.venv
ENV PATH="/.venv/bin:$PATH"

RUN useradd --create-home appuser && \
    mkdir /home/appuser/logs && \
    touch /home/appuser/logs/example.log

WORKDIR /home/appuser
USER appuser

COPY src/ ./src

CMD ["python3", "-m", "src.app"]
