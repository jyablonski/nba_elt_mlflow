FROM python:3.8-slim as base

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


FROM base AS python-deps

COPY docker/install_packages.sh .
RUN ./install_packages.sh

COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy


FROM base AS runtime

COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

COPY src/app.py ./app.py
COPY src/utils.py ./utils.py
COPY src/log_model.joblib ./log_model.joblib

RUN mkdir logs
RUN cat > logs/example.log

CMD ["python3", "app.py"]