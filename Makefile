.PHONY: test
test:
	uv run pytest

.PHONY: typecheck
typecheck:
	uv run ty check

.PHONY: docker-build
docker-build:
	docker build -f docker/Dockerfile -t ml_script_local .

.PHONY: docker-run
docker-run:
	docker run --rm ml_script_local

.PHONY: start-mlflow-server
start-mlflow-server:
	@mlflow server --backend-store-uri sqlite:///mflow.db --default-artifact-root ./artifacts
