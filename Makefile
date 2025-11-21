.PHONY: test
test:
	@docker compose -f docker/docker-compose-test.yml down
	@docker compose -f docker/docker-compose-test.yml up --exit-code-from ml_script_test_runner

.PHONY: docker-build
docker-build:
	docker build -f docker/Dockerfile -t ml_script_local .

.PHONY: docker-run
docker-run:
	docker run --rm ml_script_local

.PHONY: start-mlflow-server
start-mlflow-server:
	@mlflow server --backend-store-uri sqlite:///mflow.db --default-artifact-root ./artifacts

.PHONY: start-postgres
start-postgres:
	@docker compose -f docker/docker-compose-postgres.yml up -d

.PHONY: stop-postgres
stop-postgres:
	@docker compose -f docker/docker-compose-postgres.yml down

.PHONY: ci-test
ci-test:
	@make start-postgres
	@uv run pytest -vv --cov-report term --cov-report xml:coverage.xml --cov=src --color=yes
	@make stop-postgres