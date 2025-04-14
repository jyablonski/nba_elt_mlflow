# Lints all python files
.PHONY: lint
lint: 
	black src/app.py src/utils.py tests/conftest.py tests/unit_test.py

.PHONY: create-venv
create-venv:
	poetry install

.PHONY: venv
venv:
	poetry shell

.PHONY: test
test:
	@docker compose -f docker/docker-compose-test.yml down
	@docker compose -f docker/docker-compose-test.yml up --exit-code-from ml_script_test_runner

.PHONY: docker-build
docker-build:
	docker build -f docker/Dockerfile -t ml_script_local .

.PHONY: docker-build-test
docker-build-test:
	docker build -f docker/Dockerfile.test -t ml_script_local_test .

.PHONY: docker-run
docker-run:
	docker run --rm ml_script_local

# use to untrack all files and subsequently retrack all files, using up to date .gitignore
.PHONY: git-reset
git-reset:
	git rm -r --cached .
	git add .

PHONY: git-rebase
git-rebase:
	@git checkout master
	@git pull
	@git checkout feature_integration
	@git rebase master
	@git push

.PHONY: bump-patch
bump-patch:
	@bump2version patch
	@git push --tags
	@git push

.PHONY: bump-minor
bump-minor:
	@bump2version minor
	@git push --tags
	@git push

.PHONY: bump-major
bump-major:
	@bump2version major
	@git push --tags
	@git push

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
	@uv run pytest
	@make stop-postgres