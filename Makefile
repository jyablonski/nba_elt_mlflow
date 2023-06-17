# Lints all python files
.PHONY: lint
lint: 
	black src/app.py src/utils.py tests/conftest.py tests/unit_test.py

.PHONY: create-venv
create-venv:
	pipenv install

.PHONY: venv
venv:
	pipenv shell

.PHONY: test
test:
	pytest tests/ -v

.PHONY: docker-build
docker-build:
	docker build -t python_docker_local .

.PHONY: docker-run
docker-run:
	docker run --rm python_docker_local

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
