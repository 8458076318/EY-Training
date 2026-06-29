.PHONY: dev prod migrate test lint format build push

dev:
	docker-compose -f infra/docker/docker-compose.yml up --build

prod:
	docker-compose -f infra/docker/docker-compose.prod.yml up -d

migrate:
	alembic upgrade head

migrate-down:
	alembic downgrade -1

seed:
	python scripts/seed_db.py

test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ frontend/ tests/
	mypy src/

format:
	black src/ frontend/ tests/
	ruff check --fix src/

build:
	docker build -f infra/docker/Dockerfile.api -t day-planner-api:latest .
	docker build -f infra/docker/Dockerfile.frontend -t day-planner-frontend:latest .
	docker build -f infra/docker/Dockerfile.worker -t day-planner-worker:latest .

deploy-dev:
	kubectl apply -k infra/k8s/overlays/dev

deploy-prod:
	kubectl apply -k infra/k8s/overlays/prod

helm-install:
	helm upgrade --install day-planner infra/helm/ \
		--namespace day-planner \
		--create-namespace \
		-f infra/helm/values.prod.yaml
