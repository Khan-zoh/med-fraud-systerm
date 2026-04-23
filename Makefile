.PHONY: help install dev lint format typecheck test ingest transform train serve dashboard docker-up docker-down clean

PYTHON := python
PIP := pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install with dev dependencies + pre-commit hooks
	$(PIP) install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Auto-format code with black + ruff
	black src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run mypy type checker
	mypy src/cms_fwa/

test: ## Run pytest suite
	pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
ingest: ## Run data ingestion pipeline (download + load to DuckDB)
	$(PYTHON) -m cms_fwa.ingestion.run

transform: ## Run dbt transformations
	cd dbt && dbt run --profiles-dir .

dbt-test: ## Run dbt tests
	cd dbt && dbt test --profiles-dir .

train: ## Train all ML models
	$(PYTHON) -m cms_fwa.models.train

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
serve: ## Start FastAPI scoring server
	uvicorn cms_fwa.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit investigator dashboard
	streamlit run src/cms_fwa/serving/dashboard.py --server.port 8501

dagster-dev: ## Start Dagster development UI
	dagster dev -m cms_fwa.dagster_defs

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-up: ## Build and start all services via docker-compose
	docker compose -f docker/docker-compose.yml up --build -d

docker-down: ## Stop all Docker services
	docker compose -f docker/docker-compose.yml down

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
clean: ## Remove caches, compiled files, and generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
	rm -rf dbt/target/ dbt/dbt_packages/ dbt/logs/
