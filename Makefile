.PHONY: help install lint test test-unit test-integration test-security test-evaluation run run-ui seed-demo ablation benchmark-cache docker-up docker-down

help:
	@echo "ClinGraph Makefile commands:"
	@echo "  install          - Install dependencies using uv"
	@echo "  lint             - Check formatting and linting with ruff"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-security    - Run security and isolation tests"
	@echo "  test-evaluation  - Run evaluation tests"
	@echo "  run              - Start FastAPI application"
	@echo "  run-ui           - Start Streamlit UI"
	@echo "  seed-demo        - Seed demo documents and graph"
	@echo "  ablation         - Run 8-stage ablation suite"
	@echo "  benchmark-cache  - Run semantic cache & routing benchmark"
	@echo "  docker-up        - Start Postgres, Neo4j, Redis with Docker Compose"
	@echo "  docker-down      - Stop local docker services"

install:
	uv pip install -e ".[dev,ui,graph,cache]"

lint:
	uv run ruff check app tests configs scripts frontend
	uv run ruff format --check app tests configs scripts frontend

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

test-security:
	uv run pytest tests/security

test-evaluation:
	uv run pytest tests/evaluation

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	uv run streamlit run frontend/app.py

seed-demo:
	uv run python scripts/seed_demo.py

ablation:
	uv run python scripts/run_all_ablations.py

benchmark-cache:
	uv run python scripts/benchmark_cache_and_routing.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down
