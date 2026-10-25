.PHONY: help install lint test test-unit test-integration test-security run run-ui docker-up docker-down

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
	@echo "  docker-up        - Start Postgres, Neo4j, Redis with Docker Compose"
	@echo "  docker-down      - Stop local docker services"

install:
	python -m uv pip install -e ".[dev,ui,graph,cache]"

lint:
	python -m ruff check app tests configs
	python -m ruff format --check app tests configs

test:
	python -m pytest

test-unit:
	python -m pytest tests/unit

test-integration:
	python -m pytest tests/integration

test-security:
	python -m pytest tests/security

test-evaluation:
	python -m pytest tests/evaluation

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	python -m streamlit run frontend/app.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down
