# Makefile for Agentic Observability Platform

.PHONY: help install install-dev test lint format clean run docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make run           - Run the application locally"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker Compose stack"
	@echo "  make docker-down   - Stop Docker Compose stack"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-quick:
	pytest tests/ -v -x

lint:
	ruff check src tests
	mypy src
	black --check src tests

format:
	black src tests
	isort src tests
	ruff check --fix src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf build dist *.egg-info

run:
	python main.py

dev:
	uvicorn src.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

demo:
	python scripts/generate_demo_data.py --interval 15

docker-build:
	docker build -t agentic-observability:latest -f docker/Dockerfile .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f app

docker-clean:
	docker-compose down -v
	docker system prune -f

k8s-deploy:
	kubectl apply -f kubernetes/namespace.yaml
	kubectl apply -f kubernetes/configmap.yaml
	kubectl apply -f kubernetes/deployment.yaml
	kubectl apply -f kubernetes/ingress.yaml

k8s-delete:
	kubectl delete -f kubernetes/

k8s-logs:
	kubectl logs -f deployment/observability-platform -n observability

db-init:
	psql -h localhost -U postgres -f docker/init-db.sql

db-migrate:
	# Add migration commands here
	@echo "No migrations configured yet"

.DEFAULT_GOAL := help
