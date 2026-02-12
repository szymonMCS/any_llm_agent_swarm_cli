.PHONY: help install install-dev test test-cov lint format check clean build publish docs

help:
	@echo "AgentSwarm Development Commands"
	@echo "==============================="
	@echo "install      - Install package"
	@echo "install-dev  - Install package with dev dependencies"
	@echo "test         - Run tests"
	@echo "test-cov     - Run tests with coverage"
	@echo "lint         - Run linters (ruff)"
	@echo "format       - Format code (black)"
	@echo "check        - Run all checks (lint, format, type-check)"
	@echo "clean        - Clean build artifacts"
	@echo "build        - Build package"
	@echo "publish      - Publish to PyPI"
	@echo "docs         - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -v

test-cov:
	pytest --cov=agentswarm --cov-report=html --cov-report=term

lint:
	ruff check agentswarm tests
	mypy agentswarm

format:
	black agentswarm tests
	ruff check --fix agentswarm tests

check: lint format
	@echo "All checks passed!"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine check dist/*
	twine upload dist/*

docs:
	mkdocs serve
