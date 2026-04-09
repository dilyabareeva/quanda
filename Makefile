# Makefile
SHELL = /bin/bash

# Styling
.PHONY: clean-format
clean-format:
	python -m ruff format .
	python -m ruff check --fix .
	ruff check --fix --select D --select E501 quanda
	python -m mypy --check-untyped-defs --show-traceback quanda
	rm -f .coverage
	rm -f .coverage.*
	find . -path ./.venv -prune -o \( -name "__pycache__" -o -name "*.pyc" -o -name "*.pyo" \) -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name "checkpoints" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name "htmlcov" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name "lightning_logs" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name "hydra_logs" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name "logs" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name "outputs" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name ".tmp" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name ".tox" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name ".build" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name ".cache" -print -exec rm -rf {} +
	find . -path ./.venv -prune -o -name '*~' -print -exec rm -f {} +
