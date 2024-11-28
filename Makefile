# Makefile
SHELL = /bin/bash

# Styling
.PHONY: clean-format
clean-format:
	python -m ruff format .
	python -m ruff check --fix .
	ruff check --fix --select D --select E501 quanda
	python -m mypy quanda --check-untyped-defs
	rm -f .coverage
	rm -f .coverage.*
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ruff_cache" | xargs rm -rf
	find . | grep -E "./checkpoints" | xargs rm -rf
	find . | grep -E ".htmlcov" | xargs rm -rf
	find . | grep -E ".lightning_logs" | xargs rm -rf
	find . -name '*~' -exec rm -f {} +
