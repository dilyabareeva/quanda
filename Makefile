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
	rm -r quanda.egg-info
	rm -f .coverage.*
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . -type d -name '*cache' -exec rm -rf {} +
	find . | grep -E "./checkpoints" | xargs rm -rf
	find . | grep -E ".htmlcov" | xargs rm -rf
	find . | grep -E ".lightning_logs" | xargs rm -rf
	find . | grep -E ".hydra_logs" | xargs rm -rf
	find . | grep -E ".logs" | xargs rm -rf
	find . | grep -E ".outputs" | xargs rm -rf
	find . -type d -name ".tmp" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".build" -exec rm -rf {} +
	find . -type d -name ".cache" -exec rm -rf {} +
	find . -name '*~' -exec rm -f {} +
