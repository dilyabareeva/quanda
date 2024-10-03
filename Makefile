# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	python -m flake8 quanda --pytest-parametrize-names-type=csv
	python -m isort .
	python -m mypy quanda --check-untyped-defs
	rm -f .coverage
	rm -f .coverage.*
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E "./checkpoints" | xargs rm -rf
	find . | grep -E ".htmlcov" | xargs rm -rf
	find . | grep -E ".lightning_logs" | xargs rm -rf
	find . -name '*~' -exec rm -f {} +
