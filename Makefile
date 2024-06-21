# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	flake8 . --pytest-parametrize-names-type=csv
	python -m isort .
	rm -f .coverage
	rm -f .coverage.*
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . -name '*~' -exec rm -f {} +
