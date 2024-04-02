# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .

# Testing
.PHONY: tests
tests:
	. .venv/bin/activate && py.test tests --cov=src --cov-report=term-missing --cov-fail-under 95
