# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .
