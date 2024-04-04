# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	python3 -m isort .
