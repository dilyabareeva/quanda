"""Quanda package."""

from quanda import benchmarks, explainers, metrics, utils

__all__ = ["explainers", "metrics", "benchmarks", "utils"]


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
