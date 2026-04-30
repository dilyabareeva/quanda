"""Core fixtures importing submodules."""
import os

import pytest

from tests._fixtures.mnist import *  # noqa: F401, F403
from tests._fixtures.synthetic import *  # noqa: F401, F403
from tests._fixtures.text import *  # noqa: F401, F403


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local: only run this test if running locally"
    )


def pytest_runtest_setup(item):
    if "local" in item.keywords and os.getenv("GITHUB_ACTIONS"):
        pytest.skip("Skipping local-only tests on GitHub Actions")
