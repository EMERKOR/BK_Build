"""
Pytest configuration and fixtures for Ball Knower tests.

This file is automatically loaded by pytest and provides fixtures
that are available to all test files.
"""

import os
from pathlib import Path
import pytest

# Set environment variable BEFORE any test modules import loaders
# This ensures DEFAULT_DATA_DIR in loaders.py picks up the fixture directory
_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "current_season"
os.environ["BALL_KNOWER_DATA_DIR"] = str(_FIXTURES_DIR)


@pytest.fixture(scope="session", autouse=True)
def use_fixture_data_dir():
    """
    Point Ball Knower loaders at the test fixtures directory for all tests.

    This ensures tests do not depend on real data/current_season files.
    The BALL_KNOWER_DATA_DIR environment variable is set at conftest import time
    so that loaders.py picks it up when it's first imported.
    """
    # Already set at module level above, but document it here
    pass
