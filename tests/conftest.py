"""
Pytest configuration and shared fixtures for Ball Knower tests

Provides utilities for loading synthetic test fixtures without requiring
real NFL data files.
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """
    Return the path to the tests/fixtures directory.

    Returns:
        Path: Absolute path to tests/fixtures/

    Example:
        def test_something(fixtures_dir):
            csv_path = fixtures_dir / "power_ratings_nfelo_2025_week_11.csv"
            df = pd.read_csv(csv_path)
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir(fixtures_dir) -> Path:
    """
    Alias for fixtures_dir to match loader function signatures.

    Many loader functions accept a `data_dir` parameter, so this fixture
    provides a drop-in replacement for tests.

    Returns:
        Path: Absolute path to tests/fixtures/

    Example:
        def test_loader(data_dir):
            df = loaders.load_power_ratings("nfelo", 2025, 11, data_dir=data_dir)
    """
    return fixtures_dir


def load_fixture(name: str, fixtures_dir: Path = None) -> pd.DataFrame:
    """
    Load a CSV fixture by name.

    Args:
        name: Fixture filename (e.g., "power_ratings_nfelo_2025_week_11.csv")
        fixtures_dir: Optional path to fixtures directory (auto-detected if None)

    Returns:
        DataFrame loaded from the fixture CSV

    Raises:
        FileNotFoundError: If the fixture file does not exist

    Example:
        df = load_fixture("power_ratings_nfelo_2025_week_11.csv")
        assert len(df) > 0
    """
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent / "fixtures"

    fixture_path = fixtures_dir / name

    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Fixture not found: {name}\n"
            f"Expected location: {fixture_path}\n"
            f"Available fixtures: {list(fixtures_dir.glob('*.csv'))}"
        )

    return pd.read_csv(fixture_path)


@pytest.fixture
def load_fixture_func(fixtures_dir):
    """
    Fixture that returns the load_fixture function with fixtures_dir pre-bound.

    Returns:
        Callable: load_fixture function with fixtures_dir pre-set

    Example:
        def test_something(load_fixture_func):
            df = load_fixture_func("power_ratings_nfelo_2025_week_11.csv")
            assert len(df) == 3
    """
    def _load(name: str) -> pd.DataFrame:
        return load_fixture(name, fixtures_dir=fixtures_dir)

    return _load
