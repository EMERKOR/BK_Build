"""
Test ball_knower.io.loaders.load_team_power_ratings

Tests for the Ball Knower blended team power ratings loader.
"""

import pytest
import pandas as pd
from pathlib import Path

from ball_knower.io.loaders import load_team_power_ratings


def test_load_team_power_ratings_basic():
    """
    Test that load_team_power_ratings loads the Week 12 2025 data correctly.
    """
    season = 2025
    week = 12

    # Load the data
    df = load_team_power_ratings(season=season, week=week)

    # Assert DataFrame is non-empty
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert len(df) > 0, "DataFrame should be non-empty"
    assert len(df) == 32, "Should have 32 teams"


def test_load_team_power_ratings_required_columns():
    """
    Test that all required columns are present.
    """
    season = 2025
    week = 12

    # Load the data
    df = load_team_power_ratings(season=season, week=week)

    # Required columns
    required_cols = [
        "season", "week", "team_code", "team_name",
        "market_rating",
        "nfelo_value", "substack_points", "objective_composite",
        "athletic_rank", "pff_rank", "analyst_composite_rating",
        "structural_edge",
        "subjective_health_adj", "subjective_form_adj", "subjective_total",
        "bk_blended_rating"
    ]

    # Assert all required columns are present
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_load_team_power_ratings_lar_rating():
    """
    Test that LAR (Rams) has the expected blended rating (with structural edge).
    """
    season = 2025
    week = 12

    # Load the data
    df = load_team_power_ratings(season=season, week=week)

    # Get LAR row
    lar_row = df[df["team_code"] == "LAR"]
    assert len(lar_row) == 1, "Should have exactly one LAR row"

    # Check blended rating (approximately 5.017 with structural edge of 1.25)
    lar_rating = lar_row["bk_blended_rating"].iloc[0]
    expected = 5.016577
    assert lar_rating == pytest.approx(expected, abs=1e-3), \
        f"LAR blended rating should be ~{expected}, got {lar_rating}"


def test_load_team_power_ratings_file_not_found():
    """
    Test that FileNotFoundError is raised when file doesn't exist.
    """
    season = 2099
    week = 99

    with pytest.raises(FileNotFoundError) as exc_info:
        load_team_power_ratings(season=season, week=week)

    assert "Team power ratings file not found" in str(exc_info.value)


def test_load_team_power_ratings_season_week_metadata():
    """
    Test that season and week metadata are correct in the DataFrame.
    """
    season = 2025
    week = 12

    # Load the data
    df = load_team_power_ratings(season=season, week=week)

    # Check season and week columns
    assert (df["season"] == season).all(), "All rows should have correct season"
    assert (df["week"] == week).all(), "All rows should have correct week"
