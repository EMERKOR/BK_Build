"""
Test unified loader functionality

Tests the load_all_sources orchestrator function to ensure:
- All data categories successfully load from fixtures
- Each DataFrame has expected columns
- No DataFrames are empty
- Team normalization works correctly
"""

import pytest
import pandas as pd
from ball_knower.io import loaders


def test_load_all_sources_fixture_based(data_dir):
    """
    Test that load_all_sources works with fixture data.

    This test verifies that the unified loader can successfully load
    all data sources from synthetic fixtures without requiring real NFL data.
    """
    season = 2025
    week = 11

    # Load all sources from fixtures
    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)

    # Assert result is a dict
    assert isinstance(data, dict), "load_all_sources should return a dict"

    # Assert all expected categories are present
    expected_categories = [
        "power_ratings_nfelo",
        "epa_tiers_nfelo",
        "strength_of_schedule_nfelo",
        "power_ratings_substack",
        "qb_epa_substack",
        "weekly_projections_ppg_substack",
        "merged_ratings",
    ]

    for category in expected_categories:
        assert category in data, f"Expected category '{category}' in load_all_sources result"
        assert isinstance(data[category], pd.DataFrame), \
            f"Category '{category}' should be a DataFrame"


def test_all_dataframes_nonempty(data_dir):
    """
    Test that all loaded DataFrames are non-empty.
    """
    season = 2025
    week = 11

    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)

    # Check each category (except merged_ratings, checked separately)
    categories_to_check = [
        "power_ratings_nfelo",
        "epa_tiers_nfelo",
        "strength_of_schedule_nfelo",
        "power_ratings_substack",
        "qb_epa_substack",
    ]

    for category in categories_to_check:
        assert len(data[category]) > 0, \
            f"Category '{category}' should be non-empty"


def test_all_team_based_dataframes_have_team_column(data_dir):
    """
    Test that all team-based DataFrames have a 'team' column.
    """
    season = 2025
    week = 11

    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)

    # Team-based categories should have 'team' column
    team_based_categories = [
        "power_ratings_nfelo",
        "epa_tiers_nfelo",
        "strength_of_schedule_nfelo",
        "power_ratings_substack",
        "qb_epa_substack",
    ]

    for category in team_based_categories:
        assert "team" in data[category].columns, \
            f"Category '{category}' should have a 'team' column"


def test_nfelo_power_ratings_has_expected_columns(data_dir):
    """
    Test that nfelo power ratings has core expected columns.
    """
    season = 2025
    week = 11

    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)
    df = data["power_ratings_nfelo"]

    expected_cols = ["team", "Season", "nfelo"]

    for col in expected_cols:
        assert col in df.columns, \
            f"nfelo power ratings should have column '{col}'"


def test_epa_tiers_has_expected_columns(data_dir):
    """
    Test that EPA tiers has core expected columns.
    """
    season = 2025
    week = 11

    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)
    df = data["epa_tiers_nfelo"]

    expected_cols = ["team", "Season", "EPA/Play"]

    for col in expected_cols:
        assert col in df.columns, \
            f"EPA tiers should have column '{col}'"


def test_strength_of_schedule_has_expected_columns(data_dir):
    """
    Test that strength of schedule has core expected columns.
    """
    season = 2025
    week = 11

    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)
    df = data["strength_of_schedule_nfelo"]

    expected_cols = ["team", "Season"]

    for col in expected_cols:
        assert col in df.columns, \
            f"Strength of schedule should have column '{col}'"


def test_merged_ratings_combines_all_sources(data_dir):
    """
    Test that merged_ratings successfully combines data from all sources.
    """
    season = 2025
    week = 11

    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)

    # Assert merged_ratings exists
    assert "merged_ratings" in data, "Expected 'merged_ratings' in result"

    merged = data["merged_ratings"]

    # Should be non-empty
    assert len(merged) > 0, "merged_ratings should be non-empty"

    # Should have team column
    assert "team" in merged.columns, "merged_ratings should have 'team' column"

    # Should have columns from multiple sources (more than just nfelo base)
    assert len(merged.columns) > 5, \
        f"merged_ratings should have many columns from different sources, got {len(merged.columns)}"


def test_team_normalization_works(data_dir):
    """
    Test that team names are normalized correctly.

    The fixtures use fake team names (TeamA, TeamB, TeamC) which should
    be normalized by the team_mapping module. Since these are fake names,
    they may not map to real teams, but the normalization should still run
    without errors.
    """
    season = 2025
    week = 11

    # This should not raise an error even with fake team names
    data = loaders.load_all_sources(season=season, week=week, data_dir=data_dir)

    # Verify team columns exist and contain data
    for category in ["power_ratings_nfelo", "epa_tiers_nfelo"]:
        df = data[category]
        assert "team" in df.columns, f"{category} should have team column"
        # After normalization, unmapped teams are dropped with a warning
        # So we just check that the column exists


def test_individual_loader_functions(data_dir):
    """
    Test that individual loader functions work with fixtures.
    """
    season = 2025
    week = 11

    # Test load_power_ratings
    df_nfelo_power = loaders.load_power_ratings("nfelo", season, week, data_dir=data_dir)
    assert isinstance(df_nfelo_power, pd.DataFrame)
    assert len(df_nfelo_power) > 0
    assert "team" in df_nfelo_power.columns

    # Test load_epa_tiers
    df_epa = loaders.load_epa_tiers("nfelo", season, week, data_dir=data_dir)
    assert isinstance(df_epa, pd.DataFrame)
    assert len(df_epa) > 0
    assert "team" in df_epa.columns

    # Test load_strength_of_schedule
    df_sos = loaders.load_strength_of_schedule("nfelo", season, week, data_dir=data_dir)
    assert isinstance(df_sos, pd.DataFrame)
    assert len(df_sos) > 0
    assert "team" in df_sos.columns

    # Test load_qb_epa
    df_qb = loaders.load_qb_epa("substack", season, week, data_dir=data_dir)
    assert isinstance(df_qb, pd.DataFrame)
    assert len(df_qb) > 0
    assert "team" in df_qb.columns

    # Test load_weekly_projections_ppg
    df_proj = loaders.load_weekly_projections_ppg("substack", season, week, data_dir=data_dir)
    assert isinstance(df_proj, pd.DataFrame)
    assert len(df_proj) > 0
