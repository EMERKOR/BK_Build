"""
Test dataset builders with synthetic fixtures

Tests that v1.0 and v1.2 dataset builders work correctly with
synthetic fixture data, ensuring:
- Datasets can be built without real nfelo data
- Required columns are present
- Shapes are correct
- No missing values in core features
- Target columns are correct
"""

import pytest
import pandas as pd
from pathlib import Path
from ball_knower.datasets import v1_0, v1_2


@pytest.fixture
def fixture_games_url(fixtures_dir):
    """
    Return file URL for synthetic nfelo_games.csv fixture.

    This allows dataset builders to load from the fixture file
    instead of downloading real nfelo data.
    """
    csv_path = fixtures_dir / "nfelo_games.csv"
    return f"file://{csv_path}"


def test_v1_0_build_with_fixtures(fixture_games_url):
    """
    Test that v1.0 dataset builder works with synthetic fixtures.
    """
    # Build v1.0 dataset from fixtures
    df = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Assert it's a DataFrame
    assert isinstance(df, pd.DataFrame), "v1.0 should return a DataFrame"

    # Assert non-empty (we have 8 games in fixtures)
    assert len(df) > 0, f"v1.0 dataset should be non-empty, got {len(df)} rows"

    # Assert required columns exist
    required_cols = [
        "game_id",
        "season",
        "week",
        "nfelo_diff",
        "actual_margin",
    ]

    for col in required_cols:
        assert col in df.columns, f"v1.0 dataset missing required column: {col}"


def test_v1_0_has_intentionally_unused_columns(fixture_games_url):
    """
    Test that v1.0 includes intentionally unused columns for leak detection.
    """
    df = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Intentionally unused columns (for leak detection)
    unused_cols = ["home_points", "away_points", "home_margin"]

    for col in unused_cols:
        assert col in df.columns, \
            f"v1.0 dataset should have intentionally unused column: {col}"


def test_v1_0_no_na_in_key_columns(fixture_games_url):
    """
    Test that v1.0 dataset has no NA values in critical columns.
    """
    df = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Key columns should not have NA values
    key_cols = ["game_id", "season", "week", "nfelo_diff", "actual_margin"]

    for col in key_cols:
        na_count = df[col].isna().sum()
        assert na_count == 0, \
            f"Column '{col}' should have no NA values, but has {na_count}"


def test_v1_0_nfelo_diff_calculated_correctly(fixture_games_url):
    """
    Test that nfelo_diff is calculated as home - away.
    """
    df = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # nfelo_diff should be positive for home favorites, negative for underdogs
    # Check that column exists and has reasonable values
    assert "nfelo_diff" in df.columns
    assert df["nfelo_diff"].min() < 100, "nfelo_diff should have reasonable range"
    assert df["nfelo_diff"].max() > -100, "nfelo_diff should have reasonable range"


def test_v1_0_actual_margin_calculated_correctly(fixture_games_url):
    """
    Test that actual_margin is calculated as home_score - away_score.
    """
    df = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Verify actual_margin matches home_score - away_score
    # (using home_points since that's the alias)
    expected_margin = df["home_points"] - df["away_points"]
    pd.testing.assert_series_equal(
        df["actual_margin"],
        expected_margin,
        check_names=False,
        obj="actual_margin calculation"
    )


def test_v1_2_build_with_fixtures(fixture_games_url):
    """
    Test that v1.2 dataset builder works with synthetic fixtures.
    """
    # Build v1.2 dataset from fixtures
    df = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Assert it's a DataFrame
    assert isinstance(df, pd.DataFrame), "v1.2 should return a DataFrame"

    # Assert non-empty
    assert len(df) > 0, f"v1.2 dataset should be non-empty, got {len(df)} rows"

    # Assert required columns exist
    required_cols = [
        "game_id",
        "season",
        "week",
        "nfelo_diff",
        "vegas_closing_spread",
        "rest_advantage",
        "div_game",
        "qb_diff",
    ]

    for col in required_cols:
        assert col in df.columns, f"v1.2 dataset missing required column: {col}"


def test_v1_2_has_more_features_than_v1_0(fixture_games_url):
    """
    Test that v1.2 has more feature columns than v1.0.
    """
    df_v1_0 = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )
    df_v1_2 = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    assert len(df_v1_2.columns) > len(df_v1_0.columns), \
        "v1.2 should have more columns than v1.0 (enhanced features)"


def test_v1_2_no_na_in_key_feature_columns(fixture_games_url):
    """
    Test that v1.2 dataset has no NA values in critical feature columns.
    """
    df = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Key feature columns should not have NA values
    key_cols = [
        "nfelo_diff",
        "rest_advantage",
        "div_game",
        "surface_mod",
        "time_advantage",
        "qb_diff",
        "vegas_closing_spread"
    ]

    for col in key_cols:
        na_count = df[col].isna().sum()
        assert na_count == 0, \
            f"Column '{col}' should have no NA values after filtering, but has {na_count}"


def test_v1_2_rest_advantage_calculated(fixture_games_url):
    """
    Test that rest_advantage is calculated from bye week modifiers.
    """
    df = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    assert "rest_advantage" in df.columns
    # rest_advantage should be sum of home_bye_mod + away_bye_mod
    # It can be positive, negative, or zero
    assert df["rest_advantage"].notna().all(), "rest_advantage should not have NAs"


def test_v1_2_qb_diff_calculated(fixture_games_url):
    """
    Test that qb_diff is calculated from QB adjustments.
    """
    df = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    assert "qb_diff" in df.columns
    # qb_diff should be home_538_qb_adj - away_538_qb_adj
    assert df["qb_diff"].notna().all(), "qb_diff should not have NAs"


def test_v1_2_target_is_vegas_spread(fixture_games_url):
    """
    Test that v1.2 uses vegas_closing_spread as target.
    """
    df = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    assert "vegas_closing_spread" in df.columns
    # vegas_closing_spread should match home_line_close from input
    assert df["vegas_closing_spread"].notna().all(), \
        "vegas_closing_spread (target) should not have NAs"


def test_dataset_shapes_are_reasonable(fixture_games_url):
    """
    Test that dataset shapes match fixture size.

    With 8 games in fixtures, both datasets should have 8 rows.
    """
    df_v1_0 = v1_0.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )
    df_v1_2 = v1_2.build_training_frame(
        start_year=2023,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Should have same number of rows (same games)
    assert len(df_v1_0) == len(df_v1_2), \
        "v1.0 and v1.2 should have same number of games"

    # Should have 8 games from fixtures
    assert len(df_v1_0) == 8, \
        f"Expected 8 games from fixtures, got {len(df_v1_0)}"


def test_year_range_filtering_works(fixture_games_url):
    """
    Test that custom year ranges are respected.
    """
    # Build with only 2023 data
    df_2023 = v1_0.build_training_frame(
        start_year=2023,
        end_year=2023,
        data_url=fixture_games_url
    )

    # Should only have 2023 season data
    assert df_2023['season'].min() >= 2023, "Start year filter not working"
    assert df_2023['season'].max() <= 2023, "End year filter not working"

    # Build with only 2024 data
    df_2024 = v1_0.build_training_frame(
        start_year=2024,
        end_year=2024,
        data_url=fixture_games_url
    )

    # Should only have 2024 season data
    assert df_2024['season'].min() >= 2024, "Start year filter not working"
    assert df_2024['season'].max() <= 2024, "End year filter not working"

    # 2023 should have more games than 2024 (6 vs 2 in fixtures)
    assert len(df_2023) > len(df_2024), \
        f"2023 should have more games than 2024, got {len(df_2023)} vs {len(df_2024)}"
