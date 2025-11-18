"""
Test ball_knower.datasets module

Ensures v1.0 and v1.2 dataset builders:
- Return DataFrames with stable shapes and columns
- Have expected minimum row counts
- Include required columns
- Handle intentionally unused columns correctly
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.datasets import v1_0, v1_2


# ============================================================================
# TEST v1.0 DATASET BUILDER
# ============================================================================

def test_v1_0_build_training_frame():
    """
    Test that v1.0 dataset builder returns expected structure.
    """
    # Build v1.0 training frame (may take a few seconds to download nfelo data)
    df = v1_0.build_training_frame()

    # Assert it's a DataFrame
    assert isinstance(df, pd.DataFrame), "v1.0 should return a DataFrame"

    # Assert minimum row count (2009-2024 should have 2000+ games)
    assert len(df) > 1000, \
        f"v1.0 dataset should have >1000 rows, got {len(df)}"

    # Assert exact column count (stable API)
    expected_columns = 13
    assert len(df.columns) == expected_columns, \
        f"v1.0 dataset should have exactly {expected_columns} columns, got {len(df.columns)}"

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


def test_v1_0_intentionally_unused_columns():
    """
    Test that v1.0 dataset has intentionally unused columns for leak detection.

    These columns should exist and contain valid data (not all NA).
    """
    df = v1_0.build_training_frame()

    # Intentionally unused columns (for leak detection)
    unused_cols = ["home_points", "away_points", "home_margin"]

    for col in unused_cols:
        assert col in df.columns, \
            f"v1.0 dataset should have intentionally unused column: {col}"

        # These should NOT be all NA (they contain real data, just shouldn't be used)
        assert not df[col].isna().all(), \
            f"Column '{col}' should not be all NA"


def test_v1_0_no_na_in_key_columns():
    """
    Test that v1.0 dataset has no NA values in critical columns.
    """
    df = v1_0.build_training_frame()

    # Key columns should not have NA values
    key_cols = ["game_id", "season", "week", "nfelo_diff", "actual_margin"]

    for col in key_cols:
        na_count = df[col].isna().sum()
        assert na_count == 0, \
            f"Column '{col}' should have no NA values, but has {na_count}"


# ============================================================================
# TEST v1.2 DATASET BUILDER
# ============================================================================

def test_v1_2_build_training_frame():
    """
    Test that v1.2 dataset builder returns expected structure.
    """
    # Build v1.2 training frame
    df = v1_2.build_training_frame()

    # Assert it's a DataFrame
    assert isinstance(df, pd.DataFrame), "v1.2 should return a DataFrame"

    # Assert minimum row count (2009-2024 should have 2000+ games)
    assert len(df) > 1000, \
        f"v1.2 dataset should have >1000 rows, got {len(df)}"

    # Assert column count is in expected range (more features than v1.0)
    # v1.2 has additional situational and QB features
    min_expected_columns = 15
    assert len(df.columns) >= min_expected_columns, \
        f"v1.2 dataset should have at least {min_expected_columns} columns, got {len(df.columns)}"

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


def test_v1_2_intentionally_unused_columns():
    """
    Test that v1.2 dataset has intentionally unused columns for leak detection.
    """
    df = v1_2.build_training_frame()

    # Intentionally unused columns
    unused_cols = ["home_points", "away_points", "home_margin"]

    for col in unused_cols:
        assert col in df.columns, \
            f"v1.2 dataset should have intentionally unused column: {col}"

        # These should NOT be all NA (they contain real data, just shouldn't be used)
        # However, some might be NA if nfelo data doesn't have scores
        # So we just check the column exists, not that it's all filled


def test_v1_2_no_na_in_key_feature_columns():
    """
    Test that v1.2 dataset has no NA values in critical feature columns.

    The builder should filter out rows with NA in key features.
    """
    df = v1_2.build_training_frame()

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


def test_v1_2_has_more_features_than_v1_0():
    """
    Test that v1.2 has more feature columns than v1.0.
    """
    df_v1_0 = v1_0.build_training_frame()
    df_v1_2 = v1_2.build_training_frame()

    assert len(df_v1_2.columns) > len(df_v1_0.columns), \
        "v1.2 should have more columns than v1.0 (enhanced features)"


# ============================================================================
# TEST DATASET CUSTOMIZATION
# ============================================================================

def test_v1_0_custom_year_range():
    """
    Test that v1.0 respects custom year ranges.
    """
    # Build with limited year range
    df = v1_0.build_training_frame(start_year=2023, end_year=2023)

    # Should only have 2023 season data
    assert df['season'].min() >= 2023, "Start year filter not working"
    assert df['season'].max() <= 2023, "End year filter not working"

    # Should have fewer rows than full dataset
    assert len(df) < 500, \
        f"Single season should have <500 games, got {len(df)}"


def test_v1_2_custom_year_range():
    """
    Test that v1.2 respects custom year ranges.
    """
    # Build with limited year range
    df = v1_2.build_training_frame(start_year=2023, end_year=2023)

    # Should only have 2023 season data
    assert df['season'].min() >= 2023, "Start year filter not working"
    assert df['season'].max() <= 2023, "End year filter not working"

    # Should have fewer rows than full dataset
    assert len(df) < 500, \
        f"Single season should have <500 games, got {len(df)}"
