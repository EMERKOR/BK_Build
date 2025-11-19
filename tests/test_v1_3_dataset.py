"""
Tests for Ball Knower v1.3 Dataset Builder

Tests cover:
- Dataset structure and column presence
- Rolling feature calculations
- Leakage validation
- Feature value ranges
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.datasets import v1_3


def test_v1_3_build_training_frame_basic():
    """Test that v1.3 dataset builder runs and returns expected structure."""
    # Build small dataset for testing (2 seasons)
    df = v1_3.build_training_frame(start_year=2019, end_year=2020)

    # Check basic structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "Dataset should have games"

    # Check required columns exist
    required_cols = [
        'game_id', 'season', 'week', 'away_team', 'home_team',
        # v1.2 features
        'nfelo_diff', 'rest_advantage', 'div_game', 'surface_mod',
        'time_advantage', 'qb_diff',
        # Rolling features - home
        'win_rate_L5_home', 'point_diff_L5_home', 'ats_rate_L5_home',
        # Rolling features - away
        'win_rate_L5_away', 'point_diff_L5_away', 'ats_rate_L5_away',
        # Rolling ELO
        'nfelo_change_L3_home', 'nfelo_change_L5_home',
        'nfelo_change_L3_away', 'nfelo_change_L5_away',
        # Game context
        'is_playoff_week', 'is_primetime',
        # Target
        'vegas_closing_spread', 'actual_margin'
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_v1_3_has_v1_2_features():
    """Verify v1.3 includes all v1.2 baseline features."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    v1_2_features = [
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff'
    ]

    for feat in v1_2_features:
        assert feat in df.columns
        # Should have non-null values
        assert df[feat].notna().sum() > 0


def test_v1_3_rolling_features_present():
    """Verify all rolling features are calculated and present."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    rolling_features = [
        # Home rolling
        'win_rate_L5_home', 'point_diff_L5_home', 'ats_rate_L5_home',
        'nfelo_change_L3_home', 'nfelo_change_L5_home',
        # Away rolling
        'win_rate_L5_away', 'point_diff_L5_away', 'ats_rate_L5_away',
        'nfelo_change_L3_away', 'nfelo_change_L5_away',
    ]

    for feat in rolling_features:
        assert feat in df.columns, f"Missing rolling feature: {feat}"


def test_v1_3_rolling_features_in_valid_range():
    """Verify rolling features have reasonable value ranges."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2020)

    # Win rates should be between 0 and 1
    win_rate_cols = ['win_rate_L5_home', 'win_rate_L5_away']
    for col in win_rate_cols:
        assert df[col].min() >= 0, f"{col} has negative values"
        assert df[col].max() <= 1, f"{col} exceeds 1.0"

    # ATS rates should be between 0 and 1
    ats_cols = ['ats_rate_L5_home', 'ats_rate_L5_away']
    for col in ats_cols:
        assert df[col].min() >= 0, f"{col} has negative values"
        assert df[col].max() <= 1, f"{col} exceeds 1.0"

    # Point differential should be reasonable (within NFL score ranges)
    point_diff_cols = ['point_diff_L5_home', 'point_diff_L5_away']
    for col in point_diff_cols:
        assert df[col].abs().max() < 50, f"{col} has unrealistic values"


def test_v1_3_first_games_warmup():
    """Verify first few games per season have low/zero rolling feature values."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    # Get first week games
    first_week = df[df['week'] == 1]

    if len(first_week) > 0:
        # Rolling features should be zero or very low for first games
        # (teams have no history yet)
        rolling_cols = [col for col in df.columns if '_L' in col]

        for col in rolling_cols:
            if col in first_week.columns:
                max_first_week_val = first_week[col].abs().max()
                # Should be minimal (filled with 0 or small values)
                assert max_first_week_val < 5.0, \
                    f"First week {col} has large values: {max_first_week_val}"


def test_v1_3_leakage_validation():
    """Test that leakage validation function passes on v1.3 dataset."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2020)

    # Should not raise
    v1_3.validate_v1_3_no_leakage(df)


def test_v1_3_target_present():
    """Verify target variable is present and valid."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    assert 'vegas_closing_spread' in df.columns
    assert 'actual_margin' in df.columns

    # All rows should have target (after filtering)
    assert df['vegas_closing_spread'].notna().all()

    # Vegas spread should be in reasonable NFL range
    assert df['vegas_closing_spread'].abs().max() < 30


def test_v1_3_no_missing_critical_features():
    """Verify no NaN values in critical feature columns."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    critical_features = [
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff',
        'vegas_closing_spread'
    ]

    for feat in critical_features:
        assert df[feat].notna().all(), f"{feat} has NaN values"


def test_v1_3_game_context_features():
    """Test game context features are calculated."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    assert 'is_playoff_week' in df.columns
    assert 'is_primetime' in df.columns

    # Playoff week should be 0 or 1
    assert set(df['is_playoff_week'].unique()).issubset({0, 1})

    # Should have some playoff week games (week >= 15)
    playoff_games = df[df['week'] >= 15]
    if len(playoff_games) > 0:
        assert playoff_games['is_playoff_week'].sum() > 0


def test_v1_3_team_perspective_consistency():
    """Verify home/away rolling features are properly separated."""
    df = v1_3.build_training_frame(start_year=2019, end_year=2019)

    # Each game should have different rolling features for home and away
    # (unless it's early season with no history)
    late_season = df[df['week'] > 5]

    if len(late_season) > 0:
        # Home and away win rates should not always be identical
        same_win_rate = (
            late_season['win_rate_L5_home'] == late_season['win_rate_L5_away']
        ).sum()

        # Most games should have different win rates for home/away teams
        assert same_win_rate < len(late_season) * 0.5, \
            "Home and away rolling features are too similar"


def test_v1_3_season_filtering():
    """Test that season filtering works correctly."""
    # Request only 2019
    df_2019 = v1_3.build_training_frame(start_year=2019, end_year=2019)

    # Should only have 2019 games
    assert df_2019['season'].min() == 2019
    assert df_2019['season'].max() == 2019

    # Should have reasonable number of games (around 256 regular season)
    assert 200 < len(df_2019) < 300


def test_v1_3_row_count_reasonable():
    """Test that dataset has reasonable number of rows."""
    # 2 seasons should have ~500 games
    df = v1_3.build_training_frame(start_year=2019, end_year=2020)

    assert 400 < len(df) < 600, f"Unexpected row count: {len(df)}"
