"""
Test ball_knower.io.schemas module

Tests schema validation for nfelo and Substack data sources.
Validates that:
- Valid DataFrames pass validation
- Invalid DataFrames (missing columns, wrong dtypes) raise clear ValueError messages
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.io import schemas


# ============================================================================
# TEST NFELO POWER RATINGS SCHEMA
# ============================================================================

def test_nfelo_power_ratings_valid():
    """Test that valid nfelo power ratings pass validation."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'nfelo': [1650.0, 1625.0, 1600.0],
        'QB Adj': [50.0, 45.0, 40.0],
        'Value': [100.0, 95.0, 90.0],
    })

    # Should not raise
    schemas.validate_nfelo_power_ratings_df(df)


def test_nfelo_power_ratings_missing_column():
    """Test that nfelo power ratings with missing required column fails."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        # Missing 'nfelo' column
        'QB Adj': [50.0, 45.0, 40.0],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        schemas.validate_nfelo_power_ratings_df(df)


def test_nfelo_power_ratings_non_numeric():
    """Test that nfelo power ratings with non-numeric column fails."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'nfelo': ['high', 'medium', 'low'],  # Should be numeric
    })

    with pytest.raises(ValueError, match="should be numeric"):
        schemas.validate_nfelo_power_ratings_df(df)


def test_nfelo_power_ratings_empty():
    """Test that empty nfelo power ratings fails."""
    df = pd.DataFrame(columns=['team', 'nfelo'])

    with pytest.raises(ValueError, match="empty"):
        schemas.validate_nfelo_power_ratings_df(df)


# ============================================================================
# TEST NFELO EPA TIERS SCHEMA
# ============================================================================

def test_nfelo_epa_tiers_valid():
    """Test that valid nfelo EPA tiers pass validation."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'epa_off': [0.15, 0.12, 0.10],
        'epa_def': [-0.10, -0.08, -0.05],
        'epa_margin': [0.25, 0.20, 0.15],
    })

    # Should not raise
    schemas.validate_nfelo_epa_tiers_df(df)


def test_nfelo_epa_tiers_missing_column():
    """Test that nfelo EPA tiers with missing required column fails."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'epa_off': [0.15, 0.12, 0.10],
        # Missing 'epa_def' column
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        schemas.validate_nfelo_epa_tiers_df(df)


def test_nfelo_epa_tiers_non_numeric():
    """Test that nfelo EPA tiers with non-numeric EPA fails."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'epa_off': ['good', 'better', 'best'],  # Should be numeric
        'epa_def': [-0.10, -0.08, -0.05],
    })

    with pytest.raises(ValueError, match="should be numeric"):
        schemas.validate_nfelo_epa_tiers_df(df)


# ============================================================================
# TEST NFELO SOS SCHEMA
# ============================================================================

def test_nfelo_sos_valid():
    """Test that valid nfelo SOS pass validation."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'SOS': [0.5, 0.48, 0.52],
    })

    # Should not raise
    schemas.validate_nfelo_sos_df(df)


def test_nfelo_sos_missing_team():
    """Test that nfelo SOS without team column fails."""
    df = pd.DataFrame({
        'SOS': [0.5, 0.48, 0.52],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        schemas.validate_nfelo_sos_df(df)


# ============================================================================
# TEST SUBSTACK POWER RATINGS SCHEMA
# ============================================================================

def test_substack_power_ratings_valid():
    """Test that valid Substack power ratings pass validation."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'Off.': [8.5, 8.2, 8.0],
        'Def.': [7.5, 7.2, 7.0],
        'Ovr.': [8.0, 7.7, 7.5],
    })

    # Should not raise
    schemas.validate_substack_power_ratings_df(df)


def test_substack_power_ratings_missing_column():
    """Test that Substack power ratings with missing required column fails."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'Off.': [8.5, 8.2, 8.0],
        # Missing 'Def.' and 'Ovr.' columns
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        schemas.validate_substack_power_ratings_df(df)


def test_substack_power_ratings_non_numeric():
    """Test that Substack power ratings with non-numeric rating fails."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'Off.': ['A', 'B', 'C'],  # Should be numeric
        'Def.': [7.5, 7.2, 7.0],
        'Ovr.': [8.0, 7.7, 7.5],
    })

    with pytest.raises(ValueError, match="should be numeric"):
        schemas.validate_substack_power_ratings_df(df)


# ============================================================================
# TEST SUBSTACK QB EPA SCHEMA
# ============================================================================

def test_substack_qb_epa_valid():
    """Test that valid Substack QB EPA pass validation."""
    df = pd.DataFrame({
        'team': ['KC', 'BUF', 'SF'],
        'EPA': [0.20, 0.18, 0.15],
        'Player': ['Patrick Mahomes', 'Josh Allen', 'Brock Purdy'],
    })

    # Should not raise
    schemas.validate_substack_qb_epa_df(df)


def test_substack_qb_epa_missing_team():
    """Test that Substack QB EPA without team column fails."""
    df = pd.DataFrame({
        'EPA': [0.20, 0.18, 0.15],
        'Player': ['Patrick Mahomes', 'Josh Allen', 'Brock Purdy'],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        schemas.validate_substack_qb_epa_df(df)


# ============================================================================
# TEST SUBSTACK WEEKLY PROJECTIONS SCHEMA
# ============================================================================

def test_substack_weekly_proj_valid():
    """Test that valid Substack weekly projections pass validation."""
    df = pd.DataFrame({
        'team_away': ['KC', 'BUF'],
        'team_home': ['SF', 'MIA'],
        'Win Prob.': [0.55, 0.62],
    })

    # Should not raise
    schemas.validate_substack_weekly_proj_df(df)


def test_substack_weekly_proj_empty():
    """Test that empty Substack weekly projections fails."""
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        schemas.validate_substack_weekly_proj_df(df)


# ============================================================================
# TEST NFELO HISTORICAL SCHEMA
# ============================================================================

def test_nfelo_historical_valid():
    """Test that valid nfelo historical games pass validation."""
    df = pd.DataFrame({
        'game_id': ['2024_01_KC_BUF', '2024_01_SF_MIA'],
        'season': [2024, 2024],
        'week': [1, 1],
        'home_team': ['BUF', 'MIA'],
        'away_team': ['KC', 'SF'],
        'starting_nfelo_home': [1625.0, 1580.0],
        'starting_nfelo_away': [1650.0, 1600.0],
        'home_line_close': [3.0, -2.5],
    })

    # Should not raise
    schemas.validate_nfelo_historical_df(df)


def test_nfelo_historical_missing_column():
    """Test that nfelo historical with missing required column fails."""
    df = pd.DataFrame({
        'game_id': ['2024_01_KC_BUF'],
        'season': [2024],
        'week': [1],
        'home_team': ['BUF'],
        # Missing several required columns
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        schemas.validate_nfelo_historical_df(df)


def test_nfelo_historical_non_numeric():
    """Test that nfelo historical with non-numeric season fails."""
    df = pd.DataFrame({
        'game_id': ['2024_01_KC_BUF'],
        'season': ['twenty-twenty-four'],  # Should be numeric
        'week': [1],
        'home_team': ['BUF'],
        'away_team': ['KC'],
        'starting_nfelo_home': [1625.0],
        'starting_nfelo_away': [1650.0],
        'home_line_close': [3.0],
    })

    with pytest.raises(ValueError, match="should be numeric"):
        schemas.validate_nfelo_historical_df(df)
