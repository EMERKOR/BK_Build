"""
Tests for Offensive and Defensive Series Success Rate (OSR/DSR) metrics.

Tests leak-free computation and proper series aggregation.
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.structural.osr_dsr import (
    compute_offensive_series_metrics,
    compute_defensive_series_metrics,
    normalize_osr_dsr,
)


def create_toy_pbp(season=2023):
    """Create a minimal synthetic play-by-play DataFrame for testing."""
    data = []

    # Week 1: Team A strong, Team B weak
    for i in range(10):
        data.append({
            'game_id': f'{season}_01_A_B',
            'season': season,
            'week': 1,
            'posteam': 'A',
            'defteam': 'B',
            'down': (i % 4) + 1,
            'ydstogo': 10,
            'play_type': 'pass' if i % 2 == 0 else 'run',
            'epa': 0.5,  # Success
            'success': 1,
        })

    for i in range(10):
        data.append({
            'game_id': f'{season}_01_A_B',
            'season': season,
            'week': 1,
            'posteam': 'B',
            'defteam': 'A',
            'down': (i % 4) + 1,
            'ydstogo': 10,
            'play_type': 'pass' if i % 2 == 0 else 'run',
            'epa': -0.3,  # Failure
            'success': 0,
        })

    # Week 2: Similar pattern
    for i in range(10):
        data.append({
            'game_id': f'{season}_02_A_C',
            'season': season,
            'week': 2,
            'posteam': 'A',
            'defteam': 'C',
            'down': (i % 4) + 1,
            'ydstogo': 10,
            'play_type': 'pass',
            'epa': 0.4,
            'success': 1,
        })

    for i in range(10):
        data.append({
            'game_id': f'{season}_02_B_C',
            'season': season,
            'week': 2,
            'posteam': 'B',
            'defteam': 'C',
            'down': (i % 4) + 1,
            'ydstogo': 10,
            'play_type': 'run',
            'epa': -0.2,
            'success': 0,
        })

    # Week 3: More data
    for i in range(10):
        data.append({
            'game_id': f'{season}_03_A_B',
            'season': season,
            'week': 3,
            'posteam': 'A',
            'defteam': 'B',
            'down': (i % 4) + 1,
            'ydstogo': 10,
            'play_type': 'pass',
            'epa': 0.3,
            'success': 1,
        })

    return pd.DataFrame(data)


def test_osr_dsr_shapes_basic():
    """Test that OSR/DSR return expected (season, week, team) combinations."""
    pbp = create_toy_pbp(season=2023)

    osr = compute_offensive_series_metrics(pbp, season=2023)
    dsr = compute_defensive_series_metrics(pbp, season=2023)

    # Should have data for all weeks and teams
    assert len(osr) > 0, "OSR should return data"
    assert len(dsr) > 0, "DSR should return data"

    # Check required columns
    assert 'season' in osr.columns
    assert 'week' in osr.columns
    assert 'team' in osr.columns
    assert 'osr_raw' in osr.columns

    assert 'season' in dsr.columns
    assert 'week' in dsr.columns
    assert 'team' in dsr.columns
    assert 'dsr_raw' in dsr.columns

    # All values should be for the specified season
    assert (osr['season'] == 2023).all()
    assert (dsr['season'] == 2023).all()


def test_osr_dsr_leak_free_monotonic():
    """Test that OSR for week W only uses data from weeks < W."""
    pbp = create_toy_pbp(season=2023)

    osr = compute_offensive_series_metrics(pbp, season=2023)

    # Week 1 should have NaN (no prior data)
    week1_osr = osr[osr['week'] == 1]
    if len(week1_osr) > 0:
        assert week1_osr['osr_raw'].isna().all(), "Week 1 should have NaN OSR (no prior weeks)"

    # Week 3 should only use weeks 1-2 data
    week3_osr = osr[(osr['week'] == 3) & (osr['team'] == 'A')]
    if len(week3_osr) > 0:
        # Team A was consistently successful in weeks 1-2
        # So week 3 OSR should reflect that (non-NaN, positive value)
        team_a_osr = week3_osr['osr_raw'].iloc[0]
        assert not pd.isna(team_a_osr), "Week 3 should have OSR based on weeks 1-2"
        # Team A had 100% success in toy data, so OSR should be high
        assert team_a_osr > 0.8, f"Team A OSR should be high (got {team_a_osr})"


def test_normalize_osr_dsr():
    """Test that normalization produces valid z-scores."""
    # Create a simple DataFrame with raw OSR/DSR values
    df = pd.DataFrame({
        'season': [2023, 2023, 2023, 2023],
        'week': [2, 2, 3, 3],
        'team': ['A', 'B', 'A', 'B'],
        'osr_raw': [0.8, 0.4, 0.75, 0.35],
        'dsr_raw': [0.6, 0.3, 0.65, 0.25],
    })

    normalized = normalize_osr_dsr(df)

    # Should have z-score columns
    assert 'osr_z' in normalized.columns
    assert 'dsr_z' in normalized.columns

    # Z-scores should have mean ~0 and std ~1 within season
    assert abs(normalized['osr_z'].mean()) < 0.1, "OSR z-scores should have mean ~0"
    assert abs(normalized['dsr_z'].mean()) < 0.1, "DSR z-scores should have mean ~0"

    # Check that higher raw values correspond to higher z-scores
    assert normalized.loc[0, 'osr_z'] > normalized.loc[1, 'osr_z'], \
        "Higher OSR should have higher z-score"


def test_osr_dsr_defensive_perspective():
    """Test that DSR correctly measures defensive performance."""
    pbp = create_toy_pbp(season=2023)

    dsr = compute_defensive_series_metrics(pbp, season=2023)

    # Team A defense (playing against Team B offense)
    # Team B offense was weak (0% success), so Team A defense should be strong
    week3_dsr_a = dsr[(dsr['week'] == 3) & (dsr['team'] == 'A')]

    if len(week3_dsr_a) > 0:
        # DSR for Team A (defending against weak Team B) should be high
        dsr_value = week3_dsr_a['dsr_raw'].iloc[0]
        if not pd.isna(dsr_value):
            assert dsr_value > 0.5, \
                f"Team A DSR should be high (defending weak Team B), got {dsr_value}"
