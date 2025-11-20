"""
Tests for structural metrics dataset builder.

Tests end-to-end metric composition and structural_edge calculation.
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.structural.build_structural_dataset import (
    build_structural_metrics_for_season,
    build_structural_metrics_all_seasons,
)


def create_comprehensive_toy_pbp(season=2023):
    """Create a more comprehensive synthetic PBP for full pipeline testing."""
    data = []

    teams = ['A', 'B', 'C']
    weeks = [1, 2, 3, 4]

    for week in weeks:
        for team_idx, team in enumerate(teams):
            opponent = teams[(team_idx + 1) % len(teams)]

            # Regular plays (for OSR/DSR)
            for i in range(20):
                data.append({
                    'game_id': f'{season}_{week:02d}_{team}_{opponent}',
                    'season': season,
                    'week': week,
                    'posteam': team,
                    'defteam': opponent,
                    'down': (i % 4) + 1,
                    'ydstogo': 10,
                    'play_type': 'pass' if i % 2 == 0 else 'run',
                    'epa': 0.2 * (team_idx + 1),  # Team-dependent success
                    'success': 1 if team_idx == 0 else 0,
                    'yardline_100': 50,
                    'quarter': 2,
                    'score_differential': 0,
                })

            # Pass plays (for OLSI)
            for i in range(10):
                data.append({
                    'game_id': f'{season}_{week:02d}_{team}_{opponent}',
                    'season': season,
                    'week': week,
                    'posteam': team,
                    'defteam': opponent,
                    'down': 2,
                    'ydstogo': 10,
                    'play_type': 'pass',
                    'sack': 1 if (i < team_idx) else 0,
                    'qb_hit': 1 if (i < team_idx * 2) else 0,
                    'yardline_100': 50,
                })

            # 4th down situations (for CEA)
            for i in range(3):
                data.append({
                    'game_id': f'{season}_{week:02d}_{team}_{opponent}',
                    'season': season,
                    'week': week,
                    'posteam': team,
                    'defteam': opponent,
                    'down': 4,
                    'ydstogo': 2,
                    'play_type': 'run' if team == 'A' else 'punt',  # Team A aggressive
                    'yardline_100': 50,
                    'quarter': 2,
                    'score_differential': 0,
                })

    return pd.DataFrame(data)


def test_build_structural_metrics_for_season_columns():
    """Test that build_structural_metrics_for_season returns all required columns."""
    pbp = create_comprehensive_toy_pbp(season=2023)

    result = build_structural_metrics_for_season(pbp, season=2023)

    # Should return DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0, "Should have data"

    # Required columns
    required_cols = [
        'season', 'week', 'team',
        'osr_raw', 'dsr_raw',
        'osr_z', 'dsr_z',
        'olsi_raw', 'olsi_z',
        'go_rate_over_expected_raw',
        'wpa_lost_raw',
        'cea_raw', 'cea_z',
        'structural_edge',
    ]

    for col in required_cols:
        assert col in result.columns, f"Missing required column: {col}"


def test_structural_edge_composition():
    """Test that structural_edge is correctly computed from components."""
    pbp = create_comprehensive_toy_pbp(season=2023)

    result = build_structural_metrics_for_season(pbp, season=2023)

    # Filter to rows with valid data (not week 1)
    valid_rows = result[result['week'] >= 2]

    if len(valid_rows) > 0:
        # Manually recompute structural_edge for first valid row
        row = valid_rows.iloc[0]

        expected_edge = (
            0.35 * row['osr_z'] +
            0.35 * row['dsr_z'] +
            0.20 * row['olsi_z'] +
            0.10 * row['cea_z']
        )

        actual_edge = row['structural_edge']

        # Allow small floating point differences
        assert abs(actual_edge - expected_edge) < 1e-6, \
            f"Structural edge mismatch: expected {expected_edge}, got {actual_edge}"


def test_build_structural_metrics_all_seasons():
    """Test that all-seasons builder concatenates correctly."""
    # Create PBP for two seasons
    pbp_2022 = create_comprehensive_toy_pbp(season=2022)
    pbp_2023 = create_comprehensive_toy_pbp(season=2023)

    pbp_all = pd.concat([pbp_2022, pbp_2023], ignore_index=True)

    result = build_structural_metrics_all_seasons(pbp_all, seasons=[2022, 2023])

    # Should have data for both seasons
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    # Check both seasons are present
    seasons_in_result = result['season'].unique()
    assert 2022 in seasons_in_result
    assert 2023 in seasons_in_result

    # Should have structural_edge column
    assert 'structural_edge' in result.columns


def test_structural_metrics_leak_free_week_ordering():
    """Test that structural metrics respect leak-free constraint across all metrics."""
    pbp = create_comprehensive_toy_pbp(season=2023)

    result = build_structural_metrics_for_season(pbp, season=2023)

    # Week 1 should have NaN/zero structural_edge (no prior data)
    week1 = result[result['week'] == 1]
    if len(week1) > 0:
        # Z-scores should be 0 or NaN for week 1
        assert (week1['osr_z'].fillna(0) == 0).all(), "Week 1 OSR z-scores should be 0 or NaN"

    # Later weeks should have non-zero structural edge for some teams
    later_weeks = result[result['week'] >= 3]
    if len(later_weeks) > 0:
        # At least some rows should have non-zero structural edge
        non_zero_count = (later_weeks['structural_edge'].abs() > 0.01).sum()
        assert non_zero_count > 0, "Later weeks should have some non-zero structural edges"


def test_structural_metrics_no_missing_teams():
    """Test that all teams in each week get structural metrics."""
    pbp = create_comprehensive_toy_pbp(season=2023)

    result = build_structural_metrics_for_season(pbp, season=2023)

    # For each week, should have entries for all teams that played
    for week in [2, 3, 4]:  # Skip week 1 (no prior data)
        week_pbp = pbp[pbp['week'] == week]
        week_result = result[result['week'] == week]

        teams_in_pbp = set(week_pbp['posteam'].unique())
        teams_in_result = set(week_result['team'].unique())

        # All teams in PBP should be in result
        assert teams_in_pbp.issubset(teams_in_result), \
            f"Week {week}: Teams in PBP {teams_in_pbp} not all in result {teams_in_result}"


@pytest.mark.skip(reason="Integration test – requires full pbp dataset")
def test_structural_metrics_real_data_smoke():
    """
    Smoke test with real nflverse data (skipped by default).

    To run: pytest -v --run-integration tests/test_structural_dataset.py
    """
    # This would load real data and run the full pipeline
    # import nfl_data_py as nfl
    # pbp = nfl.import_pbp_data(years=[2023])
    # result = build_structural_metrics_for_season(pbp, season=2023)
    # assert len(result) > 0
    # assert result['structural_edge'].notna().sum() > 100
    pass


def test_structural_edge_range():
    """Test that structural_edge values are reasonable (not extreme)."""
    pbp = create_comprehensive_toy_pbp(season=2023)

    result = build_structural_metrics_for_season(pbp, season=2023)

    # Filter to valid data
    valid_data = result[result['week'] >= 2]

    if len(valid_data) > 0:
        # Structural edge should be roughly in range [-3, 3] (reasonable z-scores)
        min_edge = valid_data['structural_edge'].min()
        max_edge = valid_data['structural_edge'].max()

        assert min_edge > -5, f"Structural edge min too low: {min_edge}"
        assert max_edge < 5, f"Structural edge max too high: {max_edge}"
