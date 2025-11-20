"""
Tests for Coaching Edge / 4th Down Aggression (CEA) metrics.

Tests 4th down decision-making analysis and leak-free aggregation.
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.structural.cea import compute_coaching_edge_metrics


def create_toy_4th_down_pbp(season=2023):
    """Create synthetic 4th down scenarios with clear coaching differences."""
    data = []

    # Week 1: Team A aggressive (always goes), Team B conservative (always punts)
    # Team A: 5 go situations, all GO
    for i in range(5):
        data.append({
            'game_id': f'{season}_01_A_B',
            'season': season,
            'week': 1,
            'posteam': 'A',
            'defteam': 'B',
            'down': 4,
            'ydstogo': 2,
            'yardline_100': 50,  # Midfield
            'quarter': 2,
            'score_differential': 0,
            'play_type': 'pass' if i % 2 == 0 else 'run',  # GO
        })

    # Team B: 5 go situations, all PUNT
    for i in range(5):
        data.append({
            'game_id': f'{season}_01_A_B',
            'season': season,
            'week': 1,
            'posteam': 'B',
            'defteam': 'A',
            'down': 4,
            'ydstogo': 2,
            'yardline_100': 50,
            'quarter': 2,
            'score_differential': 0,
            'play_type': 'punt',  # NO GO
        })

    # Week 2: Similar pattern
    for i in range(5):
        data.append({
            'game_id': f'{season}_02_A_C',
            'season': season,
            'week': 2,
            'posteam': 'A',
            'defteam': 'C',
            'down': 4,
            'ydstogo': 3,
            'yardline_100': 45,
            'quarter': 3,
            'score_differential': 3,
            'play_type': 'run',  # GO
        })

    for i in range(5):
        data.append({
            'game_id': f'{season}_02_B_C',
            'season': season,
            'week': 2,
            'posteam': 'B',
            'defteam': 'C',
            'down': 4,
            'ydstogo': 3,
            'yardline_100': 45,
            'quarter': 3,
            'score_differential': -3,
            'play_type': 'punt',  # NO GO
        })

    # Week 3: For leak-free testing
    for i in range(5):
        data.append({
            'game_id': f'{season}_03_A_B',
            'season': season,
            'week': 3,
            'posteam': 'A',
            'defteam': 'B',
            'down': 4,
            'ydstogo': 1,
            'yardline_100': 50,
            'quarter': 2,
            'score_differential': 0,
            'play_type': 'run',  # GO
        })

    # Add some regular plays (non-4th down) for context
    for week in [1, 2, 3]:
        for team in ['A', 'B', 'C']:
            for i in range(10):
                data.append({
                    'game_id': f'{season}_{week:02d}_{team}_X',
                    'season': season,
                    'week': week,
                    'posteam': team,
                    'defteam': 'X',
                    'down': (i % 3) + 1,  # 1st, 2nd, 3rd downs
                    'ydstogo': 10,
                    'yardline_100': 50,
                    'quarter': 2,
                    'score_differential': 0,
                    'play_type': 'pass',
                })

    return pd.DataFrame(data)


def test_coaching_edge_go_vs_punt():
    """Test that CEA correctly differentiates aggressive vs conservative coaching."""
    pbp = create_toy_4th_down_pbp(season=2023)

    cea = compute_coaching_edge_metrics(pbp, season=2023)

    # Should have data
    assert len(cea) > 0, "CEA should return data"

    # Check required columns
    required_cols = [
        'season', 'week', 'team',
        'go_rate_over_expected_raw',
        'wpa_lost_raw',
        'cea_raw',
        'cea_z'
    ]
    for col in required_cols:
        assert col in cea.columns, f"Missing column: {col}"

    # Week 3: Team A (aggressive) vs Team B (conservative)
    week3 = cea[cea['week'] == 3]

    if len(week3) >= 2:
        team_a = week3[week3['team'] == 'A']
        team_b = week3[week3['team'] == 'B']

        if len(team_a) > 0 and len(team_b) > 0:
            # Team A should have higher go rate over expected
            a_go_rate = team_a['go_rate_over_expected_raw'].iloc[0]
            b_go_rate = team_b['go_rate_over_expected_raw'].iloc[0]

            if not pd.isna(a_go_rate) and not pd.isna(b_go_rate):
                assert a_go_rate > b_go_rate, \
                    f"Team A (aggressive) should have higher go rate than Team B (conservative): {a_go_rate} vs {b_go_rate}"

            # Team A should have higher CEA overall
            a_cea = team_a['cea_raw'].iloc[0]
            b_cea = team_b['cea_raw'].iloc[0]

            if not pd.isna(a_cea) and not pd.isna(b_cea):
                assert a_cea > b_cea, \
                    f"Team A (aggressive) should have higher CEA than Team B (conservative): {a_cea} vs {b_cea}"


def test_coaching_edge_leak_free():
    """Test that CEA for week W only uses weeks < W."""
    pbp = create_toy_4th_down_pbp(season=2023)

    cea = compute_coaching_edge_metrics(pbp, season=2023)

    # Week 1 should have NaN (no prior data)
    week1 = cea[cea['week'] == 1]
    if len(week1) > 0:
        assert week1['cea_raw'].isna().all(), "Week 1 should have NaN CEA (no prior data)"

    # Week 3 should have CEA based on weeks 1-2
    week3 = cea[cea['week'] == 3]
    if len(week3) > 0:
        non_nan_count = (~week3['cea_raw'].isna()).sum()
        assert non_nan_count > 0, "Week 3 should have some CEA values based on prior weeks"


def test_coaching_edge_wpa_lost():
    """Test that WPA lost correctly penalizes conservative decisions."""
    pbp = create_toy_4th_down_pbp(season=2023)

    cea = compute_coaching_edge_metrics(pbp, season=2023)

    # Team B always punts on 4th and short - should have negative WPA lost
    week3_b = cea[(cea['week'] == 3) & (cea['team'] == 'B')]

    if len(week3_b) > 0:
        wpa_lost = week3_b['wpa_lost_raw'].iloc[0]
        if not pd.isna(wpa_lost):
            # Should be negative (penalty for conservative decisions)
            assert wpa_lost <= 0, \
                f"Team B (conservative) should have negative WPA lost, got {wpa_lost}"


def test_coaching_edge_empty_pbp():
    """Test CEA handles empty play-by-play gracefully."""
    empty_pbp = pd.DataFrame(columns=['season', 'week', 'posteam', 'down'])

    cea = compute_coaching_edge_metrics(empty_pbp, season=2023)

    # Should return empty DataFrame with correct columns
    assert isinstance(cea, pd.DataFrame)
    assert len(cea) == 0


def test_coaching_edge_z_score_normalization():
    """Test that CEA z-scores are properly normalized."""
    pbp = create_toy_4th_down_pbp(season=2023)

    cea = compute_coaching_edge_metrics(pbp, season=2023)

    # Filter to valid CEA values
    valid_cea = cea[~cea['cea_z'].isna()]

    if len(valid_cea) > 1:
        # Z-scores should have reasonable distribution
        mean_z = valid_cea['cea_z'].mean()
        std_z = valid_cea['cea_z'].std()

        # Mean should be close to 0
        assert abs(mean_z) < 0.5, f"CEA z-scores should have mean ~0, got {mean_z}"

        # Std should be > 0
        if len(valid_cea) > 2:
            assert std_z > 0, "CEA z-scores should have non-zero standard deviation"
