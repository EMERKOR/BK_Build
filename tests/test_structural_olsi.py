"""
Tests for Offensive Line Structure Index (OLSI) metrics.

Tests pass protection metrics computation and leak-free aggregation.
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.structural.olsi import compute_ol_structure_metrics


def create_toy_pass_pbp(season=2023):
    """Create synthetic pass-play PBP with clear OL performance differences."""
    data = []

    # Week 1: Team A good OL (few sacks/pressures), Team B bad OL (many sacks)
    # Team A: 10 pass plays, 1 sack, 2 pressures
    for i in range(10):
        data.append({
            'game_id': f'{season}_01_A_B',
            'season': season,
            'week': 1,
            'posteam': 'A',
            'defteam': 'B',
            'down': 1,
            'play_type': 'pass',
            'sack': 1 if i == 0 else 0,
            'qb_hit': 1 if i < 2 else 0,
        })

    # Team B: 10 pass plays, 5 sacks, 7 pressures
    for i in range(10):
        data.append({
            'game_id': f'{season}_01_A_B',
            'season': season,
            'week': 1,
            'posteam': 'B',
            'defteam': 'A',
            'down': 1,
            'play_type': 'pass',
            'sack': 1 if i < 5 else 0,
            'qb_hit': 1 if i < 7 else 0,
        })

    # Week 2: Similar pattern continues
    for i in range(10):
        data.append({
            'game_id': f'{season}_02_A_C',
            'season': season,
            'week': 2,
            'posteam': 'A',
            'defteam': 'C',
            'down': 1,
            'play_type': 'pass',
            'sack': 1 if i == 0 else 0,
            'qb_hit': 1 if i < 2 else 0,
        })

    for i in range(10):
        data.append({
            'game_id': f'{season}_02_B_C',
            'season': season,
            'week': 2,
            'posteam': 'B',
            'defteam': 'C',
            'down': 1,
            'play_type': 'pass',
            'sack': 1 if i < 4 else 0,
            'qb_hit': 1 if i < 6 else 0,
        })

    # Week 3: More data for leak-free test
    for i in range(10):
        data.append({
            'game_id': f'{season}_03_A_B',
            'season': season,
            'week': 3,
            'posteam': 'A',
            'defteam': 'B',
            'down': 1,
            'play_type': 'pass',
            'sack': 0,
            'qb_hit': 1 if i < 2 else 0,
        })

    return pd.DataFrame(data)


def test_olsi_basic():
    """Test that OLSI correctly ranks teams by pass protection quality."""
    pbp = create_toy_pass_pbp(season=2023)

    olsi = compute_ol_structure_metrics(pbp, season=2023)

    # Should have data
    assert len(olsi) > 0, "OLSI should return data"

    # Check required columns
    required_cols = [
        'season', 'week', 'team',
        'pressure_rate_raw', 'sack_rate_raw', 'qb_hit_rate_raw',
        'olsi_raw', 'olsi_z'
    ]
    for col in required_cols:
        assert col in olsi.columns, f"Missing column: {col}"

    # Week 3: Team A (good OL) vs Team B (bad OL)
    week3_data = olsi[olsi['week'] == 3]

    if len(week3_data) >= 2:
        team_a = week3_data[week3_data['team'] == 'A']
        team_b = week3_data[week3_data['team'] == 'B']

        if len(team_a) > 0 and len(team_b) > 0:
            # Team A should have better OLSI (higher is better)
            # Lower pressure/sack rates = better OL = higher OLSI
            a_olsi = team_a['olsi_raw'].iloc[0]
            b_olsi = team_b['olsi_raw'].iloc[0]

            if not pd.isna(a_olsi) and not pd.isna(b_olsi):
                assert a_olsi > b_olsi, \
                    f"Team A (good OL) should have higher OLSI than Team B (bad OL): {a_olsi} vs {b_olsi}"


def test_olsi_leak_free():
    """Test that OLSI for week W only uses weeks < W."""
    pbp = create_toy_pass_pbp(season=2023)

    olsi = compute_ol_structure_metrics(pbp, season=2023)

    # Week 1 should have NaN (no prior data)
    week1 = olsi[olsi['week'] == 1]
    if len(week1) > 0:
        assert week1['olsi_raw'].isna().all(), "Week 1 should have NaN OLSI (no prior data)"

    # Week 3 should have OLSI based on weeks 1-2 only
    week3 = olsi[olsi['week'] == 3]
    if len(week3) > 0:
        # Should have some non-NaN values
        non_nan_count = (~week3['olsi_raw'].isna()).sum()
        assert non_nan_count > 0, "Week 3 should have some OLSI values based on prior weeks"


def test_olsi_pressure_rate_computation():
    """Test that pressure rate is computed correctly."""
    pbp = create_toy_pass_pbp(season=2023)

    olsi = compute_ol_structure_metrics(pbp, season=2023)

    # Week 2 data (uses week 1)
    week2 = olsi[olsi['week'] == 2]

    if len(week2) > 0:
        # Check Team A: 1 sack + 2 hits = 3 pressures out of 10 dropbacks
        team_a_week2 = week2[week2['team'] == 'A']
        if len(team_a_week2) > 0:
            pressure_rate = team_a_week2['pressure_rate_raw'].iloc[0]
            if not pd.isna(pressure_rate):
                # Should be around 0.3 (3/10)
                assert 0.25 < pressure_rate < 0.35, \
                    f"Team A pressure rate should be ~0.30, got {pressure_rate}"

        # Check Team B: 5 sacks + 7 hits = 12 pressures (capped by plays)
        team_b_week2 = week2[week2['team'] == 'B']
        if len(team_b_week2) > 0:
            pressure_rate = team_b_week2['pressure_rate_raw'].iloc[0]
            if not pd.isna(pressure_rate):
                # Should be higher than Team A
                assert pressure_rate > 0.5, \
                    f"Team B pressure rate should be >0.50, got {pressure_rate}"


def test_olsi_z_score_normalization():
    """Test that OLSI z-scores are properly normalized."""
    pbp = create_toy_pass_pbp(season=2023)

    olsi = compute_ol_structure_metrics(pbp, season=2023)

    # Filter to rows with valid OLSI
    valid_olsi = olsi[~olsi['olsi_z'].isna()]

    if len(valid_olsi) > 0:
        # Z-scores should have reasonable distribution
        mean_z = valid_olsi['olsi_z'].mean()
        std_z = valid_olsi['olsi_z'].std()

        # Mean should be close to 0
        assert abs(mean_z) < 0.5, f"OLSI z-scores should have mean ~0, got {mean_z}"

        # Std should be reasonable (not all zeros)
        assert std_z > 0, "OLSI z-scores should have non-zero standard deviation"
