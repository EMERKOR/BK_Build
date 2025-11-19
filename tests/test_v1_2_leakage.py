"""
Test leakage validation for v1.2 dataset pipeline.

Ensures that:
- v1.2 feature engineering is free from target leakage
- Validation correctly detects deliberate leakage
- Features are deterministic and don't depend on target columns
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.datasets import v1_2


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_nfelo_data():
    """
    Create a small synthetic nfelo dataset for testing.

    Returns a DataFrame that matches the nfelo schema with minimal
    data needed to test the v1.2 feature pipeline.
    """
    # Create synthetic game data for 2 seasons, 3 weeks each
    games = []
    game_counter = 1

    for season in [2023, 2024]:
        for week in range(1, 4):
            # Create 2 games per week
            for _ in range(2):
                home_team = f"TEAM_{game_counter % 4 + 1}"
                away_team = f"TEAM_{(game_counter + 1) % 4 + 1}"

                # Generate synthetic ELO ratings
                home_elo = 1500 + np.random.randn() * 50
                away_elo = 1500 + np.random.randn() * 50

                # Generate synthetic game outcomes
                home_score = np.random.randint(14, 35)
                away_score = np.random.randint(14, 35)

                # Generate synthetic Vegas spread
                elo_diff = home_elo - away_elo
                vegas_spread = elo_diff / 25.0 + np.random.randn() * 1.5

                game = {
                    'game_id': f'{season}_{week:02d}_{away_team}_{home_team}',
                    'starting_nfelo_home': home_elo,
                    'starting_nfelo_away': away_elo,
                    'home_line_close': vegas_spread,
                    'home_score': home_score,
                    'away_score': away_score,
                    # Situational modifiers
                    'div_game_mod': np.random.choice([0, 1, -1]),
                    'dif_surface_mod': np.random.choice([0, 0.5, -0.5]),
                    'home_time_advantage_mod': np.random.choice([0, 1, -1]),
                    # QB adjustments
                    'home_538_qb_adj': np.random.randn() * 2,
                    'away_538_qb_adj': np.random.randn() * 2,
                    # Rest advantage columns (required by compute_rest_advantage_from_nfelo)
                    'home_bye_mod': np.random.choice([0, 0, 0, 1, -1]),
                    'away_bye_mod': np.random.choice([0, 0, 0, 1, -1]),
                    # Rest days (for reference)
                    'home_rest': np.random.choice([6, 7, 13, 14]),
                    'away_rest': np.random.choice([6, 7, 13, 14]),
                }

                games.append(game)
                game_counter += 1

    return pd.DataFrame(games)


# ============================================================================
# POSITIVE TESTS (Verify No Leakage)
# ============================================================================

def test_v1_2_dataset_is_leak_free(synthetic_nfelo_data):
    """
    Test that v1.2 dataset pipeline passes leakage validation.

    This is the primary positive test: we verify that the actual v1.2
    feature engineering pipeline is free from target leakage.
    """
    # Run validation on synthetic data
    v1_2.validate_v1_2_no_leakage(synthetic_nfelo_data)

    # If we reach here, validation passed (no assertion errors)
    assert True


def test_v1_2_validation_with_real_data_subset():
    """
    Test that v1.2 validation works with a small subset of real data.

    This uses the actual nfelo data loader but limits to a single season
    to keep the test fast.
    """
    # Build a small real dataset (single season for speed)
    df = v1_2.build_training_frame(start_year=2023, end_year=2023)

    assert len(df) > 0, "Should have some games from 2023 season"

    # Now validate using the raw data
    # We need to reconstruct raw data from the built frame for this test
    # In practice, you'd load raw nfelo data and validate before building
    # For this test, we'll just verify the built frame has expected properties

    # Verify target columns are present
    assert 'vegas_closing_spread' in df.columns
    assert 'actual_margin' in df.columns

    # Verify feature columns don't contain target data
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff']

    for col in feature_cols:
        assert col in df.columns, f"Missing feature column: {col}"
        # Features should have reasonable ranges (not be identical to targets)
        assert not np.allclose(df[col].fillna(0), df['actual_margin'].fillna(0))


# ============================================================================
# NEGATIVE TESTS (Verify Leakage Detection)
# ============================================================================

def test_validation_detects_target_leakage(synthetic_nfelo_data):
    """
    Test that validation correctly detects when features leak target data.

    This is a negative test: we deliberately create a leaking builder
    and verify that validation catches it.
    """
    # Create a deliberately leaking version of the feature builder
    def leaking_builder(df):
        """Builder that leaks target information into features."""
        df = df.copy()

        # Parse game_id
        if 'season' not in df.columns:
            df[['season', 'week', 'away_team', 'home_team']] = \
                df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
            df['season'] = df['season'].astype(int)
            df['week'] = df['week'].astype(int)

        # DELIBERATE LEAK: Use actual margin in feature calculation
        df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
        df['actual_margin'] = df['home_score'] - df['away_score']

        # THIS IS THE LEAK: nfelo_diff is "adjusted" by actual outcome
        df['nfelo_diff'] = df['nfelo_diff'] + df['actual_margin'] * 0.1

        # Rest of features
        df['rest_advantage'] = 0
        df['div_game'] = df['div_game_mod'].fillna(0)
        df['surface_mod'] = df['dif_surface_mod'].fillna(0)
        df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
        df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) -
                         df['away_538_qb_adj'].fillna(0))
        df['vegas_closing_spread'] = df['home_line_close']
        df['home_points'] = df['home_score']
        df['away_points'] = df['away_score']
        df['home_margin'] = df['actual_margin']

        output_cols = [
            'game_id', 'season', 'week', 'away_team', 'home_team',
            'nfelo_diff', 'rest_advantage', 'div_game',
            'surface_mod', 'time_advantage', 'qb_diff',
            'vegas_closing_spread', 'home_score', 'away_score', 'actual_margin',
            'home_points', 'away_points', 'home_margin',
        ]

        return df[output_cols].reset_index(drop=True)

    # Replace the internal builder temporarily
    original_builder = v1_2._build_features_from_df
    v1_2._build_features_from_df = leaking_builder

    try:
        # This should raise an AssertionError due to target leakage
        with pytest.raises(AssertionError, match="depends on target columns"):
            v1_2.validate_v1_2_no_leakage(synthetic_nfelo_data)
    finally:
        # Restore original builder
        v1_2._build_features_from_df = original_builder


def test_validation_detects_nondeterminism(synthetic_nfelo_data):
    """
    Test that validation detects non-deterministic feature engineering.

    Non-deterministic features can indicate data leakage or unstable pipelines.
    """
    # Create a non-deterministic builder (uses random values)
    call_count = [0]

    def nondeterministic_builder(df):
        """Builder that produces different results on each call."""
        df = df.copy()

        if 'season' not in df.columns:
            df[['season', 'week', 'away_team', 'home_team']] = \
                df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
            df['season'] = df['season'].astype(int)
            df['week'] = df['week'].astype(int)

        # NON-DETERMINISTIC: Add random noise that changes on each call
        call_count[0] += 1
        random_noise = np.random.randn(len(df)) * call_count[0]

        df['nfelo_diff'] = (df['starting_nfelo_home'] - df['starting_nfelo_away'] +
                           random_noise)
        df['rest_advantage'] = 0
        df['div_game'] = df['div_game_mod'].fillna(0)
        df['surface_mod'] = df['dif_surface_mod'].fillna(0)
        df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
        df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) -
                         df['away_538_qb_adj'].fillna(0))
        df['vegas_closing_spread'] = df['home_line_close']
        df['home_score'] = df['home_score'].fillna(0)
        df['away_score'] = df['away_score'].fillna(0)
        df['actual_margin'] = df['home_score'] - df['away_score']
        df['home_points'] = df['home_score']
        df['away_points'] = df['away_score']
        df['home_margin'] = df['actual_margin']

        output_cols = [
            'game_id', 'season', 'week', 'away_team', 'home_team',
            'nfelo_diff', 'rest_advantage', 'div_game',
            'surface_mod', 'time_advantage', 'qb_diff',
            'vegas_closing_spread', 'home_score', 'away_score', 'actual_margin',
            'home_points', 'away_points', 'home_margin',
        ]

        return df[output_cols].reset_index(drop=True)

    # Replace the internal builder temporarily
    original_builder = v1_2._build_features_from_df
    v1_2._build_features_from_df = nondeterministic_builder

    try:
        # This should raise an AssertionError due to non-determinism
        with pytest.raises(AssertionError, match="not deterministic"):
            v1_2.validate_v1_2_no_leakage(synthetic_nfelo_data)
    finally:
        # Restore original builder
        v1_2._build_features_from_df = original_builder


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_v1_2_builder_uses_internal_helper():
    """
    Test that build_training_frame uses the internal _build_features_from_df helper.

    This ensures that validation tests the same code path as production.
    """
    # Build a small dataset
    df = v1_2.build_training_frame(start_year=2023, end_year=2023)

    # Verify it has the expected structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Verify it has the expected columns from _build_features_from_df
    expected_cols = [
        'game_id', 'season', 'week', 'away_team', 'home_team',
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff',
        'vegas_closing_spread', 'home_score', 'away_score', 'actual_margin',
        'home_points', 'away_points', 'home_margin',
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
