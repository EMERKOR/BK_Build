"""
Test src.features module

Ensures feature engineering functions produce expected outputs
and maintain consistent behavior across refactorings.
"""

import pytest
import pandas as pd
import numpy as np

from src import features


# ============================================================================
# TEST NFelo REST ADVANTAGE HELPER
# ============================================================================

def test_add_nfelo_rest_advantage_basic():
    """
    Test that add_nfelo_rest_advantage() correctly computes rest_advantage
    using the canonical NFelo formula: home_bye_mod + away_bye_mod.
    """
    # Create test data with known bye modifiers
    test_data = pd.DataFrame({
        'game_id': ['2023_01_KC_DET', '2023_02_BUF_MIA', '2023_03_SF_DAL'],
        'home_bye_mod': [0.0, 1.5, -1.0],
        'away_bye_mod': [0.0, -1.5, 2.0],
    })

    # Apply the helper
    result = features.add_nfelo_rest_advantage(test_data)

    # Verify rest_advantage column exists
    assert 'rest_advantage' in result.columns, \
        "add_nfelo_rest_advantage() should add 'rest_advantage' column"

    # Verify correct calculation (home + away)
    expected = test_data['home_bye_mod'] + test_data['away_bye_mod']
    pd.testing.assert_series_equal(
        result['rest_advantage'],
        expected,
        check_names=False
    )

    # Verify specific values
    assert result.loc[0, 'rest_advantage'] == 0.0, "Game 1: 0.0 + 0.0 = 0.0"
    assert result.loc[1, 'rest_advantage'] == 0.0, "Game 2: 1.5 + (-1.5) = 0.0"
    assert result.loc[2, 'rest_advantage'] == 1.0, "Game 3: -1.0 + 2.0 = 1.0"


def test_add_nfelo_rest_advantage_handles_na():
    """
    Test that add_nfelo_rest_advantage() treats NaN bye modifiers as 0.

    This matches the existing behavior in v1_2 dataset builder and run_backtests.py.
    """
    # Create test data with NaN values
    test_data = pd.DataFrame({
        'game_id': ['2023_01_KC_DET', '2023_02_BUF_MIA', '2023_03_SF_DAL'],
        'home_bye_mod': [np.nan, 1.5, np.nan],
        'away_bye_mod': [2.0, np.nan, np.nan],
    })

    # Apply the helper
    result = features.add_nfelo_rest_advantage(test_data)

    # Verify NaN handling (fillna(0) behavior)
    assert result.loc[0, 'rest_advantage'] == 2.0, "Game 1: NaN + 2.0 = 0.0 + 2.0 = 2.0"
    assert result.loc[1, 'rest_advantage'] == 1.5, "Game 2: 1.5 + NaN = 1.5 + 0.0 = 1.5"
    assert result.loc[2, 'rest_advantage'] == 0.0, "Game 3: NaN + NaN = 0.0 + 0.0 = 0.0"


def test_add_nfelo_rest_advantage_returns_copy():
    """
    Test that add_nfelo_rest_advantage() returns a copy and doesn't modify original.
    """
    # Create test data
    test_data = pd.DataFrame({
        'game_id': ['2023_01_KC_DET'],
        'home_bye_mod': [1.0],
        'away_bye_mod': [2.0],
    })

    # Store original shape
    original_columns = test_data.columns.tolist()

    # Apply the helper
    result = features.add_nfelo_rest_advantage(test_data)

    # Verify original wasn't modified
    assert 'rest_advantage' not in test_data.columns, \
        "Original DataFrame should not be modified"
    assert test_data.columns.tolist() == original_columns, \
        "Original DataFrame columns should be unchanged"

    # Verify result is different object
    assert result is not test_data, "Should return a copy, not modify in place"


def test_add_nfelo_rest_advantage_preserves_other_columns():
    """
    Test that add_nfelo_rest_advantage() preserves all other columns.
    """
    # Create test data with extra columns
    test_data = pd.DataFrame({
        'game_id': ['2023_01_KC_DET'],
        'season': [2023],
        'week': [1],
        'home_bye_mod': [1.0],
        'away_bye_mod': [2.0],
        'nfelo_diff': [50.0],
    })

    # Apply the helper
    result = features.add_nfelo_rest_advantage(test_data)

    # Verify all original columns are preserved
    for col in test_data.columns:
        assert col in result.columns, f"Column '{col}' should be preserved"

    # Verify data in other columns is unchanged
    assert result['game_id'].iloc[0] == '2023_01_KC_DET'
    assert result['season'].iloc[0] == 2023
    assert result['week'].iloc[0] == 1
    assert result['nfelo_diff'].iloc[0] == 50.0


def test_add_nfelo_rest_advantage_can_overwrite():
    """
    Test that add_nfelo_rest_advantage() can overwrite existing rest_advantage column.

    This is useful when re-processing data or fixing values.
    """
    # Create test data with existing (incorrect) rest_advantage
    test_data = pd.DataFrame({
        'game_id': ['2023_01_KC_DET'],
        'home_bye_mod': [1.0],
        'away_bye_mod': [2.0],
        'rest_advantage': [999.0],  # Wrong value
    })

    # Apply the helper
    result = features.add_nfelo_rest_advantage(test_data)

    # Verify rest_advantage was overwritten with correct value
    assert result['rest_advantage'].iloc[0] == 3.0, \
        "Should overwrite existing rest_advantage with correct calculation (1.0 + 2.0 = 3.0)"

    # Verify original still has wrong value (not modified in place)
    assert test_data['rest_advantage'].iloc[0] == 999.0, \
        "Original DataFrame should not be modified"
