"""
Test ball_knower.evaluation.calibration_v1_3

Unit tests for v1.3 model calibration utilities using synthetic data.
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.evaluation import calibration_v1_3


# ============================================================================
# TEST compute_v1_3_calibration (season-aggregated)
# ============================================================================

def test_compute_v1_3_calibration_with_synthetic_data():
    """
    Test compute_v1_3_calibration with synthetic season-aggregated backtest results.
    """
    # Create synthetic backtest results (season-aggregated)
    backtest_df = pd.DataFrame({
        'season': [2019, 2020, 2021],
        'model': ['v1.3', 'v1.3', 'v1.3'],
        'edge_threshold': [0.0, 0.0, 0.0],
        'n_games': [256, 256, 272],
        'n_bets': [256, 256, 272],
        'mae_vs_vegas': [1.5, 1.6, 1.4],
        'rmse_vs_vegas': [2.0, 2.1, 1.9],
        'mean_edge': [0.1, -0.05, 0.15],
    })

    # Compute calibration
    calibration = calibration_v1_3.compute_v1_3_calibration(
        backtest_df,
        edge_bins=[0.5, 1.0, 1.5, 2.0]
    )

    # Assert structure
    assert 'mean_error' in calibration
    assert 'mae' in calibration
    assert 'rmse' in calibration
    assert 'edge_bins' in calibration
    assert 'n_seasons' in calibration
    assert 'n_games_total' in calibration
    assert 'calibration_seasons' in calibration
    assert 'model_version' in calibration

    # Assert values
    assert calibration['n_seasons'] == 3
    assert calibration['n_games_total'] == 784  # 256 + 256 + 272
    assert calibration['model_version'] == 'v1.3'
    assert calibration['edge_bins'] == [0.5, 1.0, 1.5, 2.0]
    assert calibration['calibration_seasons'] == '2019-2021'

    # Assert mean error (average of mean_edge)
    expected_mean_error = (0.1 + (-0.05) + 0.15) / 3
    assert abs(calibration['mean_error'] - expected_mean_error) < 0.01

    # Assert weighted MAE
    # (1.5*256 + 1.6*256 + 1.4*272) / 784 = (384 + 409.6 + 380.8) / 784 = 1.496
    expected_mae = (1.5*256 + 1.6*256 + 1.4*272) / 784
    assert abs(calibration['mae'] - expected_mae) < 0.01

    # Assert RMSE is positive and reasonable
    assert calibration['rmse'] > 0
    assert calibration['rmse'] < 5.0


def test_compute_v1_3_calibration_with_single_season():
    """
    Test compute_v1_3_calibration with a single season.
    """
    backtest_df = pd.DataFrame({
        'season': [2019],
        'model': ['v1.3'],
        'edge_threshold': [0.0],
        'n_games': [256],
        'n_bets': [256],
        'mae_vs_vegas': [1.5],
        'rmse_vs_vegas': [2.0],
        'mean_edge': [0.1],
    })

    calibration = calibration_v1_3.compute_v1_3_calibration(backtest_df)

    assert calibration['n_seasons'] == 1
    assert calibration['n_games_total'] == 256
    assert calibration['mean_error'] == 0.1
    assert calibration['mae'] == 1.5
    assert abs(calibration['rmse'] - 2.0) < 0.01


def test_compute_v1_3_calibration_missing_columns():
    """
    Test that compute_v1_3_calibration raises ValueError when required columns are missing.
    """
    # Missing 'mean_edge' column
    backtest_df = pd.DataFrame({
        'season': [2019],
        'model': ['v1.3'],
        'n_games': [256],
        'mae_vs_vegas': [1.5],
        'rmse_vs_vegas': [2.0],
    })

    with pytest.raises(ValueError, match="missing required columns"):
        calibration_v1_3.compute_v1_3_calibration(backtest_df)


def test_compute_v1_3_calibration_no_v1_3_data():
    """
    Test that compute_v1_3_calibration raises ValueError when no v1.3 results are present.
    """
    backtest_df = pd.DataFrame({
        'season': [2019],
        'model': ['v1.2'],  # Wrong model
        'edge_threshold': [0.0],
        'n_games': [256],
        'n_bets': [256],
        'mae_vs_vegas': [1.5],
        'rmse_vs_vegas': [2.0],
        'mean_edge': [0.1],
    })

    with pytest.raises(ValueError, match="No v1.3 model results found"):
        calibration_v1_3.compute_v1_3_calibration(backtest_df)


# ============================================================================
# TEST compute_game_level_calibration
# ============================================================================

def test_compute_game_level_calibration_with_synthetic_data():
    """
    Test compute_game_level_calibration with synthetic game-level predictions.
    """
    # Create synthetic game-level predictions
    np.random.seed(42)
    n_games = 100

    # Generate predictions with a slight bias
    vegas_line = np.random.randn(n_games) * 5
    bk_v1_3_spread = vegas_line + np.random.randn(n_games) * 2 + 0.5  # +0.5 bias
    edge = bk_v1_3_spread - vegas_line
    actual_margin = vegas_line + np.random.randn(n_games) * 10

    game_predictions_df = pd.DataFrame({
        'game_id': [f'2019_10_TEAM1_TEAM2_{i}' for i in range(n_games)],
        'season': [2019] * n_games,
        'week': [10] * n_games,
        'bk_v1_3_spread': bk_v1_3_spread,
        'vegas_line': vegas_line,
        'edge': edge,
        'actual_margin': actual_margin,
    })

    # Compute game-level calibration
    calibration = calibration_v1_3.compute_game_level_calibration(
        game_predictions_df,
        edge_bins=[0.5, 1.0, 1.5, 2.0]
    )

    # Assert structure
    assert 'mean_error' in calibration
    assert 'slope' in calibration
    assert 'intercept' in calibration
    assert 'edge_bins' in calibration
    assert 'ats_win_rates' in calibration
    assert 'n_games_per_bin' in calibration
    assert 'n_games_total' in calibration
    assert 'model_version' in calibration

    # Assert values
    assert calibration['n_games_total'] == 100
    assert calibration['model_version'] == 'v1.3'
    assert len(calibration['ats_win_rates']) == len(calibration['edge_bins']) + 1  # One bin per threshold + last bin
    assert len(calibration['n_games_per_bin']) == len(calibration['edge_bins']) + 1

    # Assert mean error is positive (we introduced +0.5 bias)
    assert calibration['mean_error'] > 0
    assert calibration['mean_error'] < 2.0  # Reasonable range

    # Assert slope is close to 1 (predictions should track Vegas)
    assert 0.5 < calibration['slope'] < 1.5

    # Assert ATS win rates are between 0 and 1
    for win_rate in calibration['ats_win_rates']:
        if win_rate is not None:  # Some bins might be empty
            assert 0.0 <= win_rate <= 1.0


def test_compute_game_level_calibration_without_actual_margins():
    """
    Test that compute_game_level_calibration handles missing actual_margin column.
    """
    # Create synthetic predictions without actual margins
    game_predictions_df = pd.DataFrame({
        'game_id': ['game1', 'game2', 'game3'],
        'bk_v1_3_spread': [-3.0, 2.5, -7.0],
        'vegas_line': [-3.5, 2.0, -6.5],
        'edge': [0.5, 0.5, -0.5],
    })

    calibration = calibration_v1_3.compute_game_level_calibration(
        game_predictions_df,
        edge_bins=[1.0]
    )

    # ATS win rates should be None when actual_margin is missing
    assert all(rate is None for rate in calibration['ats_win_rates'])

    # But other metrics should still be computed
    assert calibration['mean_error'] is not None
    assert calibration['slope'] is not None
    assert calibration['intercept'] is not None


def test_compute_game_level_calibration_missing_required_columns():
    """
    Test that compute_game_level_calibration raises ValueError when required columns are missing.
    """
    # Missing 'edge' column
    game_predictions_df = pd.DataFrame({
        'game_id': ['game1'],
        'bk_v1_3_spread': [-3.0],
        'vegas_line': [-3.5],
    })

    with pytest.raises(ValueError, match="missing required columns"):
        calibration_v1_3.compute_game_level_calibration(game_predictions_df)


# ============================================================================
# TEST apply_bias_correction
# ============================================================================

def test_apply_bias_correction():
    """
    Test that apply_bias_correction properly adjusts predictions.
    """
    predictions_df = pd.DataFrame({
        'game_id': ['game1', 'game2', 'game3'],
        'bk_v1_3_spread': [-3.0, 2.5, -7.0],
    })

    mean_error = 0.5

    corrected_df = calibration_v1_3.apply_bias_correction(predictions_df, mean_error)

    # Assert corrected column was added
    assert 'bk_v1_3_spread_corrected' in corrected_df.columns

    # Assert correction is applied correctly (subtract mean_error)
    expected_corrected = [-3.5, 2.0, -7.5]
    assert np.allclose(corrected_df['bk_v1_3_spread_corrected'].values, expected_corrected)


def test_apply_bias_correction_missing_column():
    """
    Test that apply_bias_correction raises ValueError when required column is missing.
    """
    predictions_df = pd.DataFrame({
        'game_id': ['game1'],
    })

    with pytest.raises(ValueError, match="must contain 'bk_v1_3_spread' column"):
        calibration_v1_3.apply_bias_correction(predictions_df, mean_error=0.5)
