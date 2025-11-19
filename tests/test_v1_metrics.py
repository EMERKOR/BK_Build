"""
Unit tests for Ball Knower v1.0 metrics helper.

Tests the core metrics computation logic with synthetic data.
"""

import pytest
import pandas as pd
import numpy as np

from ball_knower.benchmarks.v1_metrics import (
    compute_v1_0_metrics,
    compute_v1_0_errors
)


class TestComputeV1Metrics:
    """Test suite for compute_v1_0_metrics function."""

    def test_basic_metrics_computation(self):
        """Test basic MAE and mean error calculation with known values."""
        # Create synthetic data with known errors
        df = pd.DataFrame({
            'model_spread': [-3.0, -6.0, 2.0, 5.0],
            'market_spread': [-4.0, -5.0, 3.0, 6.0],
            'actual_margin': [-5.0, -8.0, 0.0, 4.0]
        })

        # Expected calculations:
        # model_error = model_spread - actual_margin
        #   Game 1: -3.0 - (-5.0) = 2.0
        #   Game 2: -6.0 - (-8.0) = 2.0
        #   Game 3: 2.0 - 0.0 = 2.0
        #   Game 4: 5.0 - 4.0 = 1.0
        # model_mae = mean([2.0, 2.0, 2.0, 1.0]) = 1.75
        # model_mean_error = mean([2.0, 2.0, 2.0, 1.0]) = 1.75

        # market_error = market_spread - actual_margin
        #   Game 1: -4.0 - (-5.0) = 1.0
        #   Game 2: -5.0 - (-8.0) = 3.0
        #   Game 3: 3.0 - 0.0 = 3.0
        #   Game 4: 6.0 - 4.0 = 2.0
        # market_mae = mean([1.0, 3.0, 3.0, 2.0]) = 2.25
        # market_mean_error = mean([1.0, 3.0, 3.0, 2.0]) = 2.25

        metrics = compute_v1_0_metrics(df)

        assert metrics['n_games'] == 4
        assert metrics['model_mae'] == pytest.approx(1.75, abs=0.01)
        assert metrics['market_mae'] == pytest.approx(2.25, abs=0.01)
        assert metrics['model_mean_error'] == pytest.approx(1.75, abs=0.01)
        assert metrics['market_mean_error'] == pytest.approx(2.25, abs=0.01)

    def test_mae_improvement_calculation(self):
        """Test that MAE improvement is correctly computed."""
        df = pd.DataFrame({
            'model_spread': [-3.0, -6.0, 2.0],
            'market_spread': [-4.0, -5.0, 3.0],
            'actual_margin': [-5.0, -8.0, 0.0]
        })

        metrics = compute_v1_0_metrics(df)

        # model_mae_improvement = market_mae - model_mae
        expected_improvement = metrics['market_mae'] - metrics['model_mae']
        assert metrics['model_mae_improvement'] == pytest.approx(expected_improvement, abs=0.01)

    def test_bias_detection(self):
        """Test mean error correctly identifies prediction bias."""
        # Create data where model consistently over-predicts
        df = pd.DataFrame({
            'model_spread': [0.0, 1.0, 2.0, 3.0],
            'market_spread': [0.0, 0.0, 0.0, 0.0],
            'actual_margin': [-2.0, -2.0, -2.0, -2.0]
        })

        metrics = compute_v1_0_metrics(df)

        # model_error = model_spread - actual_margin
        #   All games: [0-(-2), 1-(-2), 2-(-2), 3-(-2)] = [2, 3, 4, 5]
        # Mean = 3.5 (positive bias = predicting too high)
        assert metrics['model_mean_error'] == pytest.approx(3.5, abs=0.01)

    def test_percentile_computation(self):
        """Test error percentile calculations."""
        # Create data with known distribution
        df = pd.DataFrame({
            'model_spread': [0.0] * 10,
            'market_spread': [0.0] * 10,
            'actual_margin': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        })

        metrics = compute_v1_0_metrics(df)

        # model_error (absolute) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # 50th percentile = 4.5
        # 75th percentile = 6.75
        # 90th percentile = 8.1
        assert metrics['model_error_pct_50'] == pytest.approx(4.5, abs=0.1)
        assert metrics['model_error_pct_75'] == pytest.approx(6.75, abs=0.1)
        assert metrics['model_error_pct_90'] == pytest.approx(8.1, abs=0.1)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error."""
        df = pd.DataFrame({
            'model_spread': [],
            'market_spread': [],
            'actual_margin': []
        })

        with pytest.raises(ValueError, match="No valid rows"):
            compute_v1_0_metrics(df)

    def test_missing_values_are_dropped(self):
        """Test that rows with NaN values are excluded."""
        df = pd.DataFrame({
            'model_spread': [-3.0, -6.0, np.nan, 5.0],
            'market_spread': [-4.0, -5.0, 3.0, 6.0],
            'actual_margin': [-5.0, -8.0, 0.0, 4.0]
        })

        metrics = compute_v1_0_metrics(df)

        # Should only count 3 games (row with NaN is dropped)
        assert metrics['n_games'] == 3


class TestComputeV1Errors:
    """Test suite for compute_v1_0_errors function."""

    def test_error_columns_added(self):
        """Test that error columns are correctly added to DataFrame."""
        df = pd.DataFrame({
            'game_id': ['2024_01_KC_BUF', '2024_01_SF_DAL'],
            'model_spread': [-3.0, -6.0],
            'market_spread': [-4.0, -5.0],
            'actual_margin': [-5.0, -8.0]
        })

        df_with_errors = compute_v1_0_errors(df)

        # Check that new columns exist
        assert 'model_error' in df_with_errors.columns
        assert 'market_error' in df_with_errors.columns
        assert 'abs_model_error' in df_with_errors.columns
        assert 'abs_market_error' in df_with_errors.columns

        # Check values
        # Game 1: model_error = -3.0 - (-5.0) = 2.0
        assert df_with_errors.loc[0, 'model_error'] == pytest.approx(2.0, abs=0.01)
        # Game 1: market_error = -4.0 - (-5.0) = 1.0
        assert df_with_errors.loc[0, 'market_error'] == pytest.approx(1.0, abs=0.01)

        # Check absolute values
        assert df_with_errors.loc[0, 'abs_model_error'] == pytest.approx(2.0, abs=0.01)
        assert df_with_errors.loc[0, 'abs_market_error'] == pytest.approx(1.0, abs=0.01)

    def test_original_dataframe_unchanged(self):
        """Test that original DataFrame is not modified in-place."""
        df = pd.DataFrame({
            'model_spread': [-3.0],
            'market_spread': [-4.0],
            'actual_margin': [-5.0]
        })

        original_columns = set(df.columns)
        df_with_errors = compute_v1_0_errors(df)

        # Original should be unchanged
        assert set(df.columns) == original_columns
        # New DataFrame should have additional columns
        assert len(df_with_errors.columns) > len(df.columns)
