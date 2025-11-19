"""
Tests for v1.x model comparison harness.

These tests validate that the comparison utilities work correctly and
return properly structured results.
"""

import pytest
import pandas as pd
import numpy as np
from ball_knower.benchmarks.v1_comparison import (
    compare_v1_models,
    run_v1_0_backtest_on_frame,
    run_v1_2_backtest_on_frame,
    run_v1_3_backtest_on_frame,
    build_common_test_frame,
)


class TestV1Comparison:
    """Test suite for v1.x model comparison harness."""

    def test_compare_v1_models_structure(self):
        """Test that compare_v1_models returns proper structure."""
        # Use a small test period to keep test fast
        # 2018 is a good test year - far enough back to have stable data
        results = compare_v1_models(test_seasons=[2018])

        # Check top-level structure
        assert 'test_seasons' in results
        assert 'n_games' in results
        assert 'models' in results

        # Check test_seasons
        assert results['test_seasons'] == [2018]
        assert results['n_games'] > 0

        # Check models dict has all three versions
        assert 'v1.0' in results['models']
        assert 'v1.2' in results['models']
        assert 'v1.3' in results['models']

    def test_model_results_have_required_keys(self):
        """Test that each model's results have required keys."""
        results = compare_v1_models(test_seasons=[2018])

        required_keys = ['model_name', 'mae_spread', 'mae_total', 'n_games', 'status']

        for model_name in ['v1.0', 'v1.2', 'v1.3']:
            model_results = results['models'][model_name]

            # Check all required keys present
            for key in required_keys:
                assert key in model_results, f"Missing key '{key}' in {model_name} results"

    def test_successful_models_have_finite_metrics(self):
        """Test that successful models return finite MAE values."""
        results = compare_v1_models(test_seasons=[2018])

        for model_name in ['v1.0', 'v1.2', 'v1.3']:
            model_results = results['models'][model_name]

            if model_results['status'] == 'ok':
                # Spread MAE should be finite for all successful models
                assert model_results['mae_spread'] is not None
                assert np.isfinite(model_results['mae_spread'])
                assert model_results['mae_spread'] > 0  # Should have some error

                # v1.3 should also have total MAE
                if model_name == 'v1.3':
                    assert model_results['mae_total'] is not None
                    assert np.isfinite(model_results['mae_total'])
                    assert model_results['mae_total'] > 0

    def test_build_common_test_frame(self):
        """Test that common test frame has required columns."""
        df = build_common_test_frame(test_seasons=[2018])

        # Check we got data
        assert len(df) > 0

        # Check required columns for v1.0
        assert 'nfelo_diff' in df.columns
        assert 'actual_margin' in df.columns

        # Check required columns for v1.2
        v1_2_features = ['nfelo_diff', 'rest_advantage', 'div_game',
                         'surface_mod', 'time_advantage', 'qb_diff']
        for col in v1_2_features:
            assert col in df.columns

        # Check required columns for v1.3
        assert 'home_score' in df.columns
        assert 'away_score' in df.columns

        # Check game identifiers
        assert 'game_id' in df.columns
        assert 'season' in df.columns
        assert 'week' in df.columns

    def test_v1_0_backtest_basic(self):
        """Test v1.0 backtest on small sample."""
        df = build_common_test_frame(test_seasons=[2018])
        results = run_v1_0_backtest_on_frame(df)

        assert results['model_name'] == 'v1.0'
        assert results['status'] == 'ok'
        assert results['mae_spread'] is not None
        assert results['mae_total'] is None  # v1.0 doesn't predict totals
        assert results['n_games'] == len(df)

    def test_v1_2_backtest_basic(self):
        """Test v1.2 backtest on small sample."""
        df = build_common_test_frame(test_seasons=[2018])
        results = run_v1_2_backtest_on_frame(df)

        assert results['model_name'] == 'v1.2'

        # v1.2 should work if model file exists
        if results['status'] == 'ok':
            assert results['mae_spread'] is not None
            assert results['mae_total'] is None  # v1.2 doesn't predict totals
            assert results['n_games'] == len(df)

    def test_v1_3_backtest_basic(self):
        """Test v1.3 backtest on small sample."""
        # Note: v1.3 requires training, so this might take a bit
        df = build_common_test_frame(test_seasons=[2018])
        results = run_v1_3_backtest_on_frame(df)

        assert results['model_name'] == 'v1.3'

        if results['status'] == 'ok':
            # v1.3 should have both spread and total MAE
            assert results['mae_spread'] is not None
            assert results['mae_total'] is not None
            assert results['n_games'] > 0

            # v1.3 should also have score metrics
            assert 'mae_home_score' in results
            assert 'mae_away_score' in results

    def test_comparison_game_counts_match(self):
        """Test that all models see the same number of games."""
        results = compare_v1_models(test_seasons=[2018])

        n_games = results['n_games']

        # All successful models should report same game count
        for model_name in ['v1.0', 'v1.2', 'v1.3']:
            model_results = results['models'][model_name]
            if model_results['status'] == 'ok':
                assert model_results['n_games'] == n_games

    def test_hit_rates_are_percentages(self):
        """Test that hit rates are in valid percentage range."""
        results = compare_v1_models(test_seasons=[2018])

        for model_name in ['v1.0', 'v1.2', 'v1.3']:
            model_results = results['models'][model_name]

            if model_results['status'] == 'ok':
                # Check hit rates if present
                if 'hit_rate_spread_within_3' in model_results:
                    hit_3 = model_results['hit_rate_spread_within_3']
                    assert 0 <= hit_3 <= 100

                if 'hit_rate_spread_within_7' in model_results:
                    hit_7 = model_results['hit_rate_spread_within_7']
                    assert 0 <= hit_7 <= 100

                    # ±7 should be >= ±3
                    if 'hit_rate_spread_within_3' in model_results:
                        assert hit_7 >= model_results['hit_rate_spread_within_3']

    def test_multiple_seasons(self):
        """Test comparison across multiple seasons."""
        # Test with 2 seasons to ensure multi-season logic works
        results = compare_v1_models(test_seasons=[2018, 2019])

        assert results['test_seasons'] == [2018, 2019]
        assert results['n_games'] > 0

        # Should have more games than single season
        single_season = compare_v1_models(test_seasons=[2018])
        assert results['n_games'] > single_season['n_games']


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
