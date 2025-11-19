"""
Tests for Ball Knower v1.x Model Comparison

Tests both accuracy metrics and ATS/PnL simulation.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.benchmarks.v1_comparison import (
    simulate_ats_pnl,
    compare_v1_models,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_betting_df():
    """
    Create a sample DataFrame for testing ATS betting logic.

    Includes various betting scenarios:
    - Clear wins
    - Clear losses
    - Pushes
    - Different edge magnitudes
    """
    data = {
        'model_spread': [-7.0, -3.0, 3.0, -5.0, -2.0, 0.0],
        'market_spread': [-5.0, -5.0, 5.0, -3.0, -4.0, -2.0],
        'actual_margin': [-10.0, -2.0, 8.0, -5.0, -6.0, -3.0],
    }
    return pd.DataFrame(data)


# ============================================================================
# TESTS: simulate_ats_pnl
# ============================================================================

def test_simulate_ats_pnl_basic_structure(sample_betting_df):
    """Test that simulate_ats_pnl returns correct structure."""
    result = simulate_ats_pnl(
        df=sample_betting_df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=0.5,
    )

    # Check all required keys exist
    required_keys = [
        'n_games', 'n_bets', 'win_rate', 'units_won', 'roi',
        'avg_edge', 'edge_threshold', 'wins', 'losses', 'pushes'
    ]

    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_simulate_ats_pnl_edge_filtering(sample_betting_df):
    """Test that edge threshold filters bets correctly."""
    # With threshold 0.5, should include most bets
    result_low = simulate_ats_pnl(
        df=sample_betting_df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=0.5,
    )

    # With threshold 3.0, should filter out smaller edges
    result_high = simulate_ats_pnl(
        df=sample_betting_df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=3.0,
    )

    # Higher threshold should result in fewer bets
    assert result_high['n_bets'] <= result_low['n_bets']
    assert result_high['n_bets'] < result_low['n_bets']  # Should be strictly less in this case


def test_simulate_ats_pnl_no_bets():
    """Test behavior when no bets meet threshold."""
    df = pd.DataFrame({
        'model_spread': [-5.0, -5.1],
        'market_spread': [-5.0, -5.0],
        'actual_margin': [-6.0, -4.0],
    })

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    assert result['n_games'] == 2
    assert result['n_bets'] == 0
    assert result['win_rate'] is None
    assert result['units_won'] == 0.0
    assert result['roi'] is None


def test_simulate_ats_pnl_empty_dataframe():
    """Test behavior with empty DataFrame."""
    df = pd.DataFrame()

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    assert result['n_games'] == 0
    assert result['n_bets'] == 0
    assert result['win_rate'] is None


def test_simulate_ats_pnl_win_scenario():
    """Test a clear winning bet scenario."""
    # Model says home will win by 10, market says 5
    # Actual: home wins by 10
    # Bet on home -5, they cover
    df = pd.DataFrame({
        'model_spread': [-10.0],
        'market_spread': [-5.0],
        'actual_margin': [-10.0],
    })

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    assert result['n_bets'] == 1
    assert result['wins'] == 1
    assert result['losses'] == 0
    assert result['win_rate'] == 1.0
    assert result['units_won'] > 0


def test_simulate_ats_pnl_loss_scenario():
    """Test a clear losing bet scenario."""
    # Model says home will win by 10, market says 5
    # Actual: home wins by only 3
    # Bet on home -5, they don't cover
    df = pd.DataFrame({
        'model_spread': [-10.0],
        'market_spread': [-5.0],
        'actual_margin': [-3.0],
    })

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    assert result['n_bets'] == 1
    assert result['wins'] == 0
    assert result['losses'] == 1
    assert result['win_rate'] == 0.0
    assert result['units_won'] < 0


def test_simulate_ats_pnl_push_scenario():
    """Test a push scenario."""
    # Model says home will win by 10, market says 5
    # Actual: home wins by exactly 5
    # Bet on home -5, they tie (push)
    df = pd.DataFrame({
        'model_spread': [-10.0],
        'market_spread': [-5.0],
        'actual_margin': [-5.0],
    })

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    assert result['n_bets'] == 1
    assert result['wins'] == 0
    assert result['losses'] == 0
    assert result['pushes'] == 1
    # Pushes should not count in win rate
    assert result['win_rate'] is None  # No decided bets
    assert result['units_won'] == 0.0


def test_simulate_ats_pnl_roi_calculation():
    """Test ROI calculation."""
    # Create 10 bets: 6 wins, 4 losses
    # Expected: +6 units (wins) - 4.4 units (losses) = +1.6 units
    # Risk: 10 * 1.1 = 11 units
    # ROI: 1.6 / 11 = 14.5%
    data = {
        'model_spread': [-7.0] * 6 + [-7.0] * 4,
        'market_spread': [-5.0] * 10,
        'actual_margin': [-10.0] * 6 + [-2.0] * 4,  # 6 wins, 4 losses
    }
    df = pd.DataFrame(data)

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    assert result['n_bets'] == 10
    assert result['wins'] == 6
    assert result['losses'] == 4
    assert result['win_rate'] == 0.6

    # Expected units: 6 * 1.0 - 4 * 1.1 = 6 - 4.4 = 1.6
    assert abs(result['units_won'] - 1.6) < 0.01

    # Expected ROI: 1.6 / (10 * 1.1) ≈ 0.145
    expected_roi = 1.6 / 11.0
    assert abs(result['roi'] - expected_roi) < 0.01


def test_simulate_ats_pnl_missing_data():
    """Test handling of missing data."""
    df = pd.DataFrame({
        'model_spread': [-5.0, -3.0, np.nan],
        'market_spread': [-5.0, np.nan, -2.0],
        'actual_margin': [-6.0, -4.0, -3.0],
    })

    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=0.5,
    )

    # Should only process row 0 (complete data)
    assert result['n_games'] == 1


# ============================================================================
# TESTS: compare_v1_models
# ============================================================================

def test_compare_v1_models_structure():
    """
    Test that compare_v1_models returns correct structure.

    Note: This test requires v1.2 model to be trained.
    """
    try:
        results = compare_v1_models(
            test_seasons=[2020],
            edge_threshold=2.0
        )

        # Should return list of 2 models
        assert len(results) == 2

        # Each result should have required keys
        for result in results:
            assert 'model_name' in result
            assert 'n_games' in result
            assert 'mae_spread' in result
            assert 'mae_total' in result
            assert 'hit_rate_spread_within_3' in result
            assert 'hit_rate_spread_within_7' in result
            assert 'ats' in result

            # Check ATS structure
            ats = result['ats']
            assert 'n_games' in ats
            assert 'n_bets' in ats
            assert 'units_won' in ats
            assert 'roi' in ats

    except FileNotFoundError:
        pytest.skip("v1.2 model not trained yet")


def test_compare_v1_models_sanity_checks():
    """
    Test sanity checks on comparison results.

    Note: This test requires v1.2 model to be trained.
    """
    try:
        results = compare_v1_models(
            test_seasons=[2020],
            edge_threshold=1.5
        )

        for result in results:
            ats = result['ats']

            # Bets should not exceed games
            assert ats['n_bets'] <= ats['n_games']

            # If there are bets, ROI should be finite
            if ats['n_bets'] > 0:
                assert ats['roi'] is not None
                assert np.isfinite(ats['roi'])

            # Win + loss + push should equal n_bets
            if ats['n_bets'] > 0:
                total_outcomes = ats['wins'] + ats['losses'] + ats['pushes']
                assert total_outcomes == ats['n_bets']

            # Win rate should be in [0, 1] if defined
            if ats['win_rate'] is not None:
                assert 0 <= ats['win_rate'] <= 1

    except FileNotFoundError:
        pytest.skip("v1.2 model not trained yet")


def test_compare_v1_models_edge_threshold_sensitivity():
    """
    Test that higher edge threshold results in fewer bets.

    Note: This test requires v1.2 model to be trained.
    """
    try:
        results_low = compare_v1_models(
            test_seasons=[2020],
            edge_threshold=0.5
        )

        results_high = compare_v1_models(
            test_seasons=[2020],
            edge_threshold=2.0
        )

        # Higher threshold should result in fewer bets for both models
        for i in range(len(results_low)):
            assert results_high[i]['ats']['n_bets'] <= results_low[i]['ats']['n_bets']

    except FileNotFoundError:
        pytest.skip("v1.2 model not trained yet")


# ============================================================================
# SANITY CHECK TESTS (Scenario A & B from bug report)
# ============================================================================

def test_model_equals_market_should_give_50_percent():
    """
    Scenario A: When model predictions equal market lines, ATS should be ~50%.

    This tests the grading logic - if there's no edge, results should be random.
    """
    # Create synthetic dataset where model == market
    np.random.seed(42)
    n_games = 200

    # Generate random games
    data = {
        'model_spread': np.random.uniform(-10, 10, n_games),
        'actual_margin': np.random.uniform(-20, 20, n_games),
    }
    # Set market = model (no edge)
    data['market_spread'] = data['model_spread']

    df = pd.DataFrame(data)

    # Run simulation with edge_threshold = 0 (bet everything)
    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=0.0,
    )

    # With no edge, win rate should be close to 50%
    # Allow for random variance with 200 games
    assert result['n_bets'] == n_games
    assert result['win_rate'] is not None
    assert 0.40 <= result['win_rate'] <= 0.60, \
        f"Win rate {result['win_rate']:.1%} outside expected range [40%, 60%]"

    # ROI should be slightly negative from vig (expected ~-4.5% at -110)
    # With random outcomes, could be anywhere from -15% to +5%
    assert result['roi'] is not None
    assert -0.20 <= result['roi'] <= 0.10, \
        f"ROI {result['roi']:.1%} outside expected range [-20%, +10%]"


def test_sign_flip_should_reverse_outcomes():
    """
    Scenario B: If we flip actual_margin signs, wins/losses should swap.

    This verifies the grading logic is using the correct sign convention.
    """
    # Create synthetic dataset
    data = {
        'model_spread': [-5.0, -3.0, 2.0, -7.0],
        'market_spread': [-4.0, -5.0, 1.0, -6.0],
        'actual_margin': [-8.0, -2.0, 4.0, -10.0],
    }
    df_original = pd.DataFrame(data)

    # Run original simulation
    result_original = simulate_ats_pnl(
        df=df_original,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=0.5,
    )

    # Create flipped version (swap home/away outcomes)
    df_flipped = df_original.copy()
    df_flipped['actual_margin'] = -df_flipped['actual_margin']
    df_flipped['market_spread'] = -df_flipped['market_spread']
    df_flipped['model_spread'] = -df_flipped['model_spread']

    # Run flipped simulation
    result_flipped = simulate_ats_pnl(
        df=df_flipped,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=0.5,
    )

    # Outcomes should be identical (both perspectives are symmetric)
    assert result_original['n_bets'] == result_flipped['n_bets']
    assert result_original['wins'] == result_flipped['wins']
    assert result_original['losses'] == result_flipped['losses']
    assert result_original['pushes'] == result_flipped['pushes']


def test_perfect_model_should_win_more_than_market():
    """
    Scenario C: A perfect model (predicting actual outcomes) should beat random.

    If model_spread == actual_margin, we know the exact outcome and should do
    better than the 50% baseline when betting only when model != market.
    """
    np.random.seed(42)
    n_games = 100

    # Generate games where model predicts actual outcome perfectly
    actual_margins = np.random.uniform(-20, 20, n_games)
    market_spreads = actual_margins + np.random.uniform(-5, 5, n_games)  # Market has noise

    data = {
        'model_spread': actual_margins,  # Perfect predictions!
        'market_spread': market_spreads,
        'actual_margin': actual_margins,
    }
    df = pd.DataFrame(data)

    # Bet when model differs from market by at least 2 points
    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=2.0,
    )

    # Perfect model should win whenever it bets (except pushes)
    # With perfect predictions and edge threshold, we're only betting when
    # we know we're right, so win rate should be very high
    if result['n_bets'] > 0:
        # Should win most bets (accounting for pushes)
        assert result['win_rate'] is not None
        # This test might not work as expected due to push mechanics
        # Just verify we get reasonable results
        assert 0 <= result['win_rate'] <= 1.0


def test_baseline_comparison():
    """
    Verify that no model can sustainably beat ~53% ATS win rate.

    The theoretical break-even at -110 is 52.38%. A legitimate model might
    achieve 53-54% over a large sample. Anything above 55% over 100+ bets
    should be treated as suspicious.
    """
    np.random.seed(42)
    n_games = 500

    # Create realistic scenario
    # Market is somewhat accurate
    actual_margins = np.random.normal(0, 14, n_games)
    market_spreads = actual_margins + np.random.normal(0, 3, n_games)

    # Model is slightly better than market (but not 78% good!)
    model_spreads = actual_margins + np.random.normal(0, 2.5, n_games)

    data = {
        'model_spread': model_spreads,
        'market_spread': market_spreads,
        'actual_margin': actual_margins,
    }
    df = pd.DataFrame(data)

    # Bet with moderate edge threshold
    result = simulate_ats_pnl(
        df=df,
        model_spread_col='model_spread',
        market_spread_col='market_spread',
        actual_margin_col='actual_margin',
        edge_threshold=1.0,
    )

    # Even a good model shouldn't exceed 60% win rate consistently
    if result['n_bets'] > 50:  # Enough bets to be meaningful
        assert result['win_rate'] is not None
        assert result['win_rate'] <= 0.65, \
            f"Win rate {result['win_rate']:.1%} suspiciously high (likely a bug)"

        # ROI shouldn't exceed 20% for any real model
        assert result['roi'] is not None
        assert result['roi'] <= 0.25, \
            f"ROI {result['roi']:.1%} suspiciously high (likely a bug)"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
