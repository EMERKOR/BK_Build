"""
Tests for backtest PnL and CLV metrics

Tests the compute_ats_pnl and compute_clv_metrics functions
using synthetic DataFrames.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.run_backtests import compute_ats_pnl, compute_clv_metrics


def test_compute_ats_pnl_all_wins():
    """Test PnL calculation with all winning bets."""
    df = pd.DataFrame({
        'bet_flag': [1, 1, 1, 0, 0],
        'bet_result': [1, 1, 1, 0, 0]  # 3 wins, 2 no-bets
    })

    result = compute_ats_pnl(df)

    assert result['n_bets'] == 3
    assert result['n_wins'] == 3
    assert result['n_losses'] == 0
    assert result['n_pushes'] == 0
    assert result['win_rate'] == 1.0
    assert abs(result['units_won'] - 2.7273) < 0.01  # 3 * 0.9091
    assert abs(result['roi'] - 0.9091) < 0.01

    print("✓ test_compute_ats_pnl_all_wins passed")


def test_compute_ats_pnl_all_losses():
    """Test PnL calculation with all losing bets."""
    df = pd.DataFrame({
        'bet_flag': [1, 1, 1, 0],
        'bet_result': [-1, -1, -1, 0]  # 3 losses, 1 no-bet
    })

    result = compute_ats_pnl(df)

    assert result['n_bets'] == 3
    assert result['n_wins'] == 0
    assert result['n_losses'] == 3
    assert result['n_pushes'] == 0
    assert result['win_rate'] == 0.0
    assert result['units_won'] == -3.0
    assert result['roi'] == -1.0

    print("✓ test_compute_ats_pnl_all_losses passed")


def test_compute_ats_pnl_mixed():
    """Test PnL calculation with mixed results including pushes."""
    df = pd.DataFrame({
        'bet_flag': [1, 1, 1, 1, 1, 0],
        'bet_result': [1, 1, -1, -1, 0, 0]  # 2 wins, 2 losses, 1 push, 1 no-bet
    })

    result = compute_ats_pnl(df)

    assert result['n_bets'] == 5
    assert result['n_wins'] == 2
    assert result['n_losses'] == 2
    assert result['n_pushes'] == 1
    assert result['win_rate'] == 0.5  # 2/(2+2), pushes excluded
    # Units: 2 * 0.9091 - 2 * 1.0 = 1.8182 - 2.0 = -0.1818
    assert abs(result['units_won'] - (-0.1818)) < 0.01
    assert abs(result['roi'] - (-0.1818/5)) < 0.01

    print("✓ test_compute_ats_pnl_mixed passed")


def test_compute_ats_pnl_no_bets():
    """Test PnL calculation with no bets placed."""
    df = pd.DataFrame({
        'bet_flag': [0, 0, 0],
        'bet_result': [0, 0, 0]
    })

    result = compute_ats_pnl(df)

    assert result['n_bets'] == 0
    assert result['n_wins'] == 0
    assert result['n_losses'] == 0
    assert result['n_pushes'] == 0
    assert result['win_rate'] == 0.0
    assert result['units_won'] == 0.0
    assert result['roi'] == 0.0

    print("✓ test_compute_ats_pnl_no_bets passed")


def test_compute_ats_pnl_breakeven():
    """Test PnL calculation at breakeven (52.38% win rate)."""
    # At -110 pricing, breakeven is ~52.38% win rate (11 wins in 21 bets)
    df = pd.DataFrame({
        'bet_flag': [1] * 21,
        'bet_result': [1] * 11 + [-1] * 10  # 11 wins, 10 losses
    })

    result = compute_ats_pnl(df)

    assert result['n_bets'] == 21
    assert result['n_wins'] == 11
    assert result['n_losses'] == 10
    # Units: 11 * 0.9091 - 10 * 1.0 = 10.0001 - 10.0 ≈ 0
    assert abs(result['units_won']) < 0.01  # Should be very close to 0

    print("✓ test_compute_ats_pnl_breakeven passed")


def test_compute_clv_metrics_all_positive():
    """Test CLV calculation with all positive CLV."""
    df = pd.DataFrame({
        'bet_flag': [1, 1, 1, 0, 0],
        'clv_diff': [2.5, 1.0, 3.0, np.nan, np.nan]
    })

    result = compute_clv_metrics(df)

    assert result['n_bets'] == 3
    assert abs(result['mean_clv_diff'] - 2.166667) < 0.01
    assert result['pct_beating_closing_line'] == 1.0

    print("✓ test_compute_clv_metrics_all_positive passed")


def test_compute_clv_metrics_mixed():
    """Test CLV calculation with mixed positive/negative CLV."""
    df = pd.DataFrame({
        'bet_flag': [1, 1, 1, 1, 0],
        'clv_diff': [2.0, -0.5, 1.5, -1.0, np.nan]
    })

    result = compute_clv_metrics(df)

    assert result['n_bets'] == 4
    assert abs(result['mean_clv_diff'] - 0.5) < 0.01  # (2.0 - 0.5 + 1.5 - 1.0) / 4
    assert result['pct_beating_closing_line'] == 0.5  # 2 out of 4

    print("✓ test_compute_clv_metrics_mixed passed")


def test_compute_clv_metrics_no_bets():
    """Test CLV calculation with no bets."""
    df = pd.DataFrame({
        'bet_flag': [0, 0, 0],
        'clv_diff': [np.nan, np.nan, np.nan]
    })

    result = compute_clv_metrics(df)

    assert result['n_bets'] == 0
    assert result['mean_clv_diff'] == 0.0
    assert result['pct_beating_closing_line'] == 0.0

    print("✓ test_compute_clv_metrics_no_bets passed")


def test_compute_clv_metrics_missing_column():
    """Test CLV calculation when clv_diff column is missing."""
    df = pd.DataFrame({
        'bet_flag': [1, 1, 1]
        # No clv_diff column
    })

    result = compute_clv_metrics(df)

    assert result['n_bets'] == 0
    assert result['mean_clv_diff'] == 0.0
    assert result['pct_beating_closing_line'] == 0.0

    print("✓ test_compute_clv_metrics_missing_column passed")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*60)
    print("Running Backtest Metrics Tests")
    print("="*60 + "\n")

    test_compute_ats_pnl_all_wins()
    test_compute_ats_pnl_all_losses()
    test_compute_ats_pnl_mixed()
    test_compute_ats_pnl_no_bets()
    test_compute_ats_pnl_breakeven()

    test_compute_clv_metrics_all_positive()
    test_compute_clv_metrics_mixed()
    test_compute_clv_metrics_no_bets()
    test_compute_clv_metrics_missing_column()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
