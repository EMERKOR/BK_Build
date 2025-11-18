#!/usr/bin/env python3
"""
Ball Knower v1.2 Backtest CLI

Evaluates v1.2 model against actual game outcomes using edge betting strategies.

Usage:
    python src/backtest_v1_2_model.py
    python src/backtest_v1_2_model.py --model-path /path/to/model.json
    python src/backtest_v1_2_model.py --start-season 2020 --end-season 2024
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ball_knower.eval import v1_2_backtest


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Backtest Ball Knower v1.2 against actual game outcomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model JSON file (default: output/ball_knower_v1_2_model.json)'
    )

    parser.add_argument(
        '--start-season',
        type=int,
        default=None,
        help='Filter to games from this season onwards (optional)'
    )

    parser.add_argument(
        '--end-season',
        type=int,
        default=None,
        help='Filter to games up to this season (optional)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BALL KNOWER v1.2 - BACKTEST vs ACTUAL OUTCOMES")
    print("="*80)

    # ========================================================================
    # STEP 1: Load v1.2 dataset
    # ========================================================================

    print("\n[1/5] Loading v1.2 dataset...")
    df = v1_2_backtest.load_v1_2_dataset(
        start_season=args.start_season,
        end_season=args.end_season
    )

    # ========================================================================
    # STEP 2: Merge game outcomes
    # ========================================================================

    print("\n[2/5] Merging actual game outcomes...")
    df = v1_2_backtest.merge_game_outcomes(df)

    # ========================================================================
    # STEP 3: Load trained model
    # ========================================================================

    print("\n[3/5] Loading trained v1.2 model...")
    model = v1_2_backtest.load_trained_v1_2_model(model_path=args.model_path)

    # ========================================================================
    # STEP 4: Generate predictions
    # ========================================================================

    print("\n[4/5] Generating predictions...")
    df = v1_2_backtest.add_v1_2_predictions(df, model)

    # ========================================================================
    # STEP 5: Compute metrics
    # ========================================================================

    print("\n[5/5] Computing backtest metrics...")

    # Replication metrics (how well we match Vegas)
    replication_metrics = v1_2_backtest.compute_replication_metrics(df)

    # Edge betting metrics at various thresholds
    edge_thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
    edge_betting_results = v1_2_backtest.compute_edge_betting_metrics(
        df, edge_thresholds
    )

    # ========================================================================
    # REPORT RESULTS
    # ========================================================================

    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)

    print(f"\nTotal games: {len(df):,}")
    print(f"Season range: {df['season'].min()} - {df['season'].max()}")
    print(f"Games with outcomes: {df['home_points'].notna().sum():,}")

    # ========================================================================
    # REPLICATION METRICS
    # ========================================================================

    print("\n" + "="*80)
    print("REPLICATION METRICS (How well do we match Vegas?)")
    print("="*80)

    print(f"\nMean Absolute Error:     {replication_metrics['mae']:.2f} points")
    print(f"Root Mean Squared Error: {replication_metrics['rmse']:.2f} points")
    print(f"Mean Residual:           {replication_metrics['mean_residual']:.3f} points (should be ~0)")
    print(f"Std of Residuals:        {replication_metrics['std_residual']:.2f} points")
    print(f"Median Abs Residual:     {replication_metrics['median_abs_residual']:.2f} points")

    print(f"\nCalibration:")
    print(f"  Within 1 point of Vegas: {replication_metrics['pct_within_1pt']:.1f}%")
    print(f"  Within 2 points:         {replication_metrics['pct_within_2pt']:.1f}%")
    print(f"  Within 3 points:         {replication_metrics['pct_within_3pt']:.1f}%")

    # ========================================================================
    # EDGE BETTING RESULTS
    # ========================================================================

    print("\n" + "="*80)
    print("EDGE BETTING STRATEGY RESULTS")
    print("="*80)

    print("\nStrategy: Bet when |BK_line - Vegas_line| >= threshold")
    print("  - If BK_line - Vegas_line >= threshold: BET HOME")
    print("  - If BK_line - Vegas_line <= -threshold: BET AWAY")
    print("  - Otherwise: NO BET")
    print("\nAssuming -110 odds (risk 1.1 units to win 1.0 unit)")

    print("\n" + "-"*80)
    print(f"{'Threshold':>10}  {'Bets':>6}  {'Win Rate':>10}  {'Push Rate':>10}  {'Units Won':>11}  {'ROI':>7}")
    print("-"*80)

    for _, row in edge_betting_results.iterrows():
        threshold = row['threshold']
        num_bets = int(row['num_bets'])
        win_rate = row['win_rate'] * 100
        push_rate = row['push_rate'] * 100
        units_won = row['units_won']
        roi_pct = row['roi_pct']

        print(f"{threshold:>10.1f}  {num_bets:>6}  {win_rate:>9.1f}%  {push_rate:>9.1f}%  {units_won:>+11.2f}  {roi_pct:>+6.1f}%")

    print("-"*80)

    # ========================================================================
    # KEY FINDINGS
    # ========================================================================

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Find best threshold (highest ROI)
    if len(edge_betting_results) > 0 and edge_betting_results['num_bets'].sum() > 0:
        best_idx = edge_betting_results['roi_pct'].idxmax()
        best_row = edge_betting_results.loc[best_idx]

        print(f"\nBest performing threshold: {best_row['threshold']:.1f}")
        print(f"  Bets: {int(best_row['num_bets'])}")
        print(f"  Win rate: {best_row['win_rate']*100:.1f}%")
        print(f"  ROI: {best_row['roi_pct']:+.1f}%")
        print(f"  Total units: {best_row['units_won']:+.2f}")

        # Calculate break-even win rate
        breakeven_rate = 11.0 / 21.0  # 52.38% at -110 odds
        print(f"\nBreak-even win rate at -110 odds: {breakeven_rate*100:.2f}%")

        # Show which thresholds beat break-even
        profitable = edge_betting_results[edge_betting_results['win_rate'] > breakeven_rate]
        if len(profitable) > 0:
            print(f"\nThresholds above break-even:")
            for _, row in profitable.iterrows():
                print(f"  {row['threshold']:.1f}: {row['win_rate']*100:.1f}% win rate, {row['roi_pct']:+.1f}% ROI")
        else:
            print("\nNo thresholds above break-even win rate.")
    else:
        print("\nNo bets placed at any threshold (edge thresholds may be too high).")

    # Model calibration assessment
    print(f"\nModel Calibration:")
    if abs(replication_metrics['mean_residual']) < 0.5:
        print("  ✓ Well calibrated (mean residual near 0)")
    else:
        direction = "overestimates" if replication_metrics['mean_residual'] > 0 else "underestimates"
        print(f"  ⚠ Model {direction} spreads by {abs(replication_metrics['mean_residual']):.2f} points on average")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
