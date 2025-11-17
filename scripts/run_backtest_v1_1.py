#!/usr/bin/env python3
"""
Ball Knower v1.1 Backtest Runner

Command-line interface for running calibration and backtesting of the
Ball Knower v1.1 calibrated spread model.

Usage:
    python scripts/run_backtest_v1_1.py \\
        --season 2025 \\
        --train-weeks 1-10 \\
        --test-weeks 11-18 \\
        --edge-thresholds 1,2,3,4

This will:
    1. Calibrate weights on weeks 1-10
    2. Generate predictions for weeks 11-18
    3. Calculate ATS performance at edge thresholds [1, 2, 3, 4]
    4. Save full results to output/backtest_v1_1_2025_train_1-10_test_11-18.csv
    5. Print summary statistics to stdout
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple


# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from ball_knower.models.v1_1_calibration import (
    calibrate_weights,
    build_week_lines_v1_1,
    load_schedule_data,
)


def parse_week_range(week_str: str) -> List[int]:
    """
    Parse week range string into list of week numbers.

    Supports:
        - Ranges: "1-10" → [1, 2, 3, ..., 10]
        - Comma-separated: "1,3,5" → [1, 3, 5]
        - Mixed: "1-3,5,7-9" → [1, 2, 3, 5, 7, 8, 9]

    Args:
        week_str: Week range string

    Returns:
        List of week numbers

    Raises:
        ValueError: If format is invalid
    """
    weeks = []

    for part in week_str.split(','):
        part = part.strip()

        if '-' in part:
            # Range like "1-10"
            try:
                start, end = part.split('-')
                weeks.extend(range(int(start), int(end) + 1))
            except ValueError:
                raise ValueError(f"Invalid week range: {part}")
        else:
            # Single week
            try:
                weeks.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid week number: {part}")

    return sorted(list(set(weeks)))  # Remove duplicates and sort


def calculate_ats_performance(
    predictions_df: pd.DataFrame,
    edge_threshold: float,
    model_col: str = 'bk_line_v1_1'
) -> dict:
    """
    Calculate Against-The-Spread (ATS) performance at a given edge threshold.

    A bet is placed when |model_line - vegas_line| >= edge_threshold.

    Args:
        predictions_df: DataFrame with predictions and actual results
        edge_threshold: Minimum edge (in points) to place a bet
        model_col: Column name for model predictions (default: 'bk_line_v1_1')

    Returns:
        Dictionary with ATS metrics:
            - num_bets: Number of bets placed
            - wins: Number of winning bets
            - losses: Number of losing bets
            - pushes: Number of pushes
            - win_rate: Win percentage (excluding pushes)
            - roi: Return on investment (assuming -110 odds)
    """
    # Filter for games with results and Vegas lines
    df = predictions_df[
        predictions_df['vegas_line'].notna() &
        predictions_df['home_score'].notna() &
        predictions_df['away_score'].notna()
    ].copy()

    if len(df) == 0:
        return {
            'num_bets': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'win_rate': 0.0,
            'roi': 0.0,
        }

    # Calculate actual margin (negative = home won by more than spread)
    df['actual_margin'] = df['home_score'] - df['away_score']

    # Calculate edge
    df['edge'] = df[model_col] - df['vegas_line']

    # Filter for bets where |edge| >= threshold
    bets = df[df['edge'].abs() >= edge_threshold].copy()

    if len(bets) == 0:
        return {
            'num_bets': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'win_rate': 0.0,
            'roi': 0.0,
        }

    # Determine bet outcome
    # If model line < Vegas line: bet on home team
    # If model line > Vegas line: bet on away team
    # Win if: (bet home and home covers) OR (bet away and away covers)

    def determine_outcome(row):
        """Determine if bet won, lost, or pushed."""
        edge = row['edge']
        vegas = row['vegas_line']
        actual = row['actual_margin']

        # Bet on home if model thinks home will cover (model line < vegas)
        # Bet on away if model thinks away will cover (model line > vegas)

        if edge < 0:
            # Model favors home more than Vegas → bet home
            # Home covers if actual_margin < vegas_line
            if actual < vegas - 0.5:
                return 'W'
            elif actual > vegas + 0.5:
                return 'L'
            else:
                return 'P'
        else:
            # Model favors away more than Vegas → bet away
            # Away covers if actual_margin > vegas_line
            if actual > vegas + 0.5:
                return 'W'
            elif actual < vegas - 0.5:
                return 'L'
            else:
                return 'P'

    bets['outcome'] = bets.apply(determine_outcome, axis=1)

    wins = (bets['outcome'] == 'W').sum()
    losses = (bets['outcome'] == 'L').sum()
    pushes = (bets['outcome'] == 'P').sum()

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    # ROI calculation (assuming -110 odds: risk $110 to win $100)
    # Win: +$100, Loss: -$110, Push: $0
    total_wagered = (wins + losses) * 110
    total_profit = (wins * 100) - (losses * 110)
    roi = (total_profit / total_wagered) if total_wagered > 0 else 0.0

    return {
        'num_bets': len(bets),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'roi': roi,
    }


def run_backtest(
    season: int,
    train_weeks: List[int],
    test_weeks: List[int],
    edge_thresholds: List[float],
    data_dir: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run full v1.1 backtest: calibrate on train weeks, test on test weeks.

    Args:
        season: NFL season year
        train_weeks: List of weeks to train on
        test_weeks: List of weeks to test on
        edge_thresholds: List of edge thresholds for ATS analysis
        data_dir: Directory containing data files
        output_dir: Directory to save results

    Returns:
        Tuple of:
            - predictions_df: DataFrame with all predictions
            - metrics: Dictionary with backtest metrics
    """
    print(f"\n{'='*70}")
    print(f"BALL KNOWER v1.1 BACKTEST")
    print(f"{'='*70}")
    print(f"Season: {season}")
    print(f"Training weeks: {min(train_weeks)}-{max(train_weeks)} ({len(train_weeks)} weeks)")
    print(f"Testing weeks: {min(test_weeks)}-{max(test_weeks)} ({len(test_weeks)} weeks)")
    print(f"Edge thresholds: {edge_thresholds}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*70}\n")

    # Step 1: Calibrate weights on training weeks
    print("STEP 1: Calibrating weights on training data\n")
    weights = calibrate_weights(season, train_weeks, data_dir)

    # Step 2: Generate predictions for test weeks
    print(f"\nSTEP 2: Generating predictions for test weeks\n")
    all_predictions = []

    for week in test_weeks:
        try:
            week_preds = build_week_lines_v1_1(season, week, weights, data_dir)
            week_preds['week'] = week
            all_predictions.append(week_preds)
        except FileNotFoundError as e:
            print(f"⚠️  Warning: Could not generate predictions for week {week}: {e}")
            continue

    if not all_predictions:
        raise ValueError("No predictions generated for any test week")

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Load schedule to get actual scores
    schedule = load_schedule_data(season)
    schedule_subset = schedule[schedule['week'].isin(test_weeks)][
        ['week', 'home_team', 'away_team', 'home_score', 'away_score']
    ].copy()

    # Merge actual scores into predictions
    predictions_df = predictions_df.merge(
        schedule_subset,
        on=['week', 'home_team', 'away_team'],
        how='left'
    )

    print(f"\n✓ Generated {len(predictions_df)} predictions across {len(test_weeks)} weeks\n")

    # Step 3: Calculate overall metrics
    print("STEP 3: Calculating performance metrics\n")

    # Filter for games with Vegas lines and results
    completed_games = predictions_df[
        predictions_df['vegas_line'].notna() &
        predictions_df['home_score'].notna()
    ].copy()

    if len(completed_games) == 0:
        print("⚠️  No completed games with Vegas lines found in test period")
        metrics = {
            'mae_v1_1': None,
            'rmse_v1_1': None,
            'mae_v1_0': None,
            'rmse_v1_0': None,
            'correlation_v1_1': None,
            'correlation_v1_0': None,
        }
    else:
        # Calculate errors
        completed_games['error_v1_1'] = completed_games['bk_line_v1_1'] - completed_games['vegas_line']
        completed_games['error_v1_0'] = completed_games['bk_line_v1_0'] - completed_games['vegas_line']

        mae_v1_1 = completed_games['error_v1_1'].abs().mean()
        rmse_v1_1 = np.sqrt((completed_games['error_v1_1'] ** 2).mean())

        mae_v1_0 = completed_games['error_v1_0'].abs().mean()
        rmse_v1_0 = np.sqrt((completed_games['error_v1_0'] ** 2).mean())

        # Correlation with Vegas
        corr_v1_1 = completed_games[['bk_line_v1_1', 'vegas_line']].corr().iloc[0, 1]
        corr_v1_0 = completed_games[['bk_line_v1_0', 'vegas_line']].corr().iloc[0, 1]

        metrics = {
            'mae_v1_1': mae_v1_1,
            'rmse_v1_1': rmse_v1_1,
            'mae_v1_0': mae_v1_0,
            'rmse_v1_0': rmse_v1_0,
            'correlation_v1_1': corr_v1_1,
            'correlation_v1_0': corr_v1_0,
            'num_games': len(completed_games),
        }

        print(f"Model Performance (vs Vegas):")
        print(f"  v1.1 MAE:  {mae_v1_1:.3f} points")
        print(f"  v1.1 RMSE: {rmse_v1_1:.3f} points")
        print(f"  v1.1 Correlation: {corr_v1_1:.3f}")
        print(f"\n  v1.0 MAE:  {mae_v1_0:.3f} points")
        print(f"  v1.0 RMSE: {rmse_v1_0:.3f} points")
        print(f"  v1.0 Correlation: {corr_v1_0:.3f}")
        print(f"\n  Games analyzed: {len(completed_games)}")

    # Step 4: ATS analysis at different edge thresholds
    print(f"\n{'='*70}")
    print("STEP 4: Against-The-Spread (ATS) Performance")
    print(f"{'='*70}\n")

    ats_results = []

    for threshold in edge_thresholds:
        ats_v1_1 = calculate_ats_performance(predictions_df, threshold, 'bk_line_v1_1')
        ats_v1_0 = calculate_ats_performance(predictions_df, threshold, 'bk_line_v1_0')

        ats_results.append({
            'threshold': threshold,
            'model': 'v1.1',
            **ats_v1_1
        })
        ats_results.append({
            'threshold': threshold,
            'model': 'v1.0',
            **ats_v1_0
        })

        print(f"Edge >= {threshold} points:")
        print(f"  v1.1: {ats_v1_1['wins']}-{ats_v1_1['losses']}-{ats_v1_1['pushes']} "
              f"({ats_v1_1['win_rate']:.1%} win rate, {ats_v1_1['roi']:+.1%} ROI, "
              f"{ats_v1_1['num_bets']} bets)")
        print(f"  v1.0: {ats_v1_0['wins']}-{ats_v1_0['losses']}-{ats_v1_0['pushes']} "
              f"({ats_v1_0['win_rate']:.1%} win rate, {ats_v1_0['roi']:+.1%} ROI, "
              f"{ats_v1_0['num_bets']} bets)")
        print()

    metrics['ats_results'] = ats_results

    # Step 5: Save results
    print(f"{'='*70}")
    print("STEP 5: Saving results")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV filename
    train_range = f"{min(train_weeks)}-{max(train_weeks)}"
    test_range = f"{min(test_weeks)}-{max(test_weeks)}"
    csv_filename = f"backtest_v1_1_{season}_train_{train_range}_test_{test_range}.csv"
    csv_path = output_dir / csv_filename

    predictions_df.to_csv(csv_path, index=False)
    print(f"✓ Saved predictions: {csv_path}")

    # Save ATS summary
    ats_df = pd.DataFrame(ats_results)
    ats_filename = f"backtest_v1_1_{season}_ats_summary.csv"
    ats_path = output_dir / ats_filename
    ats_df.to_csv(ats_path, index=False)
    print(f"✓ Saved ATS summary: {ats_path}")

    return predictions_df, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run Ball Knower v1.1 calibration and backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on weeks 1-10, test on week 11
  python scripts/run_backtest_v1_1.py --season 2025 --train-weeks 1-10 --test-weeks 11

  # Train on weeks 1-10, test on weeks 11-18, with multiple edge thresholds
  python scripts/run_backtest_v1_1.py --season 2025 \\
      --train-weeks 1-10 \\
      --test-weeks 11-18 \\
      --edge-thresholds 1,2,3,4,5

  # Train on specific weeks, test on others
  python scripts/run_backtest_v1_1.py --season 2025 \\
      --train-weeks 1-5,7-10 \\
      --test-weeks 6,11-13
        """
    )

    parser.add_argument('--season', type=int, required=True,
                        help='NFL season year (e.g., 2025)')
    parser.add_argument('--train-weeks', type=str, required=True,
                        help='Training weeks (e.g., "1-10" or "1,3,5-8")')
    parser.add_argument('--test-weeks', type=str, required=True,
                        help='Testing weeks (e.g., "11-18" or "11,12,15")')
    parser.add_argument('--edge-thresholds', type=str, default='1,2,3,4',
                        help='Comma-separated edge thresholds in points (default: "1,2,3,4")')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: data/current_season/)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory (default: output/)')

    args = parser.parse_args()

    # Parse week ranges
    try:
        train_weeks = parse_week_range(args.train_weeks)
        test_weeks = parse_week_range(args.test_weeks)
        edge_thresholds = [float(x.strip()) for x in args.edge_thresholds.split(',')]
    except ValueError as e:
        print(f"❌ Error parsing arguments: {e}")
        sys.exit(1)

    # Validate no overlap
    overlap = set(train_weeks) & set(test_weeks)
    if overlap:
        print(f"⚠️  Warning: Train and test weeks overlap: {sorted(overlap)}")
        print("    This may lead to overfitting. Consider using separate weeks.")

    # Resolve directories
    if args.data_dir is None:
        data_dir = _REPO_ROOT / 'data' / 'current_season'
    else:
        data_dir = Path(args.data_dir).resolve()

    output_dir = Path(args.output_dir).resolve()

    # Run backtest
    try:
        predictions_df, metrics = run_backtest(
            season=args.season,
            train_weeks=train_weeks,
            test_weeks=test_weeks,
            edge_thresholds=edge_thresholds,
            data_dir=data_dir,
            output_dir=output_dir,
        )

        print(f"\n{'='*70}")
        print("✓ BACKTEST COMPLETE")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
