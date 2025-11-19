#!/usr/bin/env python3
"""
Ball Knower v1.0 Performance Report

Reports v1.0 spread prediction performance vs Vegas in terms of MAE and
error distribution, comparing both model and market predictions to actual outcomes.

Model: Ball Knower v1.0 (nfelo-based baseline)
Formula: spread = 2.67 + (nfelo_diff × 0.0447)

Usage:
    python src/report_v1_0_performance.py
    python src/report_v1_0_performance.py --season-min 2015 --season-max 2023
    python src/report_v1_0_performance.py --output output/v1_0_errors.csv
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import ball_knower modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from ball_knower.datasets.v1_0 import build_training_frame
from ball_knower.benchmarks.v1_metrics import compute_v1_0_metrics, compute_v1_0_errors


# v1.0 Model Parameters (calibrated from nfelo historical analysis)
V1_0_INTERCEPT = 2.67
V1_0_NFELO_COEF = 0.0447


def apply_v1_0_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Ball Knower v1.0 model formula to generate spread predictions.

    v1.0 Formula: spread = 2.67 + (nfelo_diff × 0.0447)

    Args:
        df: DataFrame with nfelo_diff column

    Returns:
        DataFrame with added 'model_spread' column
    """
    df = df.copy()
    df['model_spread'] = V1_0_INTERCEPT + (df['nfelo_diff'] * V1_0_NFELO_COEF)
    return df


def print_performance_report(metrics: dict, season_min: int, season_max: int):
    """
    Print human-readable performance report.

    Args:
        metrics: Dictionary from compute_v1_0_metrics()
        season_min: Minimum season in dataset
        season_max: Maximum season in dataset
    """
    print("\n" + "="*70)
    print(f"Ball Knower v1.0 Performance vs Vegas ({season_min}–{season_max})")
    print("="*70)

    print(f"\nGames analyzed: {metrics['n_games']:,}")

    print("\n" + "-"*70)
    print("Mean Absolute Error (MAE)")
    print("-"*70)
    print(f"  Model (v1.0):  {metrics['model_mae']:>6.2f} points")
    print(f"  Vegas:         {metrics['market_mae']:>6.2f} points")

    improvement = metrics['model_mae_improvement']
    if improvement > 0:
        print(f"  Improvement:   {improvement:>6.2f} points (model better)")
    elif improvement < 0:
        print(f"  Improvement:   {improvement:>6.2f} points (model worse)")
    else:
        print(f"  Improvement:   {improvement:>6.2f} points (tied)")

    print("\n" + "-"*70)
    print("Mean Signed Error (Bias Check)")
    print("-"*70)
    print(f"  Model bias:    {metrics['model_mean_error']:>6.2f} points")
    print(f"  Vegas bias:    {metrics['market_mean_error']:>6.2f} points")

    if abs(metrics['model_mean_error']) > 0.5:
        direction = "high" if metrics['model_mean_error'] > 0 else "low"
        print(f"  ⚠ Model shows bias toward predicting too {direction}")
    else:
        print(f"  ✓ Model shows minimal bias")

    print("\n" + "-"*70)
    print("Error Distribution Percentiles")
    print("-"*70)
    print(f"                    Model      Vegas")
    print(f"  50th percentile:  {metrics['model_error_pct_50']:>5.2f}      {metrics['market_error_pct_50']:>5.2f}")
    print(f"  75th percentile:  {metrics['model_error_pct_75']:>5.2f}      {metrics['market_error_pct_75']:>5.2f}")
    print(f"  90th percentile:  {metrics['model_error_pct_90']:>5.2f}      {metrics['market_error_pct_90']:>5.2f}")

    print("\n" + "="*70)
    print()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Report Ball Knower v1.0 performance vs Vegas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: all available seasons
  python src/report_v1_0_performance.py

  # Specific season range
  python src/report_v1_0_performance.py --season-min 2015 --season-max 2023

  # Save per-game errors to CSV
  python src/report_v1_0_performance.py --output output/v1_0_errors.csv
        """
    )

    parser.add_argument(
        '--season-min',
        type=int,
        default=2009,
        help='Minimum season year (default: 2009, earliest in v1.0 dataset)'
    )

    parser.add_argument(
        '--season-max',
        type=int,
        default=2024,
        help='Maximum season year (default: 2024)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: save per-game errors to CSV file'
    )

    args = parser.parse_args()

    # Validate season range
    if args.season_min > args.season_max:
        print("Error: --season-min must be <= --season-max", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading v1.0 dataset ({args.season_min}–{args.season_max})...")

    # Load v1.0 training data
    try:
        df = build_training_frame(
            start_year=args.season_min,
            end_year=args.season_max
        )
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Loaded {len(df):,} games")

    # Apply v1.0 model
    print("Generating v1.0 predictions...")
    df = apply_v1_0_model(df)

    # Rename columns to match metrics helper expectations
    df = df.rename(columns={'home_line_close': 'market_spread'})

    # Compute metrics
    print("Computing performance metrics...")
    try:
        metrics = compute_v1_0_metrics(
            df,
            model_col='model_spread',
            market_col='market_spread',
            actual_col='actual_margin'
        )
    except Exception as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)
        sys.exit(1)

    # Print report
    print_performance_report(metrics, args.season_min, args.season_max)

    # Save detailed errors if requested
    if args.output:
        print(f"Saving per-game errors to {args.output}...")

        # Add error columns
        df_with_errors = compute_v1_0_errors(
            df,
            model_col='model_spread',
            market_col='market_spread',
            actual_col='actual_margin'
        )

        # Select columns for output
        output_cols = [
            'game_id', 'season', 'week',
            'home_team', 'away_team',
            'model_spread', 'market_spread', 'actual_margin',
            'home_score', 'away_score',
            'model_error', 'market_error',
            'abs_model_error', 'abs_market_error'
        ]

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV
        df_with_errors[output_cols].to_csv(output_path, index=False)
        print(f"✓ Saved {len(df_with_errors):,} game records to {args.output}")

    print("\nReport complete.\n")


if __name__ == '__main__':
    main()
