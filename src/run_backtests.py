#!/usr/bin/env python3
"""
Unified Backtest Driver for Ball Knower Models

This script runs season-by-season backtests for v1.0 and v1.2 models,
computing performance metrics and outputting results to CSV.

Features:
- Configurable season range
- Multiple model versions (v1.0, v1.2)
- Edge threshold filtering
- ATS record tracking
- Flat-stake ROI calculation
- Season-by-season performance metrics

Usage:
    python src/run_backtests.py --start-season 2019 --end-season 2024 --model v1.2
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import backtest_utils, config

# ============================================================================
# CONSTANTS
# ============================================================================

# v1.0 model parameters (calibrated)
V1_0_INTERCEPT = 2.67
V1_0_NFELO_COEF = 0.0447


# ============================================================================
# MAIN BACKTEST LOGIC
# ============================================================================

def run_backtest(start_season: int,
                end_season: int,
                model: str,
                edge_threshold: float,
                output_path: Path) -> None:
    """
    Run backtest across multiple seasons and output results to CSV.

    Args:
        start_season: First season to include
        end_season: Last season to include
        model: Model version ('v1.0' or 'v1.2')
        edge_threshold: Minimum absolute edge to count as a bet
        output_path: Path to save CSV results
    """
    print("\n" + "="*80)
    print(f"BALL KNOWER BACKTEST DRIVER - {model.upper()}")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Season range:    {start_season}-{end_season}")
    print(f"  Model:           {model}")
    print(f"  Edge threshold:  {edge_threshold} points")
    print(f"  Output:          {output_path}")

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print(f"\n[1/4] Loading historical data...")
    df = backtest_utils.load_nfelo_historical_data(
        min_season=start_season,
        max_season=end_season
    )
    print(f"  Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

    # ========================================================================
    # ENGINEER FEATURES
    # ========================================================================

    print(f"\n[2/4] Engineering features for {model}...")
    if model == 'v1.0':
        df = backtest_utils.engineer_v1_0_features(df)
    elif model == 'v1.2':
        df = backtest_utils.engineer_v1_2_features(df)
    else:
        raise ValueError(f"Unknown model: {model}")

    print(f"  Prepared {len(df):,} games with complete features")

    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================

    print(f"\n[3/4] Generating predictions...")
    if model == 'v1.0':
        df = backtest_utils.generate_v1_0_predictions(
            df,
            intercept=V1_0_INTERCEPT,
            nfelo_coef=V1_0_NFELO_COEF
        )
    elif model == 'v1.2':
        # Load calibrated model weights
        model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'
        if not model_file.exists():
            raise FileNotFoundError(
                f"v1.2 model file not found: {model_file}\n"
                f"Please run the v1.2 training script first."
            )

        model_params = backtest_utils.load_v1_2_model(model_file)
        df = backtest_utils.generate_v1_2_predictions(df, model_params)

    print(f"  Generated predictions for {len(df):,} games")
    print(f"  Mean absolute edge: {df['abs_edge'].mean():.2f} points")

    # ========================================================================
    # CALCULATE ATS RESULTS
    # ========================================================================

    # Check if actual scores are available
    has_scores = 'home_score' in df.columns and 'away_score' in df.columns

    if has_scores:
        print(f"\n  Computing ATS results from actual game outcomes...")
        df['ats_result'] = df.apply(backtest_utils.calculate_ats_result, axis=1)

        # Count unknowns (games without scores)
        unknown_count = (df['ats_result'] == 'unknown').sum()
        if unknown_count > 0:
            print(f"  Warning: {unknown_count} games missing scores")
    else:
        print(f"\n  Warning: No actual game scores available - ATS metrics will be zero")
        df['ats_result'] = 'unknown'

    # ========================================================================
    # COMPUTE SEASON-BY-SEASON METRICS
    # ========================================================================

    print(f"\n[4/4] Computing season-by-season metrics...")

    season_results = []

    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season]

        metrics = backtest_utils.compute_season_metrics(
            season_df,
            model_name=model,
            edge_threshold=edge_threshold
        )

        season_results.append({
            'season': season,
            'model': model,
            'edge_threshold': edge_threshold,
            'n_games': metrics['n_games'],
            'n_bets': metrics['n_bets'],
            'mae_vs_vegas': round(metrics['mae_vs_vegas'], 3),
            'ats_wins': metrics['ats_wins'],
            'ats_losses': metrics['ats_losses'],
            'ats_pushes': metrics['ats_pushes'],
            'flat_roi': round(metrics['flat_roi'], 4),
        })

    results_df = pd.DataFrame(season_results)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Results saved to: {output_path}")

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("SEASON-BY-SEASON RESULTS")
    print("="*80)
    print()
    print(results_df.to_string(index=False))

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    total_games = results_df['n_games'].sum()
    total_bets = results_df['n_bets'].sum()
    avg_mae = results_df['mae_vs_vegas'].mean()
    total_wins = results_df['ats_wins'].sum()
    total_losses = results_df['ats_losses'].sum()
    total_pushes = results_df['ats_pushes'].sum()

    print(f"\nTotal games: {total_games:,}")
    print(f"Total bets (edge >= {edge_threshold}): {total_bets:,} ({total_bets/total_games*100:.1f}%)")
    print(f"Average MAE vs Vegas: {avg_mae:.2f} points")

    if total_bets > 0:
        print(f"\nATS Record: {total_wins}-{total_losses}-{total_pushes}")
        win_pct = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
        print(f"Win Rate: {win_pct:.1f}%")

        overall_roi = backtest_utils.calculate_flat_roi(total_wins, total_losses, total_pushes)
        print(f"Overall Flat ROI: {overall_roi*100:.2f}%")
    else:
        print(f"\nNo bets met the edge threshold of {edge_threshold} points")

    print("\n" + "="*80 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Parse arguments and run backtest."""
    parser = argparse.ArgumentParser(
        description="Run season-by-season backtests for Ball Knower models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest v1.2 from 2019-2024 with 0.5 point edge threshold
  python src/run_backtests.py --start-season 2019 --end-season 2024 --model v1.2 --edge-threshold 0.5

  # Backtest v1.0 for all available seasons
  python src/run_backtests.py --model v1.0

  # Quick test on recent seasons only
  python src/run_backtests.py --start-season 2022 --end-season 2024 --model v1.2
        """
    )

    parser.add_argument(
        '--start-season',
        type=int,
        default=2009,
        help='First season to include (default: 2009)'
    )

    parser.add_argument(
        '--end-season',
        type=int,
        default=2024,
        help='Last season to include (default: 2024)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['v1.0', 'v1.2'],
        default='v1.2',
        help='Model version to backtest (default: v1.2)'
    )

    parser.add_argument(
        '--edge-threshold',
        type=float,
        default=0.0,
        help='Minimum absolute edge to count as a bet in points (default: 0.0)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: output/backtest_{model}_{start}_{end}.csv)'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_filename = f"backtest_{args.model}_{args.start_season}_{args.end_season}.csv"
        output_path = config.OUTPUT_DIR / output_filename
    else:
        output_path = Path(args.output)

    # Run backtest
    try:
        run_backtest(
            start_season=args.start_season,
            end_season=args.end_season,
            model=args.model,
            edge_threshold=args.edge_threshold,
            output_path=output_path
        )
    except Exception as e:
        print(f"\nError running backtest: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
