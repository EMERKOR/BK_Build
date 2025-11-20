#!/usr/bin/env python3
"""
Ball Knower Build CLI

Unified command-line interface for Ball Knower training, backtesting, prediction, and export.

Usage:
    bk_build.py train-v1-2 --start-season 2009 --end-season 2024
    bk_build.py backtest --model v1.2 --start-season 2019 --end-season 2024
    bk_build.py predict --model v1.2 --season 2025 --week 11
    bk_build.py export-predictiontracker --model v1.2 --start-season 2019 --end-season 2024
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower import config
from ball_knower.utils import paths, version


# ============================================================================
# SUBCOMMAND: train-v1-2
# ============================================================================

def cmd_train_v1_2(args):
    """Train v1.2 model on historical data."""
    version.print_version_banner("train_v1_2", model_version="v1.2")

    print(f"Training v1.2 model on seasons {args.start_season}-{args.end_season}...")
    print("NOTE: v1.2 training requires nfelo historical data and feature engineering.")
    print("This functionality is currently in ball_knower.datasets.v1_2")
    print("\nTo train v1.2, use the dataset modules directly:")
    print("  from ball_knower.datasets import v1_2")
    print("  dataset = v1_2.load_v1_2_dataset(start_season=2009, end_season=2024)")
    print("  # Then train your model on dataset")

    print(f"\n⚠ Training script not yet implemented in CLI.")
    print("   Model artifacts will be saved to:", paths.get_models_dir("v1.2"))

    return 1  # Not implemented yet


# ============================================================================
# SUBCOMMAND: backtest
# ============================================================================

def cmd_backtest(args):
    """Run backtest for specified model."""
    # Import run_backtests module
    from src import run_backtests

    version.print_version_banner("backtest", model_version=args.model)

    print(f"Backtesting {args.model} model...")
    print(f"  Seasons: {args.start_season}-{args.end_season}")
    print(f"  Edge threshold: {args.edge_threshold}")

    # Call the appropriate backtest function
    if args.model == 'v1.0':
        results = run_backtests.run_backtest_v1_0(
            args.start_season,
            args.end_season,
            args.edge_threshold
        )
    elif args.model == 'v1.2':
        results = run_backtests.run_backtest_v1_2(
            args.start_season,
            args.end_season,
            args.edge_threshold
        )
    else:
        print(f"Error: Unknown model '{args.model}'", file=sys.stderr)
        return 1

    # Determine output path
    if args.output is None:
        output_path = paths.get_backtest_results_path(
            args.model,
            args.start_season,
            args.end_season
        )
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results.to_csv(output_path, index=False)

    print(f"\n✓ Backtest complete: {len(results)} seasons analyzed")
    print(f"  Results saved to: {output_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(results.to_string(index=False))

    return 0


# ============================================================================
# SUBCOMMAND: predict
# ============================================================================

def cmd_predict(args):
    """Generate weekly predictions."""
    # Import run_weekly_predictions module
    from src import run_weekly_predictions

    version.print_version_banner("predict", model_version=args.model)

    print(f"Generating predictions for {args.season} Week {args.week}...")
    print(f"  Model: {args.model}")

    # Load weekly data
    team_ratings, matchups = run_weekly_predictions.load_weekly_data(args.season, args.week)

    # Build feature matrix
    feature_df = run_weekly_predictions.build_feature_matrix(matchups, team_ratings)

    # Generate predictions
    predictions = run_weekly_predictions.generate_predictions(
        feature_df,
        model_version=args.model
    )

    # Save predictions
    predictions_df, output_file = run_weekly_predictions.save_predictions(
        predictions,
        season=args.season,
        week=args.week,
        output_path=args.output
    )

    # Print summary
    run_weekly_predictions.print_summary(predictions_df)

    print(f"\n✓ Predictions complete")
    print(f"  Output: {output_file}")

    return 0


# ============================================================================
# SUBCOMMAND: export-predictiontracker
# ============================================================================

def cmd_export_predictiontracker(args):
    """Export predictions to PredictionTracker format."""
    version.print_version_banner("export_predictiontracker", model_version=args.model)

    print(f"Exporting {args.model} predictions to PredictionTracker format...")
    print(f"  Seasons: {args.start_season}-{args.end_season}")
    print("NOTE: PredictionTracker export requires historical predictions.")
    print("This functionality is not yet implemented.")

    # Determine output path
    if args.output is None:
        output_path = paths.get_predictiontracker_export_path(
            args.model,
            args.start_season,
            args.end_season
        )
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n⚠ Export script not yet implemented in CLI.")
    print("   Output would be saved to:", output_path)

    return 1  # Not implemented yet


# ============================================================================
# CLI SETUP
# ============================================================================

def create_parser():
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description='Ball Knower Build System - Train, backtest, predict, and export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train v1.2 model on historical data
  %(prog)s train-v1-2 --start-season 2009 --end-season 2024

  # Backtest v1.2 model with 0.5 point edge threshold
  %(prog)s backtest --model v1.2 --start-season 2019 --end-season 2024 --edge-threshold 0.5

  # Generate predictions for current week
  %(prog)s predict --model v1.2 --season 2025 --week 11

  # Export to PredictionTracker format
  %(prog)s export-predictiontracker --model v1.2 --start-season 2019 --end-season 2024
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # ========================================
    # train-v1-2 subcommand
    # ========================================
    parser_train = subparsers.add_parser(
        'train-v1-2',
        help='Train v1.2 model on historical data'
    )
    parser_train.add_argument(
        '--start-season',
        type=int,
        default=2009,
        help='Start season for training (default: 2009)'
    )
    parser_train.add_argument(
        '--end-season',
        type=int,
        default=2024,
        help='End season for training (default: 2024)'
    )
    parser_train.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for model artifacts (default: output/models/v1_2/)'
    )
    parser_train.set_defaults(func=cmd_train_v1_2)

    # ========================================
    # backtest subcommand
    # ========================================
    parser_backtest = subparsers.add_parser(
        'backtest',
        help='Run backtest on historical data'
    )
    parser_backtest.add_argument(
        '--model',
        type=str,
        choices=['v1.0', 'v1.2'],
        required=True,
        help='Model version to backtest'
    )
    parser_backtest.add_argument(
        '--start-season',
        type=int,
        required=True,
        help='Start season for backtest'
    )
    parser_backtest.add_argument(
        '--end-season',
        type=int,
        required=True,
        help='End season for backtest'
    )
    parser_backtest.add_argument(
        '--edge-threshold',
        type=float,
        default=0.0,
        help='Minimum edge threshold for betting (default: 0.0)'
    )
    parser_backtest.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: output/backtests/{model}/backtest_{start}_{end}.csv)'
    )
    parser_backtest.set_defaults(func=cmd_backtest)

    # ========================================
    # predict subcommand
    # ========================================
    parser_predict = subparsers.add_parser(
        'predict',
        help='Generate weekly predictions'
    )
    parser_predict.add_argument(
        '--model',
        type=str,
        choices=['v1.0', 'v1.1'],
        default='v1.1',
        help='Model version to use (default: v1.1)'
    )
    parser_predict.add_argument(
        '--season',
        type=int,
        required=True,
        help='NFL season year (e.g., 2025)'
    )
    parser_predict.add_argument(
        '--week',
        type=int,
        required=True,
        help='Week number (1-18)'
    )
    parser_predict.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: output/predictions_{season}_week_{week}.csv)'
    )
    parser_predict.set_defaults(func=cmd_predict)

    # ========================================
    # export-predictiontracker subcommand
    # ========================================
    parser_export = subparsers.add_parser(
        'export-predictiontracker',
        help='Export predictions to PredictionTracker format'
    )
    parser_export.add_argument(
        '--model',
        type=str,
        choices=['v1.0', 'v1.2'],
        required=True,
        help='Model version to export'
    )
    parser_export.add_argument(
        '--start-season',
        type=int,
        required=True,
        help='Start season for export'
    )
    parser_export.add_argument(
        '--end-season',
        type=int,
        required=True,
        help='End season for export'
    )
    parser_export.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: output/predictiontracker/{model}_{start}_{end}.csv)'
    )
    parser_export.set_defaults(func=cmd_export_predictiontracker)

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Call the appropriate subcommand function
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
