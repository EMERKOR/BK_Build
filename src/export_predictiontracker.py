#!/usr/bin/env python3
"""
PredictionTracker CSV Export CLI

Converts Ball Knower backtest results into PredictionTracker-compatible CSV format
for external benchmarking and model tracking.

Usage Examples:
    # Export v1.2 model backtests for 2019-2024
    python src/export_predictiontracker.py \
        --model-version v1.2 \
        --start-season 2019 \
        --end-season 2024 \
        --output output/predictiontracker/v1_2_2019_2024.csv

    # Export v1.0 model with custom input file
    python src/export_predictiontracker.py \
        --model-version v1.0 \
        --start-season 2020 \
        --end-season 2023 \
        --input output/backtests/v1.0/custom_backtest.csv \
        --output output/predictiontracker/v1_0_custom.csv

    # Export without actual margins (predictions only)
    python src/export_predictiontracker.py \
        --model-version v1.2 \
        --start-season 2024 \
        --end-season 2024 \
        --no-actuals \
        --output output/predictiontracker/v1_2_2024_predictions_only.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src import config
from ball_knower.export import predictiontracker


def main():
    parser = argparse.ArgumentParser(
        description='Export Ball Knower backtests to PredictionTracker CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model-version',
        type=str,
        choices=['v1.0', 'v1.2'],
        required=True,
        help='Model version to export (v1.0 or v1.2)'
    )

    parser.add_argument(
        '--start-season',
        type=int,
        required=True,
        help='Start season year (e.g., 2019)'
    )

    parser.add_argument(
        '--end-season',
        type=int,
        required=True,
        help='End season year (e.g., 2024)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Custom input backtest CSV path (default: auto-locate from standard path)'
    )

    parser.add_argument(
        '--no-actuals',
        action='store_true',
        help='Exclude actual_margin column (for forward predictions)'
    )

    args = parser.parse_args()

    # Validate season range
    if args.start_season > args.end_season:
        print(f"Error: start-season ({args.start_season}) cannot be greater than "
              f"end-season ({args.end_season})", file=sys.stderr)
        return 1

    print(f"\n{'='*70}")
    print(f"PREDICTIONTRACKER EXPORT")
    print(f"{'='*70}")
    print(f"Model version: {args.model_version}")
    print(f"Seasons: {args.start_season}-{args.end_season}")
    print(f"Output: {args.output}")

    # =========================================================================
    # LOCATE INPUT FILE
    # =========================================================================
    if args.input is not None:
        # Use user-specified input file
        input_path = Path(args.input)
    else:
        # Auto-locate from standard backtest output path
        input_path = (
            config.OUTPUT_DIR /
            'backtests' /
            args.model_version /
            f"backtest_{args.model_version}_{args.start_season}_{args.end_season}.csv"
        )

    if not input_path.exists():
        print(f"\nError: Backtest file not found at {input_path}", file=sys.stderr)
        print("\nTo generate the backtest file, run:", file=sys.stderr)
        print(f"  python src/run_backtests.py \\", file=sys.stderr)
        print(f"    --model {args.model_version} \\", file=sys.stderr)
        print(f"    --start-season {args.start_season} \\", file=sys.stderr)
        print(f"    --end-season {args.end_season}", file=sys.stderr)
        return 1

    print(f"Input: {input_path}")

    # =========================================================================
    # LOAD BACKTEST DATA
    # =========================================================================
    try:
        backtest_df = pd.read_csv(input_path)
        print(f"\nLoaded {len(backtest_df)} games from backtest file")
    except Exception as e:
        print(f"\nError loading backtest file: {e}", file=sys.stderr)
        return 1

    # =========================================================================
    # CONVERT TO PREDICTIONTRACKER FORMAT
    # =========================================================================
    try:
        pt_df = predictiontracker.export_predictiontracker_format(
            backtest_df,
            model_version=args.model_version,
            include_actuals=not args.no_actuals
        )
        print(f"Converted to PredictionTracker format: {len(pt_df)} rows")
    except Exception as e:
        print(f"\nError converting to PredictionTracker format: {e}", file=sys.stderr)
        return 1

    # =========================================================================
    # VALIDATE FORMAT
    # =========================================================================
    try:
        predictiontracker.validate_predictiontracker_format(pt_df)
        print("✓ Validation passed")
    except ValueError as e:
        print(f"\nWarning: Validation failed: {e}", file=sys.stderr)
        print("Proceeding anyway...", file=sys.stderr)

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pt_df.to_csv(output_path, index=False)
        print(f"\n✓ PredictionTracker CSV saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving output file: {e}", file=sys.stderr)
        return 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("EXPORT SUMMARY")
    print(f"{'='*70}")
    print(f"Games exported: {len(pt_df)}")
    print(f"Columns: {', '.join(pt_df.columns)}")
    print(f"Date range: {pt_df['date'].min()} to {pt_df['date'].max()}")

    if 'actual_margin' in pt_df.columns:
        print(f"Includes actuals: Yes")
    else:
        print(f"Includes actuals: No (predictions only)")

    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
