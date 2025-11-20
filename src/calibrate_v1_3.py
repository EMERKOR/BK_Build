#!/usr/bin/env python3
"""
v1.3 Model Calibration Script

Generates calibration parameters for v1.3 model from backtest results.

Usage:
    # From existing backtest CSV:
    python src/calibrate_v1_3.py --backtest-path output/backtests/v1_3/backtest_2013_2018.csv

    # Run backtest and calibrate:
    python src/calibrate_v1_3.py --start-season 2013 --end-season 2018

    # Custom output path:
    python src/calibrate_v1_3.py --start-season 2013 --end-season 2018 \
        --output output/models/v1_3/calibration_2013_2018.json

Output:
    JSON file with calibration parameters:
        - mean_error (bias)
        - MAE and RMSE
        - Edge bin thresholds
        - Number of seasons and games
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.evaluation import calibration_v1_3
from ball_knower.utils import paths, version
from src import run_backtests


def main():
    """Main calibration script."""
    parser = argparse.ArgumentParser(
        description='Generate v1.3 model calibration parameters from backtest results'
    )

    parser.add_argument(
        '--backtest-path',
        type=str,
        default=None,
        help='Path to existing backtest CSV (if omitted, will run backtest)'
    )

    parser.add_argument(
        '--start-season',
        type=int,
        default=None,
        help='Start season for calibration (required if --backtest-path not provided)'
    )

    parser.add_argument(
        '--end-season',
        type=int,
        default=None,
        help='End season for calibration (required if --backtest-path not provided)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON path (default: output/models/v1_3/calibration_v1_3.json)'
    )

    parser.add_argument(
        '--edge-bins',
        type=str,
        default='0.5,1.0,1.5,2.0,2.5,3.0',
        help='Comma-separated edge bin thresholds (default: 0.5,1.0,1.5,2.0,2.5,3.0)'
    )

    args = parser.parse_args()

    # Print version banner
    version.print_version_banner("calibrate_v1_3", model_version="v1.3")

    # Parse edge bins
    edge_bins = [float(x) for x in args.edge_bins.split(',')]

    print("\n" + "=" * 80)
    print("v1.3 MODEL CALIBRATION")
    print("=" * 80)

    # Step 1: Get backtest results
    if args.backtest_path:
        # Load existing backtest CSV
        backtest_path = Path(args.backtest_path)
        if not backtest_path.exists():
            print(f"\nError: Backtest file not found at {backtest_path}", file=sys.stderr)
            return 1

        print(f"\n[1/3] Loading backtest results from {backtest_path}...")
        backtest_df = pd.read_csv(backtest_path)
        print(f"  ✓ Loaded {len(backtest_df)} season results")

    else:
        # Run backtest
        if args.start_season is None or args.end_season is None:
            print("\nError: Must provide --start-season and --end-season if --backtest-path not provided",
                  file=sys.stderr)
            return 1

        print(f"\n[1/3] Running v1.3 backtest ({args.start_season}-{args.end_season})...")
        print("  This may take a few minutes...")

        backtest_df = run_backtests.run_backtest_v1_3(
            start_season=args.start_season,
            end_season=args.end_season,
            edge_threshold=0.0  # Use 0.0 for calibration (include all games)
        )

        print(f"  ✓ Backtest complete: {len(backtest_df)} seasons")

    # Step 2: Compute calibration
    print(f"\n[2/3] Computing calibration parameters...")
    print(f"  Edge bins: {edge_bins}")

    try:
        calibration = calibration_v1_3.compute_v1_3_calibration(
            backtest_df,
            edge_bins=edge_bins
        )
        print(f"  ✓ Calibration computed")

    except Exception as e:
        print(f"\nError: Calibration failed: {e}", file=sys.stderr)
        return 1

    # Step 3: Save calibration
    if args.output is None:
        output_path = paths.get_models_dir("v1.3") / "calibration_v1_3.json"
    else:
        output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/3] Saving calibration to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"  ✓ Calibration saved")

    # Print summary
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"Model:           {calibration['model_version']}")
    print(f"Seasons:         {calibration['calibration_seasons']}")
    print(f"Total games:     {calibration['n_games_total']}")
    print(f"\nPerformance Metrics:")
    print(f"  Mean error:    {calibration['mean_error']:+.3f} points")
    print(f"  MAE:           {calibration['mae']:.3f} points")
    print(f"  RMSE:          {calibration['rmse']:.3f} points")
    print(f"\nEdge bins:       {', '.join(f'{b:.1f}' for b in calibration['edge_bins'])}")
    print(f"\nCalibration file: {output_path}")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
