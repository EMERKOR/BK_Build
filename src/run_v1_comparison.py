#!/usr/bin/env python3
"""
CLI script to compare Ball Knower v1.0, v1.2, and v1.3 models.

This script runs all three v1.x models on a shared test period and prints
a side-by-side comparison of their performance metrics.

Usage:
    python src/run_v1_comparison.py
    python src/run_v1_comparison.py --test-start-season 2020 --test-end-season 2023
    python src/run_v1_comparison.py --test-seasons 2022 2023 2024
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ball_knower.benchmarks.v1_comparison import compare_v1_models


def format_comparison_table(results: dict) -> str:
    """
    Format comparison results as a nice table.

    Parameters
    ----------
    results : dict
        Results from compare_v1_models()

    Returns
    -------
    str
        Formatted table string
    """
    models = results['models']

    # Build table rows
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("BALL KNOWER v1.x MODEL COMPARISON")
    lines.append("=" * 90)
    lines.append(f"\nTest Seasons: {results['test_seasons']}")
    lines.append(f"Total Games: {results['n_games']}")
    lines.append("\n" + "-" * 90)

    # Header
    header = f"{'Model':<8} | {'Spread MAE':<11} | {'Spread ±3%':<11} | {'Spread ±7%':<11} | {'Total MAE':<10} | {'Games':<6}"
    lines.append(header)
    lines.append("-" * 90)

    # Row for each model
    for model_name in ['v1.0', 'v1.2', 'v1.3']:
        m = models[model_name]

        if m['status'] != 'ok':
            # Error case
            lines.append(f"{model_name:<8} | ERROR: {m['status']}")
            continue

        # Spread MAE
        spread_mae = f"{m['mae_spread']:.2f}" if m['mae_spread'] is not None else "N/A"

        # Hit rates (format as percentages)
        spread_3 = f"{m.get('hit_rate_spread_within_3', 0):.1f}%" if 'hit_rate_spread_within_3' in m else "N/A"
        spread_7 = f"{m.get('hit_rate_spread_within_7', 0):.1f}%" if 'hit_rate_spread_within_7' in m else "N/A"

        # Total MAE (only v1.3 has this)
        total_mae = f"{m['mae_total']:.2f}" if m['mae_total'] is not None else "N/A"

        # Games
        n_games = m['n_games']

        # Format row
        row = f"{model_name:<8} | {spread_mae:<11} | {spread_3:<11} | {spread_7:<11} | {total_mae:<10} | {n_games:<6}"
        lines.append(row)

    lines.append("-" * 90)

    # Additional metrics for v1.3
    if models['v1.3']['status'] == 'ok':
        lines.append("\nv1.3 Additional Metrics:")
        v1_3 = models['v1.3']
        if 'mae_home_score' in v1_3:
            lines.append(f"  Home Score MAE: {v1_3['mae_home_score']:.2f}")
        if 'mae_away_score' in v1_3:
            lines.append(f"  Away Score MAE: {v1_3['mae_away_score']:.2f}")
        if 'hit_rate_total_within_3' in v1_3:
            lines.append(f"  Total ±3 pts:   {v1_3['hit_rate_total_within_3']:.1f}%")
        if 'hit_rate_total_within_7' in v1_3:
            lines.append(f"  Total ±7 pts:   {v1_3['hit_rate_total_within_7']:.1f}%")

    lines.append("\n" + "=" * 90)
    lines.append("Notes:")
    lines.append("  - Spread MAE: Mean absolute error for spread predictions (lower is better)")
    lines.append("  - Spread ±X%: Percentage of predictions within X points of actual margin")
    lines.append("  - Total MAE: Mean absolute error for total points (only v1.3)")
    lines.append("  - v1.0 and v1.2 predict spreads only; v1.3 predicts scores (derives spread/total)")
    lines.append("=" * 90 + "\n")

    return "\n".join(lines)


def main():
    """Run v1.x model comparison and print results."""
    parser = argparse.ArgumentParser(
        description='Compare Ball Knower v1.0, v1.2, and v1.3 models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models on default test seasons (2022-2024)
  python src/run_v1_comparison.py

  # Specify custom season range
  python src/run_v1_comparison.py --test-start-season 2020 --test-end-season 2023

  # Specify individual seasons
  python src/run_v1_comparison.py --test-seasons 2022 2023
        """
    )

    parser.add_argument(
        '--test-start-season',
        type=int,
        default=None,
        help='Start season for test period (e.g., 2020)'
    )
    parser.add_argument(
        '--test-end-season',
        type=int,
        default=None,
        help='End season for test period (e.g., 2023)'
    )
    parser.add_argument(
        '--test-seasons',
        type=int,
        nargs='+',
        default=None,
        help='Specific seasons to test on (e.g., 2022 2023 2024)'
    )
    parser.add_argument(
        '--v1-2-model-path',
        type=str,
        default=None,
        help='Path to v1.2 model JSON file (default: output/ball_knower_v1_2_model.json)'
    )

    args = parser.parse_args()

    # Determine test seasons
    test_seasons = None
    if args.test_seasons:
        test_seasons = args.test_seasons
    elif args.test_start_season and args.test_end_season:
        test_seasons = list(range(args.test_start_season, args.test_end_season + 1))
    # If neither specified, compare_v1_models will use default [2022, 2023, 2024]

    # Run comparison
    try:
        results = compare_v1_models(
            test_seasons=test_seasons,
            v1_2_model_path=args.v1_2_model_path
        )

        # Print formatted table
        print(format_comparison_table(results))

        # Return success
        return 0

    except Exception as e:
        print(f"\nERROR: Comparison failed")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
