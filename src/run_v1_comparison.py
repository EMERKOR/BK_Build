#!/usr/bin/env python3
"""
Ball Knower v1.x Model Comparison CLI

Compares v1.0 and v1.2 models on accuracy and betting performance metrics.

Usage:
    # Compare on recent seasons with default edge threshold
    python src/run_v1_comparison.py --test-start-season 2020 --test-end-season 2023

    # Use custom edge threshold
    python src/run_v1_comparison.py --test-start-season 2018 --test-end-season 2023 --edge-threshold 2.0

    # Use specific seasons
    python src/run_v1_comparison.py --test-seasons 2020 2021 2022
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.benchmarks.v1_comparison import compare_v1_models


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as percentage."""
    if value is None:
        return 'N/A'
    return f"{value * 100:.{decimals}f}%"


def format_units(value: float, decimals: int = 1, show_sign: bool = True) -> str:
    """Format units with +/- sign."""
    if value is None:
        return 'N/A'
    sign = '+' if value > 0 and show_sign else ''
    return f"{sign}{value:.{decimals}f}"


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """
    Print formatted comparison table.

    Includes both accuracy and betting metrics.
    """
    print("\n" + "=" * 100)
    print("BALL KNOWER v1.x MODEL COMPARISON")
    print("=" * 100)

    # Header
    header = (
        f"{'Model':<8} | "
        f"{'Games':<6} | "
        f"{'MAE Spread':<11} | "
        f"{'Hit <3pt':<8} | "
        f"{'Hit <7pt':<8} | "
        f"{'ATS Win%':<9} | "
        f"{'Bets':<5} | "
        f"{'Units':<7} | "
        f"{'ROI':<7}"
    )
    print(header)
    print("-" * 100)

    # Rows
    for result in results:
        model = result['model_name']
        n_games = result['n_games']
        mae_spread = result['mae_spread']
        hit_3 = result['hit_rate_spread_within_3']
        hit_7 = result['hit_rate_spread_within_7']

        ats = result['ats']
        ats_win_rate = ats['win_rate']
        n_bets = ats['n_bets']
        units = ats['units_won']
        roi = ats['roi']

        # Format values
        mae_str = f"{mae_spread:.2f}" if mae_spread is not None else "N/A"
        hit_3_str = format_percentage(hit_3)
        hit_7_str = format_percentage(hit_7)
        ats_win_str = format_percentage(ats_win_rate)
        units_str = format_units(units)
        roi_str = format_percentage(roi)

        row = (
            f"{model:<8} | "
            f"{n_games:<6} | "
            f"{mae_str:<11} | "
            f"{hit_3_str:<8} | "
            f"{hit_7_str:<8} | "
            f"{ats_win_str:<9} | "
            f"{n_bets:<5} | "
            f"{units_str:<7} | "
            f"{roi_str:<7}"
        )
        print(row)

    print("=" * 100)

    # Detailed ATS stats
    print("\nDETAILED ATS STATISTICS")
    print("-" * 100)

    detail_header = (
        f"{'Model':<8} | "
        f"{'Edge Thresh':<11} | "
        f"{'Avg Edge':<9} | "
        f"{'Wins':<5} | "
        f"{'Losses':<7} | "
        f"{'Pushes':<7} | "
        f"{'Win%':<7}"
    )
    print(detail_header)
    print("-" * 100)

    for result in results:
        model = result['model_name']
        ats = result['ats']

        edge_thresh = ats['edge_threshold']
        avg_edge = ats['avg_edge']
        wins = ats['wins']
        losses = ats['losses']
        pushes = ats['pushes']
        win_rate = ats['win_rate']

        avg_edge_str = f"{avg_edge:.2f}" if avg_edge is not None else "N/A"
        win_rate_str = format_percentage(win_rate)

        row = (
            f"{model:<8} | "
            f"{edge_thresh:<11.1f} | "
            f"{avg_edge_str:<9} | "
            f"{wins:<5} | "
            f"{losses:<7} | "
            f"{pushes:<7} | "
            f"{win_rate_str:<7}"
        )
        print(row)

    print("=" * 100)
    print()


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print quick summary and interpretation."""
    if len(results) < 2:
        return

    v1_0 = results[0]
    v1_2 = results[1]

    print("\nKEY INSIGHTS")
    print("-" * 100)

    # Accuracy comparison
    if v1_2['mae_spread'] and v1_0['mae_spread']:
        mae_improvement = v1_0['mae_spread'] - v1_2['mae_spread']
        if mae_improvement > 0:
            print(f"• Accuracy: v1.2 is {mae_improvement:.2f} points better (lower MAE) than v1.0")
        else:
            print(f"• Accuracy: v1.2 is {abs(mae_improvement):.2f} points worse (higher MAE) than v1.0")

    # Betting comparison
    v1_0_roi = v1_0['ats']['roi']
    v1_2_roi = v1_2['ats']['roi']

    if v1_0_roi is not None and v1_2_roi is not None:
        roi_diff = (v1_2_roi - v1_0_roi) * 100
        print(f"• Betting: v1.2 ROI is {roi_diff:+.2f}pp vs v1.0")

    # Units comparison
    v1_0_units = v1_0['ats']['units_won']
    v1_2_units = v1_2['ats']['units_won']
    units_diff = v1_2_units - v1_0_units

    print(f"• Profit: v1.2 made {units_diff:+.1f} more units than v1.0")

    # Bet volume
    v1_0_bets = v1_0['ats']['n_bets']
    v1_2_bets = v1_2['ats']['n_bets']

    print(f"• Volume: v1.0 made {v1_0_bets} bets, v1.2 made {v1_2_bets} bets")

    print("=" * 100)
    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare Ball Knower v1.x models on accuracy and betting performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare on 2020-2023 with 1.0 point edge threshold
  python src/run_v1_comparison.py --test-start-season 2020 --test-end-season 2023 --edge-threshold 1.0

  # Compare on specific seasons with 2.0 point edge threshold
  python src/run_v1_comparison.py --test-seasons 2018 2019 2020 --edge-threshold 2.0

  # Quick test on single season
  python src/run_v1_comparison.py --test-seasons 2023 --edge-threshold 0.5
        """
    )

    # Season selection (mutually exclusive)
    season_group = parser.add_mutually_exclusive_group()

    season_group.add_argument(
        '--test-seasons',
        type=int,
        nargs='+',
        help='Specific seasons to test (e.g., 2020 2021 2022)'
    )

    season_group.add_argument(
        '--test-start-season',
        type=int,
        help='Start season for range (use with --test-end-season)'
    )

    parser.add_argument(
        '--test-end-season',
        type=int,
        help='End season for range (use with --test-start-season)'
    )

    # Edge threshold
    parser.add_argument(
        '--edge-threshold',
        type=float,
        default=1.0,
        help='Minimum edge (in points) to place ATS bet (default: 1.0)'
    )

    # Data source
    parser.add_argument(
        '--nfelo-url',
        type=str,
        default=None,
        help='Custom nfelo data URL (default: greerreNFL GitHub)'
    )

    args = parser.parse_args()

    # Determine test seasons
    if args.test_seasons:
        test_seasons = args.test_seasons
    elif args.test_start_season and args.test_end_season:
        if args.test_start_season > args.test_end_season:
            print("Error: --test-start-season must be <= --test-end-season", file=sys.stderr)
            return 1
        test_seasons = list(range(args.test_start_season, args.test_end_season + 1))
    elif args.test_start_season or args.test_end_season:
        print("Error: Must specify both --test-start-season and --test-end-season", file=sys.stderr)
        return 1
    else:
        # Default to recent seasons
        test_seasons = [2020, 2021, 2022, 2023]
        print(f"Using default test seasons: {test_seasons}")

    # Run comparison
    try:
        results = compare_v1_models(
            test_seasons=test_seasons,
            edge_threshold=args.edge_threshold,
            nfelo_url=args.nfelo_url
        )

        # Print results
        print_comparison_table(results)
        print_summary(results)

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nMake sure you've trained the v1.2 model first:", file=sys.stderr)
        print("  python ball_knower/datasets/v1_2.py", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
