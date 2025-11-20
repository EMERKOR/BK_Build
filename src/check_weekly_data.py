"""
Weekly Data Checklist Script

Validates the presence and schema of current-season data files for a given week.

Usage:
    python src/check_weekly_data.py --season 2025 --week 11

Exit codes:
    0 - All required files present and valid
    1 - Missing or invalid data files
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower import config
from ball_knower.io import loaders
from ball_knower.utils import version


def check_data_file(
    loader_func,
    provider: str,
    season: int,
    week: int,
    data_dir: Path = None,
    required: bool = True
) -> Tuple[bool, str]:
    """
    Check if a data file exists and validates against its schema.

    Args:
        loader_func: Loader function to use (e.g., loaders.load_power_ratings)
        provider: Data provider name (e.g., "nfelo", "substack")
        season: Season year
        week: Week number
        data_dir: Optional data directory
        required: Whether this file is required

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        df = loader_func(provider, season, week, data_dir)

        if df is None or len(df) == 0:
            return False, "Empty DataFrame"

        return True, f"Valid ({len(df)} rows)"

    except FileNotFoundError as e:
        if required:
            return False, "File not found"
        else:
            return False, "Optional file not found"

    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def check_weekly_data(season: int, week: int) -> Dict:
    """
    Check all expected weekly data files for presence and validity.

    Args:
        season: Season year
        week: Week number

    Returns:
        Dictionary with check results and summary
    """
    print(f"[Weekly Data Check] Season {season}, Week {week}")
    print("=" * 80)

    data_dir = config.CURRENT_SEASON_DATA_DIR

    checks = {
        "required": [],
        "optional": []
    }

    # Required nfelo files
    print("\nRequired nfelo files:")

    # Power ratings (required)
    success, message = check_data_file(
        loaders.load_power_ratings,
        "nfelo",
        season,
        week,
        data_dir,
        required=True
    )
    checks["required"].append(("power_ratings_nfelo", success))
    status = "✓" if success else "✗"
    print(f"  {status} power_ratings_nfelo_{season}_week_{week}.csv - {message}")

    # EPA tiers (required)
    success, message = check_data_file(
        loaders.load_epa_tiers,
        "nfelo",
        season,
        week,
        data_dir,
        required=True
    )
    checks["required"].append(("epa_tiers_nfelo", success))
    status = "✓" if success else "✗"
    print(f"  {status} epa_tiers_nfelo_{season}_week_{week}.csv - {message}")

    # Strength of schedule (optional but recommended)
    success, message = check_data_file(
        loaders.load_strength_of_schedule,
        "nfelo",
        season,
        week,
        data_dir,
        required=False
    )
    checks["optional"].append(("strength_of_schedule_nfelo", success))
    status = "✓" if success else "⚠"
    print(f"  {status} strength_of_schedule_nfelo_{season}_week_{week}.csv - {message}")

    # Required Substack files
    print("\nRequired Substack files:")

    # Power ratings (required)
    success, message = check_data_file(
        loaders.load_power_ratings,
        "substack",
        season,
        week,
        data_dir,
        required=True
    )
    checks["required"].append(("power_ratings_substack", success))
    status = "✓" if success else "✗"
    print(f"  {status} power_ratings_substack_{season}_week_{week}.csv - {message}")

    # QB EPA (optional but recommended)
    success, message = check_data_file(
        loaders.load_qb_epa,
        "substack",
        season,
        week,
        data_dir,
        required=False
    )
    checks["optional"].append(("qb_epa_substack", success))
    status = "✓" if success else "⚠"
    print(f"  {status} qb_epa_substack_{season}_week_{week}.csv - {message}")

    # Weekly projections (required for matchups)
    success, message = check_data_file(
        loaders.load_weekly_projections_ppg,
        "substack",
        season,
        week,
        data_dir,
        required=True
    )
    checks["required"].append(("weekly_projections_ppg_substack", success))
    status = "✓" if success else "✗"
    print(f"  {status} weekly_projections_ppg_substack_{season}_week_{week}.csv - {message}")

    # Summary
    print("\n" + "=" * 80)
    required_count = sum(1 for _, success in checks["required"] if success)
    required_total = len(checks["required"])
    optional_count = sum(1 for _, success in checks["optional"] if success)
    optional_total = len(checks["optional"])

    print(f"Summary:")
    print(f"  Required files: {required_count}/{required_total} ✓")
    print(f"  Optional files: {optional_count}/{optional_total} ✓")

    all_required_ok = all(success for _, success in checks["required"])

    if all_required_ok:
        print("\n✓ All required files present and valid")
        print("  Ready to run weekly predictions!")
        return_code = 0
    else:
        missing_required = [name for name, success in checks["required"] if not success]
        print(f"\n✗ Missing or invalid required files: {', '.join(missing_required)}")
        print("  Cannot run weekly predictions until these are provided")
        return_code = 1

    print("=" * 80)

    return {
        "season": season,
        "week": week,
        "checks": checks,
        "all_required_ok": all_required_ok,
        "return_code": return_code
    }


def main():
    """
    Main entry point for weekly data check script.
    """
    parser = argparse.ArgumentParser(
        description='Check weekly data files for presence and validity'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=config.CURRENT_SEASON,
        help=f'Season year (default: {config.CURRENT_SEASON})'
    )
    parser.add_argument(
        '--week',
        type=int,
        required=True,
        help='Week number'
    )

    args = parser.parse_args()

    # Print version banner
    version.print_version_banner("check_weekly_data")

    # Run checks
    results = check_weekly_data(args.season, args.week)

    return results["return_code"]


if __name__ == '__main__':
    sys.exit(main())
