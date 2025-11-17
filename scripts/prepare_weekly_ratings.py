#!/usr/bin/env python3
"""
Data Ingestion Script for Ball Knower

Converts raw nfelo/Substack downloads into canonical category-first file names
for use with the Ball Knower modeling pipeline.

Usage:
    python scripts/prepare_weekly_ratings.py \
        --season 2025 \
        --week 11 \
        --source-dir raw_downloads/ \
        --nfelo-power nfelo_power_ratings_2025_week_11.csv \
        --nfelo-epa nfelo_epa_tiers_off_def_2025_week_11.csv \
        --substack-power substack_ratings_week11.csv

This will create:
    data/current_season/power_ratings_nfelo_2025_week_11.csv
    data/current_season/epa_tiers_nfelo_2025_week_11.csv
    data/current_season/power_ratings_substack_2025_week_11.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import shutil
from typing import Optional, Dict


# Expected columns for validation
EXPECTED_COLUMNS = {
    'power_ratings_nfelo': ['team', 'nfelo'],
    'epa_tiers_nfelo': ['team'],  # Can have 'EPA/Play' or 'epa_off'
    'strength_of_schedule_nfelo': ['team'],
    'power_ratings_substack': ['team', 'Ovr.'],
    'qb_epa_substack': ['team'],
    'weekly_projections_ppg_substack': [],  # Flexible, just check it loads
}


def validate_csv(file_path: Path, expected_cols: list) -> bool:
    """
    Validate that a CSV file exists and contains expected columns.

    Args:
        file_path: Path to CSV file
        expected_cols: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if not file_path.exists():
        print(f"  ❌ File not found: {file_path}")
        return False

    try:
        df = pd.read_csv(file_path, nrows=5)

        if not expected_cols:
            # No specific columns required, just check it loads
            print(f"  ✓ File loads successfully: {file_path.name}")
            return True

        # Create case-insensitive column lookup
        cols_lower = {col.lower(): col for col in df.columns}

        missing_cols = []
        for col in expected_cols:
            col_lower = col.lower()

            # For EPA columns, check for both old and new names
            if col == 'epa_off':
                if 'epa_off' not in cols_lower and 'epa/play' not in cols_lower:
                    missing_cols.append(col + ' (or EPA/Play)')
            elif col == 'epa_def':
                if 'epa_def' not in cols_lower and 'epa/play against' not in cols_lower:
                    missing_cols.append(col + ' (or EPA/Play Against)')
            elif col_lower not in cols_lower:
                missing_cols.append(col)

        if missing_cols:
            print(f"  ⚠️  Warning: Missing columns in {file_path.name}: {missing_cols}")
            print(f"      Available columns: {list(df.columns)}")
            return True  # Still proceed, just warn

        print(f"  ✓ Validated: {file_path.name} ({len(df.columns)} columns)")
        return True

    except Exception as e:
        print(f"  ❌ Error reading {file_path.name}: {e}")
        return False


def copy_and_rename(
    source_path: Path,
    dest_dir: Path,
    canonical_name: str,
    expected_cols: list
) -> bool:
    """
    Copy a file to the destination with canonical naming, after validation.

    Args:
        source_path: Source file path
        dest_dir: Destination directory
        canonical_name: Target file name (e.g., 'power_ratings_nfelo_2025_week_11.csv')
        expected_cols: List of required columns for validation

    Returns:
        True if successful, False otherwise
    """
    if not source_path.exists():
        print(f"  ⚠️  Skipping: {source_path.name} (not found)")
        return False

    # Validate before copying
    if not validate_csv(source_path, expected_cols):
        return False

    dest_path = dest_dir / canonical_name

    try:
        shutil.copy2(source_path, dest_path)
        print(f"  ✓ Copied: {source_path.name} → {canonical_name}")
        return True
    except Exception as e:
        print(f"  ❌ Error copying {source_path.name}: {e}")
        return False


def prepare_weekly_ratings(
    season: int,
    week: int,
    source_dir: Path,
    dest_dir: Path,
    nfelo_power: Optional[str] = None,
    nfelo_epa: Optional[str] = None,
    nfelo_sos: Optional[str] = None,
    substack_power: Optional[str] = None,
    substack_qb: Optional[str] = None,
    substack_proj: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Prepare weekly ratings by copying and renaming raw files to canonical names.

    Args:
        season: NFL season year
        week: Week number
        source_dir: Directory containing raw download files
        dest_dir: Destination directory (typically data/current_season/)
        nfelo_power: Filename for nfelo power ratings
        nfelo_epa: Filename for nfelo EPA tiers
        nfelo_sos: Filename for nfelo strength of schedule
        substack_power: Filename for Substack power ratings
        substack_qb: Filename for Substack QB EPA
        substack_proj: Filename for Substack weekly projections

    Returns:
        Dictionary mapping canonical names to success status
    """
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PREPARING WEEKLY RATINGS - {season} Week {week}")
    print(f"{'='*70}")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}\n")

    results = {}

    # Map of (canonical_name, source_filename, expected_columns)
    files_to_process = [
        (f"power_ratings_nfelo_{season}_week_{week}.csv", nfelo_power, EXPECTED_COLUMNS['power_ratings_nfelo']),
        (f"epa_tiers_nfelo_{season}_week_{week}.csv", nfelo_epa, EXPECTED_COLUMNS['epa_tiers_nfelo']),
        (f"strength_of_schedule_nfelo_{season}_week_{week}.csv", nfelo_sos, EXPECTED_COLUMNS['strength_of_schedule_nfelo']),
        (f"power_ratings_substack_{season}_week_{week}.csv", substack_power, EXPECTED_COLUMNS['power_ratings_substack']),
        (f"qb_epa_substack_{season}_week_{week}.csv", substack_qb, EXPECTED_COLUMNS['qb_epa_substack']),
        (f"weekly_projections_ppg_substack_{season}_week_{week}.csv", substack_proj, EXPECTED_COLUMNS['weekly_projections_ppg_substack']),
    ]

    for canonical_name, source_filename, expected_cols in files_to_process:
        if source_filename is None:
            print(f"⊘  Skipping: {canonical_name} (not specified)")
            results[canonical_name] = None  # None means not specified
            continue

        source_path = source_dir / source_filename
        success = copy_and_rename(source_path, dest_dir, canonical_name, expected_cols)
        results[canonical_name] = success  # True/False means specified and success/failure

    # Summary
    print(f"\n{'='*70}")
    successful = sum(1 for v in results.values() if v)
    total = len([v for v in results.values() if v is not None])
    print(f"SUMMARY: {successful}/{len(files_to_process)} files processed successfully")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Prepare weekly ratings data for Ball Knower modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Week 11 nfelo data
  python scripts/prepare_weekly_ratings.py --season 2025 --week 11 \\
      --source-dir raw_downloads/ \\
      --nfelo-power nfelo_power_ratings_2025_week_11.csv \\
      --nfelo-epa nfelo_epa_tiers_2025_week_11.csv

  # Prepare full week with all sources
  python scripts/prepare_weekly_ratings.py --season 2025 --week 10 \\
      --source-dir ~/Downloads/nfl_data/ \\
      --nfelo-power power_ratings.csv \\
      --nfelo-epa epa_tiers.csv \\
      --nfelo-sos sos.csv \\
      --substack-power substack_ratings.csv \\
      --substack-qb qb_epa.csv \\
      --substack-proj weekly_proj.csv
        """
    )

    parser.add_argument('--season', type=int, required=True, help='NFL season year (e.g., 2025)')
    parser.add_argument('--week', type=int, required=True, help='NFL week number (1-18)')
    parser.add_argument('--source-dir', type=str, default='raw_downloads',
                        help='Directory containing raw downloaded files (default: raw_downloads/)')
    parser.add_argument('--dest-dir', type=str, default=None,
                        help='Destination directory (default: data/current_season/)')

    # nfelo files
    parser.add_argument('--nfelo-power', type=str, help='nfelo power ratings CSV filename')
    parser.add_argument('--nfelo-epa', type=str, help='nfelo EPA tiers CSV filename')
    parser.add_argument('--nfelo-sos', type=str, help='nfelo strength of schedule CSV filename')

    # Substack files
    parser.add_argument('--substack-power', type=str, help='Substack power ratings CSV filename')
    parser.add_argument('--substack-qb', type=str, help='Substack QB EPA CSV filename')
    parser.add_argument('--substack-proj', type=str, help='Substack weekly projections CSV filename')

    args = parser.parse_args()

    # Resolve directories
    source_dir = Path(args.source_dir).resolve()

    if args.dest_dir is None:
        # Default to repo_root/data/current_season
        repo_root = Path(__file__).resolve().parents[1]
        dest_dir = repo_root / 'data' / 'current_season'
    else:
        dest_dir = Path(args.dest_dir).resolve()

    if not source_dir.exists():
        print(f"❌ Error: Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Check if any files were specified
    file_args = [args.nfelo_power, args.nfelo_epa, args.nfelo_sos,
                 args.substack_power, args.substack_qb, args.substack_proj]

    if not any(file_args):
        print("❌ Error: No files specified. Use --nfelo-power, --nfelo-epa, etc.")
        parser.print_help()
        sys.exit(1)

    # Run preparation
    results = prepare_weekly_ratings(
        season=args.season,
        week=args.week,
        source_dir=source_dir,
        dest_dir=dest_dir,
        nfelo_power=args.nfelo_power,
        nfelo_epa=args.nfelo_epa,
        nfelo_sos=args.nfelo_sos,
        substack_power=args.substack_power,
        substack_qb=args.substack_qb,
        substack_proj=args.substack_proj,
    )

    # Exit with error if any files that were specified failed to process
    # (Don't fail for files that simply weren't specified - those have v=None)
    failed_files = [k for k, v in results.items() if v is False]

    if failed_files:
        print(f"❌ Error: Some specified files failed to process: {failed_files}")
        sys.exit(1)

    successful_files = [k for k, v in results.items() if v is True]
    if not successful_files:
        print("❌ Error: No files were successfully processed")
        sys.exit(1)

    print("✓ Data preparation complete!")


if __name__ == '__main__':
    main()
