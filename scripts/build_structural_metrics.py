#!/usr/bin/env python
"""
Build Structural Metrics from Play-by-Play Data

This script generates leak-free historical structural metrics (OSR, DSR, OLSI, CEA)
for all teams and weeks from 2009-2024 using nflverse play-by-play data.

Output:
- structural/structural_metrics_{season}.csv (one file per season)
- structural/structural_metrics_all.csv (all seasons combined)

Usage:
    python scripts/build_structural_metrics.py [--seasons 2020,2021,2022] [--pbp-path data/pbp/]

Note: This script requires nflverse play-by-play data. You can download it via:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data(years=[2009, 2010, ..., 2024])
    pbp.to_parquet('data/pbp/pbp_2009_2024.parquet')
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import argparse

from ball_knower.structural.build_structural_dataset import (
    build_structural_metrics_for_season,
    build_structural_metrics_all_seasons,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build structural metrics from play-by-play data"
    )
    parser.add_argument(
        '--seasons',
        type=str,
        default=None,
        help='Comma-separated list of seasons (e.g., "2020,2021,2022"). '
             'Default: all seasons from 2009-2024'
    )
    parser.add_argument(
        '--pbp-path',
        type=str,
        default='data/pbp/pbp_2009_2024.parquet',
        help='Path to play-by-play data file (parquet or CSV)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='structural',
        help='Output directory for structural metrics'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing'
    )

    args = parser.parse_args()

    # Parse seasons
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(',')]
    else:
        seasons = list(range(2009, 2025))

    print("="*70)
    print("Ball Knower Structural Metrics Builder")
    print("="*70)
    print()
    print(f"Seasons: {seasons}")
    print(f"PBP Path: {args.pbp_path}")
    print(f"Output Dir: {args.output_dir}")
    print()

    if args.dry_run:
        print("DRY RUN MODE - No files will be written")
        print()
        return

    # Check if PBP data exists
    pbp_path = Path(args.pbp_path)
    if not pbp_path.exists():
        print(f"ERROR: Play-by-play data file not found: {pbp_path}")
        print()
        print("To generate play-by-play data:")
        print("  1. Install nfl_data_py: pip install nfl_data_py")
        print("  2. Run:")
        print("      import nfl_data_py as nfl")
        print(f"      pbp = nfl.import_pbp_data(years={seasons})")
        print(f"      pbp.to_parquet('{pbp_path}')")
        print()
        sys.exit(1)

    # Load play-by-play data
    print("Loading play-by-play data...")
    if pbp_path.suffix == '.parquet':
        pbp_all = pd.read_parquet(pbp_path)
    elif pbp_path.suffix == '.csv':
        pbp_all = pd.read_csv(pbp_path)
    else:
        print(f"ERROR: Unsupported file format: {pbp_path.suffix}")
        sys.exit(1)

    print(f"  ✓ Loaded {len(pbp_all):,} plays")
    print()

    # Build structural metrics for all seasons
    result = build_structural_metrics_all_seasons(pbp_all, seasons)

    if len(result) == 0:
        print("ERROR: No structural metrics were generated")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual season files
    print()
    print("="*70)
    print("Saving season files...")
    for season in seasons:
        season_data = result[result['season'] == season]
        if len(season_data) > 0:
            season_file = output_dir / f"structural_metrics_{season}.csv"
            season_data.to_csv(season_file, index=False)
            print(f"  ✓ {season}: {len(season_data)} rows -> {season_file}")

    # Save combined file
    all_file = output_dir / "structural_metrics_all.csv"
    result.to_csv(all_file, index=False)
    print()
    print(f"  ✓ Combined: {len(result)} rows -> {all_file}")
    print()

    # Sanity checks
    print("="*70)
    print("Sanity Checks:")
    print()

    # Check for NaN values in structural_edge for weeks >= 4
    late_weeks = result[result['week'] >= 4]
    if len(late_weeks) > 0:
        nan_count = late_weeks['structural_edge'].isna().sum()
        nan_pct = 100 * nan_count / len(late_weeks)
        print(f"  NaN structural_edge (weeks >= 4): {nan_count} / {len(late_weeks)} ({nan_pct:.1f}%)")

    # Show sample data
    print()
    print("  Sample data (first 10 rows):")
    print()
    sample_cols = ['season', 'week', 'team', 'osr_z', 'dsr_z', 'olsi_z', 'cea_z', 'structural_edge']
    available_cols = [c for c in sample_cols if c in result.columns]
    print(result[available_cols].head(10).to_string(index=False))
    print()

    # Show summary statistics
    print()
    print("  Summary Statistics:")
    print()
    print(f"    Total rows: {len(result):,}")
    print(f"    Seasons: {sorted(result['season'].unique())}")
    print(f"    Weeks: {sorted(result['week'].unique())}")
    print(f"    Teams: {result['team'].nunique()}")
    print()
    print(f"    Structural Edge range: [{result['structural_edge'].min():.2f}, {result['structural_edge'].max():.2f}]")
    print(f"    Structural Edge mean: {result['structural_edge'].mean():.2f}")
    print(f"    Structural Edge std: {result['structural_edge'].std():.2f}")
    print()

    print("="*70)
    print("✓ Structural metrics build complete!")
    print("="*70)


if __name__ == "__main__":
    main()
