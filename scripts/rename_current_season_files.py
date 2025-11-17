#!/usr/bin/env python3
"""
Phase 4: Rename Current Season CSV Files to Category-First Convention

This script renames data files from provider-first to category-first naming.

Old pattern: {provider}_{category}_{season}_week_{week}.csv
New pattern: {category}_{provider}_{season}_week_{week}.csv

Author: Ball Knower Team
Date: 2024-11-17
"""

from pathlib import Path
import sys

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "current_season"

# Rename mapping: old filename -> new filename
# Only includes files actively used by ball_knower.io.loaders
RENAME_MAP = {
    # NFelo files
    "nfelo_power_ratings_2025_week_11.csv": "power_ratings_nfelo_2025_week_11.csv",
    "nfelo_epa_tiers_off_def_2025_week_11.csv": "epa_tiers_nfelo_2025_week_11.csv",
    "nfelo_strength_of_schedule_2025_week_11.csv": "strength_of_schedule_nfelo_2025_week_11.csv",

    # Substack files
    "substack_power_ratings_2025_week_11.csv": "power_ratings_substack_2025_week_11.csv",
    "substack_qb_epa_2025_week_11.csv": "qb_epa_substack_2025_week_11.csv",
    "substack_weekly_proj_ppg_2025_week_11.csv": "weekly_projections_ppg_substack_2025_week_11.csv",
}

# Files to skip (not actively used by main loaders, but keep for reference)
SKIP_FILES = [
    "nfelo_nfl_receiving_leaders_2025_week_11.csv",
    "nfelo_nfl_win_totals_2025_week_11 (1).csv",
    "nfelo_qb_rankings_2025_week_11.csv",
    "substack_weekly_proj_elo_2025_week_11.csv",
]


def main():
    """Execute the rename operation."""
    print("=" * 70)
    print("PHASE 4: RENAME CURRENT SEASON FILES TO CATEGORY-FIRST")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"\n✗ ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    print(f"\n{len(RENAME_MAP)} files to rename:")
    print("-" * 70)

    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for old_name, new_name in RENAME_MAP.items():
        old_path = DATA_DIR / old_name
        new_path = DATA_DIR / new_name

        # Check if old file exists
        if not old_path.exists():
            print(f"⚠️  SKIP: {old_name} (not found)")
            skipped_count += 1
            continue

        # Check if new file already exists
        if new_path.exists():
            print(f"⚠️  SKIP: {new_name} (already exists)")
            skipped_count += 1
            continue

        # Perform rename
        try:
            old_path.rename(new_path)
            print(f"✓  {old_name}")
            print(f"   → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"✗  ERROR renaming {old_name}: {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Renamed: {renamed_count} files")
    print(f"⚠️  Skipped: {skipped_count} files")
    print(f"✗ Errors:  {error_count} files")

    if len(SKIP_FILES) > 0:
        print(f"\nℹ️  {len(SKIP_FILES)} files intentionally not renamed (not used by main loaders):")
        for skip_file in SKIP_FILES:
            if (DATA_DIR / skip_file).exists():
                print(f"   - {skip_file}")

    print("\n" + "=" * 70)
    print("CURRENT FILES IN data/current_season/:")
    print("=" * 70)

    all_files = sorted(DATA_DIR.glob("*.csv"))
    for file in all_files:
        marker = "✓" if any(file.name == new for new in RENAME_MAP.values()) else " "
        print(f"{marker} {file.name}")

    print("\n✓ = Category-first (new convention)")
    print("  = Provider-first (old convention) or reference file")

    if error_count > 0:
        print(f"\n✗ Rename completed with {error_count} errors")
        sys.exit(1)
    else:
        print("\n✓ Rename completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
