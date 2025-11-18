#!/usr/bin/env python3
"""
Ball Knower v2.0 - Training Frame Builder CLI

This script builds and saves the v2.0 training frame using the unified dataset pipeline.

Usage:
    python src/build_v2_training_frame.py
    python src/build_v2_training_frame.py --data-dir data --output-path output.parquet
    python src/build_v2_training_frame.py --no-market  # Exclude T2_MARKET features

Output:
    - Parquet file with canonical team-week training data
    - Console summary with feature counts by tier
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ball_knower.v2.datasets import (
    load_raw_sources,
    build_training_frame,
    get_feature_summary,
    validate_training_frame,
)
from ball_knower.v2.data_config import (
    STRUCTURAL_KEYS,
    TEAM_STRENGTH_FEATURES,
    MARKET_FEATURES,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Ball Knower v2.0 training frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default settings (includes market features)
  python src/build_v2_training_frame.py

  # Specify custom paths
  python src/build_v2_training_frame.py --data-dir ./data --output-path ./output/training.parquet

  # Exclude market features (T2)
  python src/build_v2_training_frame.py --no-market

  # Dry run (don't save file)
  python src/build_v2_training_frame.py --dry-run
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing input data files (default: data)',
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='data/v2_team_week_training.parquet',
        help='Output path for training frame (default: data/v2_team_week_training.parquet)',
    )

    parser.add_argument(
        '--include-market',
        dest='include_market',
        action='store_true',
        default=True,
        help='Include T2_MARKET features (default: True)',
    )

    parser.add_argument(
        '--no-market',
        dest='include_market',
        action='store_false',
        help='Exclude T2_MARKET features (only use T0+T1)',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build frame but do not save to disk',
    )

    parser.add_argument(
        '--show-columns',
        action='store_true',
        help='Print all column names grouped by tier',
    )

    return parser.parse_args()


def print_summary(df, feature_summary, args):
    """Print a detailed summary of the training frame."""
    print("\n" + "="*80)
    print(" BALL KNOWER v2.0 - TRAINING FRAME SUMMARY ".center(80, "="))
    print("="*80 + "\n")

    # Shape
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date range: {df['season'].min()}-{df['season'].max()}")
    print(f"Weeks: {df['week'].min()}-{df['week'].max()}")
    print(f"Unique teams: {df['team'].nunique()}")

    # Feature tier breakdown
    print("\n" + "-"*80)
    print("FEATURE TIER BREAKDOWN")
    print("-"*80)

    tier_info = [
        ('T0', 'Structural Keys', feature_summary['T0']),
        ('T1', 'Core Team Strength', feature_summary['T1']),
        ('T2', 'Market & Situational', feature_summary['T2']),
    ]

    for tier_code, tier_name, tier_cols in tier_info:
        count = len(tier_cols)
        print(f"{tier_code} ({tier_name:.<30s}): {count:3d} columns")

    print(f"{'='*40:>45s}")
    print(f"{'TOTAL FEATURES':.<40s}: {df.shape[1]:3d}")

    # Column names by tier
    if args.show_columns:
        print("\n" + "-"*80)
        print("COLUMN NAMES BY TIER")
        print("-"*80)

        for tier_code, tier_name, tier_cols in tier_info:
            if tier_cols:
                print(f"\n{tier_code} - {tier_name}:")
                for col in sorted(tier_cols):
                    print(f"  • {col}")

    # Data quality checks
    print("\n" + "-"*80)
    print("DATA QUALITY CHECKS")
    print("-"*80)

    # Check for missing values in key columns
    key_cols = ['season', 'week', 'team']
    missing_keys = df[key_cols].isna().sum()
    print(f"Missing key values: {missing_keys.sum()} total")
    if missing_keys.sum() > 0:
        for col, count in missing_keys[missing_keys > 0].items():
            print(f"  • {col}: {count} missing")

    # Overall missing value rate
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct = 100 * missing_cells / total_cells
    print(f"Overall missing values: {missing_cells:,} / {total_cells:,} ({missing_pct:.2f}%)")

    # Most complete features
    completeness = (1 - df.isna().sum() / len(df)) * 100
    most_complete = completeness.nlargest(5)
    print(f"\nMost complete features (top 5):")
    for col, pct in most_complete.items():
        print(f"  • {col}: {pct:.1f}% complete")

    # Least complete features
    least_complete = completeness[completeness < 100].nsmallest(5)
    if not least_complete.empty:
        print(f"\nLeast complete features (bottom 5):")
        for col, pct in least_complete.items():
            print(f"  • {col}: {pct:.1f}% complete ({int((100-pct)*len(df)/100)} missing)")

    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    args = parse_args()

    try:
        # Validate data directory exists
        if not os.path.isdir(args.data_dir):
            print(f"ERROR: Data directory not found: {args.data_dir}", file=sys.stderr)
            sys.exit(1)

        # Step 1: Load raw sources
        print("\n" + "="*80)
        print("STEP 1: Loading raw data sources")
        print("="*80 + "\n")

        sources = load_raw_sources(args.data_dir)

        if not sources:
            print("ERROR: No data sources loaded. Check data directory.", file=sys.stderr)
            sys.exit(1)

        print(f"\n✓ Loaded {len(sources)} data sources")

        # Step 2: Build training frame
        print("\n" + "="*80)
        print("STEP 2: Building training frame")
        print("="*80 + "\n")

        training_frame = build_training_frame(
            sources,
            include_market=args.include_market
        )

        # Step 3: Validate
        print("\n" + "="*80)
        print("STEP 3: Validating training frame")
        print("="*80 + "\n")

        is_valid = validate_training_frame(training_frame, strict=False)

        if is_valid:
            print("✓ Training frame validation PASSED")
        else:
            print("⚠ Training frame validation had warnings (see above)")

        # Step 4: Generate summary
        feature_summary = get_feature_summary(training_frame)
        print_summary(training_frame, feature_summary, args)

        # Step 5: Save to disk
        if not args.dry_run:
            print("="*80)
            print("STEP 4: Saving training frame")
            print("="*80 + "\n")

            # Create output directory if needed
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as parquet
            training_frame.to_parquet(output_path, index=False)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Saved training frame to: {output_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Format: Parquet (compressed)")

        else:
            print("="*80)
            print("DRY RUN - No file saved")
            print("="*80 + "\n")
            print(f"Would have saved to: {args.output_path}")

        print("\n" + "="*80)
        print(" BUILD COMPLETE ".center(80, "="))
        print("="*80 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
