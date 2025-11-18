#!/usr/bin/env python3
"""
Ball Knower v2.0 - Dataset Pipeline Debug Tool

This script provides detailed debugging and validation of the v2.0 dataset pipeline.
It's designed for manual inspection during development and iteration.

What it does:
1. Loads all raw sources
2. Shows head of each dataset
3. Builds team-week frame with detailed logging
4. Builds training frame
5. Reports any schema violations, missing data, or merge issues

Usage:
    python tools/debug_v2_dataset_pipeline.py
    python tools/debug_v2_dataset_pipeline.py --data-dir data --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from ball_knower.v2.datasets import (
    load_raw_sources,
    build_team_week_frame,
    build_training_frame,
    get_feature_summary,
    validate_training_frame,
)
from ball_knower.v2.data_config import (
    STRUCTURAL_KEYS,
    TEAM_STRENGTH_FEATURES,
    MARKET_FEATURES,
    FORBIDDEN_FEATURES,
    EXPERIMENTAL_FEATURES,
    canonical_schema,
    dataset_roles,
    validate_feature_tier,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug and validate Ball Knower v2.0 dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing input data files (default: data)',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output including data samples',
    )

    parser.add_argument(
        '--show-samples',
        type=int,
        default=5,
        metavar='N',
        help='Number of sample rows to show (default: 5)',
    )

    return parser.parse_args()


def debug_raw_sources(sources, args):
    """Debug raw data sources."""
    print("\n" + "="*80)
    print(" RAW DATA SOURCES ".center(80, "="))
    print("="*80 + "\n")

    for dataset_name, df in sources.items():
        print(f"\n{'─'*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*80}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Show columns with tier classification
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns[:20]:  # Show first 20
            tier = validate_feature_tier(col)
            dtype = df[col].dtype
            missing_pct = 100 * df[col].isna().sum() / len(df)
            print(f"  • {col:30s} [{tier:2s}] ({dtype}) - {missing_pct:5.1f}% missing")

        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more columns")

        # Check for key columns
        dataset_info = dataset_roles.get(dataset_name, {})
        expected_keys = dataset_info.get('keys', [])
        if expected_keys:
            print(f"\nExpected keys: {expected_keys}")
            missing_keys = [k for k in expected_keys if k not in df.columns]
            if missing_keys:
                print(f"⚠ Missing keys: {missing_keys}")
            else:
                print(f"✓ All keys present")

                # Check for missing key values
                for key in expected_keys:
                    missing_count = df[key].isna().sum()
                    if missing_count > 0:
                        print(f"⚠ {key} has {missing_count} missing values")

        # Show sample data
        if args.verbose:
            print(f"\nSample data (first {args.show_samples} rows):")
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.width', 80)
            print(df.head(args.show_samples))

    print("\n" + "="*80 + "\n")


def debug_team_week_frame(team_week_df, args):
    """Debug the team-week frame."""
    print("\n" + "="*80)
    print(" TEAM-WEEK FRAME ".center(80, "="))
    print("="*80 + "\n")

    print(f"Shape: {team_week_df.shape[0]} rows × {team_week_df.shape[1]} columns")

    # Check key completeness
    print("\nKey column analysis:")
    key_cols = ['season', 'week', 'team']
    for col in key_cols:
        if col in team_week_df.columns:
            missing = team_week_df[col].isna().sum()
            unique = team_week_df[col].nunique()
            print(f"  • {col}: {unique} unique values, {missing} missing")
            if col == 'season':
                print(f"    Range: {team_week_df[col].min()} - {team_week_df[col].max()}")
            elif col == 'week':
                print(f"    Range: {team_week_df[col].min()} - {team_week_df[col].max()}")
        else:
            print(f"  ⚠ {col}: NOT FOUND")

    # Check for rows with missing keys
    missing_keys = team_week_df[key_cols].isna().any(axis=1)
    if missing_keys.sum() > 0:
        print(f"\n⚠ Found {missing_keys.sum()} rows with missing key values:")
        print(team_week_df[missing_keys][key_cols].head(10))

    # Feature tier breakdown
    print("\nFeature tier breakdown:")
    feature_summary = get_feature_summary(team_week_df)
    for tier in ['T0', 'T1', 'T2', 'T3', 'TX']:
        count = len(feature_summary[tier])
        if count > 0:
            print(f"  {tier}: {count:3d} columns")

    unknown_cols = feature_summary.get('UNKNOWN', [])
    if unknown_cols:
        print(f"\n⚠ Unknown columns (not in any tier): {len(unknown_cols)}")
        for col in unknown_cols[:10]:
            print(f"    • {col}")
        if len(unknown_cols) > 10:
            print(f"    ... and {len(unknown_cols) - 10} more")

    # Check for forbidden features
    forbidden_present = [col for col in team_week_df.columns if col in FORBIDDEN_FEATURES]
    if forbidden_present:
        print(f"\n⚠⚠⚠ CRITICAL: Forbidden features present: {forbidden_present}")

    # Missing value analysis
    print("\nMissing value analysis:")
    missing_counts = team_week_df.isna().sum()
    missing_pct = 100 * missing_counts / len(team_week_df)

    most_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
    if not most_missing.empty:
        print("  Top 10 columns with missing values:")
        for col, pct in most_missing.items():
            count = missing_counts[col]
            tier = validate_feature_tier(col)
            print(f"    • {col:30s} [{tier:2s}]: {pct:5.1f}% ({count:,} missing)")
    else:
        print("  ✓ No missing values in any column")

    # Sample data
    if args.verbose:
        print(f"\nSample data (first {args.show_samples} rows):")
        pd.set_option('display.max_columns', 15)
        pd.set_option('display.width', 120)
        print(team_week_df.head(args.show_samples))

    print("\n" + "="*80 + "\n")


def debug_training_frame(training_df, args):
    """Debug the final training frame."""
    print("\n" + "="*80)
    print(" TRAINING FRAME ".center(80, "="))
    print("="*80 + "\n")

    print(f"Shape: {training_df.shape[0]} rows × {training_df.shape[1]} columns")

    # Validate schema
    print("\nSchema validation:")
    try:
        is_valid = validate_training_frame(training_df, strict=True)
        print("  ✓ Training frame validation PASSED")
    except ValueError as e:
        print(f"  ⚠ Training frame validation FAILED:")
        print(f"    {e}")

    # Check dtypes against canonical_schema
    print("\nData type validation:")
    dtype_issues = []
    for col in training_df.columns:
        if col in canonical_schema:
            expected_dtype = canonical_schema[col]['dtype']
            actual_dtype = training_df[col].dtype

            # Map pandas dtypes to canonical types
            dtype_map = {
                'int': ['int64', 'int32', 'int16', 'int8'],
                'float': ['float64', 'float32'],
                'str': ['object', 'string'],
            }

            valid_dtypes = dtype_map.get(expected_dtype, [])
            if str(actual_dtype) not in valid_dtypes and actual_dtype.name not in valid_dtypes:
                dtype_issues.append((col, expected_dtype, actual_dtype))

    if dtype_issues:
        print(f"  ⚠ Found {len(dtype_issues)} dtype mismatches:")
        for col, expected, actual in dtype_issues[:10]:
            print(f"    • {col}: expected {expected}, got {actual}")
    else:
        print("  ✓ All dtypes match canonical_schema")

    # Feature tier summary
    print("\nFeature tier summary:")
    feature_summary = get_feature_summary(training_df)
    tier_info = [
        ('T0', 'Structural Keys', feature_summary['T0']),
        ('T1', 'Core Team Strength', feature_summary['T1']),
        ('T2', 'Market & Situational', feature_summary['T2']),
    ]

    for tier_code, tier_name, tier_cols in tier_info:
        print(f"  {tier_code} ({tier_name}): {len(tier_cols)} columns")

    # Forbidden/experimental check
    forbidden = [col for col in training_df.columns if col in FORBIDDEN_FEATURES]
    experimental = [col for col in training_df.columns if col in EXPERIMENTAL_FEATURES]

    if forbidden:
        print(f"\n⚠⚠⚠ CRITICAL: Forbidden features in training frame: {forbidden}")
    else:
        print("\n  ✓ No forbidden features present")

    if experimental:
        print(f"  ⚠ Experimental features in training frame: {experimental}")
    else:
        print("  ✓ No experimental features present")

    # Data quality summary
    print("\nData quality summary:")
    total_cells = training_df.shape[0] * training_df.shape[1]
    missing_cells = training_df.isna().sum().sum()
    missing_pct = 100 * missing_cells / total_cells
    print(f"  Total observations: {training_df.shape[0]:,}")
    print(f"  Total features: {training_df.shape[1]}")
    print(f"  Missing values: {missing_cells:,} / {total_cells:,} ({missing_pct:.2f}%)")

    # Duplicates check
    key_cols = ['season', 'week', 'team']
    duplicates = training_df.duplicated(subset=key_cols, keep=False)
    if duplicates.sum() > 0:
        print(f"\n⚠ Found {duplicates.sum()} duplicate rows (by keys):")
        print(training_df[duplicates][key_cols].head(10))
    else:
        print(f"  ✓ No duplicate rows (unique by {key_cols})")

    # Sample data
    if args.verbose:
        print(f"\nSample data (first {args.show_samples} rows):")
        pd.set_option('display.max_columns', 15)
        pd.set_option('display.width', 120)
        print(training_df.head(args.show_samples))

        print(f"\nSample data (last {args.show_samples} rows):")
        print(training_df.tail(args.show_samples))

    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    args = parse_args()

    try:
        print("\n" + "="*80)
        print(" BALL KNOWER v2.0 - DATASET PIPELINE DEBUG ".center(80, "="))
        print("="*80)

        # Step 1: Load raw sources
        print("\n" + "="*80)
        print("STEP 1: Loading raw sources")
        print("="*80 + "\n")

        sources = load_raw_sources(args.data_dir)

        if not sources:
            print("ERROR: No sources loaded", file=sys.stderr)
            return 1

        print(f"\n✓ Loaded {len(sources)} data sources")

        # Debug raw sources
        debug_raw_sources(sources, args)

        # Step 2: Build team-week frame
        print("\n" + "="*80)
        print("STEP 2: Building team-week frame")
        print("="*80 + "\n")

        team_week_df = build_team_week_frame(sources)

        # Debug team-week frame
        debug_team_week_frame(team_week_df, args)

        # Step 3: Build training frame
        print("\n" + "="*80)
        print("STEP 3: Building training frame")
        print("="*80 + "\n")

        training_df = build_training_frame(sources, include_market=True)

        # Debug training frame
        debug_training_frame(training_df, args)

        # Summary
        print("\n" + "="*80)
        print(" DEBUG COMPLETE ".center(80, "="))
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
