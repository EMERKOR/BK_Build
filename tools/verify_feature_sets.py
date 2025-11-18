#!/usr/bin/env python3
"""
Feature Set Verification Tool

This script verifies that the feature tier categorization in ball_knower/v2/data_config.py
is consistent with the actual data sources and identifies any missing or miscategorized features.

Usage:
    python tools/verify_feature_sets.py
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ball_knower.v2 import data_config


def load_actual_columns():
    """
    Load actual column names from all data sources.

    Returns:
        dict: Mapping of dataset name to list of columns
    """
    data_dir = project_root / 'data'
    current_season_dir = data_dir / 'current_season'

    actual_columns = {}

    # Historical EPA data
    epa_file = data_dir / 'team_week_epa_2013_2024.csv'
    if epa_file.exists():
        df = pd.read_csv(epa_file, nrows=1)
        actual_columns['team_week_epa_2013_2024'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from team_week_epa_2013_2024.csv")
    else:
        print(f"⚠ Warning: {epa_file} not found")

    # nfelo power ratings
    nfelo_files = list(current_season_dir.glob('power_ratings_nfelo_*.csv'))
    if nfelo_files:
        df = pd.read_csv(nfelo_files[0])
        actual_columns['power_ratings_nfelo'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from power_ratings_nfelo")
    else:
        print(f"⚠ Warning: No power_ratings_nfelo files found")

    # Substack power ratings (has 2-row header)
    substack_files = list(current_season_dir.glob('power_ratings_substack_*.csv'))
    if substack_files:
        # Skip first row (decorative header), use second row as column names
        df = pd.read_csv(substack_files[0], skiprows=[0])
        actual_columns['power_ratings_substack'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from power_ratings_substack")
    else:
        print(f"⚠ Warning: No power_ratings_substack files found")

    # nfelo EPA tiers
    epa_tier_files = list(current_season_dir.glob('epa_tiers_nfelo_*.csv'))
    if epa_tier_files:
        df = pd.read_csv(epa_tier_files[0])
        actual_columns['epa_tiers_nfelo'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from epa_tiers_nfelo")

    # nfelo SOS
    sos_files = list(current_season_dir.glob('strength_of_schedule_nfelo_*.csv'))
    if sos_files:
        df = pd.read_csv(sos_files[0])
        actual_columns['strength_of_schedule_nfelo'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from strength_of_schedule_nfelo")

    # Substack weekly projections
    weekly_files = list(current_season_dir.glob('weekly_projections_ppg_substack_*.csv'))
    if weekly_files:
        df = pd.read_csv(weekly_files[0], skiprows=[0])
        actual_columns['weekly_projections_substack'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from weekly_projections_substack")

    # Substack QB EPA
    qb_files = list(current_season_dir.glob('qb_epa_substack_*.csv'))
    if qb_files:
        df = pd.read_csv(qb_files[0], skiprows=[0])
        actual_columns['qb_epa_substack'] = list(df.columns)
        print(f"✓ Loaded {len(df.columns)} columns from qb_epa_substack")

    return actual_columns


def verify_feature_coverage(actual_columns):
    """
    Verify that defined feature tiers cover all actual columns.

    Args:
        actual_columns (dict): Actual columns from data sources

    Returns:
        dict: Verification results
    """
    # Get all defined features
    all_defined = set(
        data_config.STRUCTURAL_KEYS +
        data_config.TEAM_STRENGTH_FEATURES +
        data_config.MARKET_FEATURES +
        data_config.EXPERIMENTAL_FEATURES +
        data_config.FORBIDDEN_FEATURES
    )

    # Get all actual columns
    all_actual = set()
    for dataset, columns in actual_columns.items():
        all_actual.update(columns)

    # Find missing and extra features
    missing_from_config = all_actual - all_defined
    extra_in_config = all_defined - all_actual

    return {
        'all_defined': all_defined,
        'all_actual': all_actual,
        'missing_from_config': missing_from_config,
        'extra_in_config': extra_in_config,
    }


def check_tier_consistency():
    """
    Check that each feature is only in one tier.

    Returns:
        list: Features that appear in multiple tiers
    """
    tier_lists = {
        'T0': set(data_config.STRUCTURAL_KEYS),
        'T1': set(data_config.TEAM_STRENGTH_FEATURES),
        'T2': set(data_config.MARKET_FEATURES),
        'T3': set(data_config.EXPERIMENTAL_FEATURES),
        'TX': set(data_config.FORBIDDEN_FEATURES),
    }

    duplicates = []
    for tier1, features1 in tier_lists.items():
        for tier2, features2 in tier_lists.items():
            if tier1 < tier2:  # Only check each pair once
                overlap = features1 & features2
                if overlap:
                    for feature in overlap:
                        duplicates.append({
                            'feature': feature,
                            'tiers': [tier1, tier2]
                        })

    return duplicates


def generate_report(actual_columns, verification_results, duplicates):
    """
    Generate verification report.

    Args:
        actual_columns (dict): Actual columns from data sources
        verification_results (dict): Results from verify_feature_coverage
        duplicates (list): Features in multiple tiers

    Returns:
        str: Formatted report text
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("BALL KNOWER v2.0 - FEATURE SET VERIFICATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")

    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-"*80)
    summary = data_config.get_all_tiers_summary()
    for tier, info in summary.items():
        report_lines.append(f"  {tier:20s}: {info['count']:4d} features - {info['description']}")
    report_lines.append("")

    # Data source coverage
    report_lines.append("DATA SOURCE COVERAGE")
    report_lines.append("-"*80)
    for dataset, columns in actual_columns.items():
        report_lines.append(f"  {dataset:40s}: {len(columns):3d} columns")
    report_lines.append(f"  {'TOTAL UNIQUE COLUMNS':40s}: {len(verification_results['all_actual']):3d}")
    report_lines.append("")

    # Tier consistency check
    report_lines.append("TIER CONSISTENCY CHECK")
    report_lines.append("-"*80)
    if duplicates:
        report_lines.append(f"  ⚠ WARNING: Found {len(duplicates)} features in multiple tiers:")
        for dup in duplicates:
            report_lines.append(f"    - {dup['feature']} appears in: {', '.join(dup['tiers'])}")
    else:
        report_lines.append("  ✓ PASS: No features appear in multiple tiers")
    report_lines.append("")

    # Missing features
    report_lines.append("MISSING FEATURES (in data but not in config)")
    report_lines.append("-"*80)
    if verification_results['missing_from_config']:
        report_lines.append(f"  ⚠ WARNING: {len(verification_results['missing_from_config'])} features found in data but not categorized:")
        for feature in sorted(verification_results['missing_from_config']):
            # Find which dataset(s) contain this feature
            sources = [ds for ds, cols in actual_columns.items() if feature in cols]
            report_lines.append(f"    - {feature:40s} (from: {', '.join(sources)})")
    else:
        report_lines.append("  ✓ PASS: All data columns are categorized")
    report_lines.append("")

    # Extra features
    report_lines.append("EXTRA FEATURES (in config but not in data)")
    report_lines.append("-"*80)
    if verification_results['extra_in_config']:
        report_lines.append(f"  ⚠ INFO: {len(verification_results['extra_in_config'])} features defined but not found in current data:")
        for feature in sorted(verification_results['extra_in_config']):
            tier = data_config.validate_feature_tier(feature)
            report_lines.append(f"    - {feature:40s} (tier: {tier})")
        report_lines.append("  NOTE: This is expected for derived/engineered features and forbidden features.")
    else:
        report_lines.append("  ✓ INFO: All defined features exist in data")
    report_lines.append("")

    # Forbidden features check
    report_lines.append("FORBIDDEN FEATURES CHECK")
    report_lines.append("-"*80)
    forbidden_in_data = verification_results['all_actual'] & set(data_config.FORBIDDEN_FEATURES)
    if forbidden_in_data:
        report_lines.append(f"  ⚠ WARNING: {len(forbidden_in_data)} forbidden features found in data sources:")
        for feature in sorted(forbidden_in_data):
            sources = [ds for ds, cols in actual_columns.items() if feature in cols]
            report_lines.append(f"    - {feature:40s} (from: {', '.join(sources)})")
        report_lines.append("  ACTION REQUIRED: Ensure these features are never used in model training!")
    else:
        report_lines.append("  ✓ PASS: No forbidden features found in current data sources")
    report_lines.append("")

    # Safe features summary
    report_lines.append("SAFE FEATURES SUMMARY (T0 + T1 + T2)")
    report_lines.append("-"*80)
    safe_count = len(data_config.SAFE_FEATURES)
    safe_in_data = verification_results['all_actual'] & set(data_config.SAFE_FEATURES)
    report_lines.append(f"  Total safe features defined:     {safe_count}")
    report_lines.append(f"  Safe features available in data: {len(safe_in_data)}")
    report_lines.append(f"  Coverage:                        {len(safe_in_data)/safe_count*100:.1f}%")
    report_lines.append("")

    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    return "\n".join(report_lines)


def main():
    """Main verification workflow."""
    print("\n" + "="*80)
    print("BALL KNOWER v2.0 - FEATURE SET VERIFICATION")
    print("="*80 + "\n")

    # Load actual columns from data sources
    print("Loading actual columns from data sources...")
    print("-"*80)
    actual_columns = load_actual_columns()
    print()

    # Verify feature coverage
    print("Verifying feature coverage...")
    print("-"*80)
    verification_results = verify_feature_coverage(actual_columns)
    print(f"✓ Analyzed {len(verification_results['all_actual'])} unique columns from data")
    print(f"✓ Found {len(verification_results['all_defined'])} defined features in config")
    print()

    # Check tier consistency
    print("Checking tier consistency...")
    print("-"*80)
    duplicates = check_tier_consistency()
    if duplicates:
        print(f"⚠ Found {len(duplicates)} features in multiple tiers")
    else:
        print("✓ No duplicate features across tiers")
    print()

    # Generate report
    print("Generating verification report...")
    print("-"*80)
    report = generate_report(actual_columns, verification_results, duplicates)

    # Write report to file
    output_file = project_root / 'data' / '_feature_set_verification.txt'
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"✓ Report written to: {output_file}")
    print()

    # Print report to console
    print(report)

    # Return exit code based on critical issues
    if duplicates:
        print("\n⚠ WARNING: Duplicate features found across tiers!")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
