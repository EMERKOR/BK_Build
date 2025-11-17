"""
Data Validation and Cleaning Framework for Ball Knower
=======================================================

This script identifies and fixes critical data quality issues that caused
Week 11 prediction failures:

1. Duplicate nfelo ratings
2. Team name mapping inconsistencies (LA vs LAR, LV vs OAK)
3. Missing team entries
4. Rating sanity checks
5. Data freshness validation

Expected behavior:
- Load raw nfelo snapshot from GitHub
- Identify all data quality issues
- Create cleaned, validated dataset
- Generate detailed diagnostic report
- Save cleaned data for prediction pipeline

Author: Ball Knower Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ============================================================================
# CONFIGURATION
# ============================================================================

NFELO_SNAPSHOT_URL = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'

# Expected team codes for 2025 season (32 teams)
EXPECTED_TEAMS_2025 = {
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
}

# Team name mapping corrections
# Maps: nfelo_code -> nflverse_code
TEAM_ALIAS_CORRECTIONS = {
    'LAR': 'LA',   # Los Angeles Rams
    'OAK': 'LV',   # Oakland -> Las Vegas Raiders (moved 2020)
}

# Sanity check bounds for nfelo ratings (based on historical range)
NFELO_RATING_MIN = 1000  # Extreme low
NFELO_RATING_MAX = 1800  # Extreme high
NFELO_TYPICAL_MIN = 1200 # Typical minimum
NFELO_TYPICAL_MAX = 1700 # Typical maximum

# ============================================================================
# DATA LOADING
# ============================================================================

def load_raw_data():
    """Load raw nfelo snapshot and Week 11 games."""
    print("\n" + "="*80)
    print("LOADING RAW DATA")
    print("="*80)

    # Load nfelo snapshot
    print(f"\nLoading nfelo snapshot from GitHub...")
    nfelo_raw = pd.read_csv(NFELO_SNAPSHOT_URL)
    print(f"✓ Loaded {len(nfelo_raw)} rows")

    # Load Week 11 2025 games
    print(f"\nLoading Week 11 2025 games from nflverse...")
    games = nflverse.games(season=2025, week=11)
    games = games[games['spread_line'].notna()].copy()
    print(f"✓ Loaded {len(games)} games with Vegas lines")

    return nfelo_raw, games

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_duplicates(nfelo_raw):
    """Identify duplicate team entries in nfelo snapshot."""
    print("\n" + "="*80)
    print("VALIDATION 1: DUPLICATE TEAM ENTRIES")
    print("="*80)

    # Check for duplicates
    duplicates = nfelo_raw[nfelo_raw.duplicated(subset=['team'], keep=False)].copy()

    if len(duplicates) > 0:
        print(f"\n🔴 CRITICAL: Found {len(duplicates)} duplicate entries")

        # Group by team and show differences
        duplicate_teams = duplicates.groupby('team')

        print(f"\nDuplicate teams ({len(duplicate_teams)} teams affected):")

        for team, group in duplicate_teams:
            print(f"\n  Team: {team}")
            print(f"  Occurrences: {len(group)}")

            # Show the rating differences
            if 'nfelo' in group.columns:
                ratings = group['nfelo'].values
                print(f"  nfelo ratings: {ratings}")
                print(f"  Rating spread: {ratings.max() - ratings.min():.2f} points")

            # Show first few columns of each duplicate
            print(f"\n{group.head().to_string()}\n")
    else:
        print("\n✓ No duplicate team entries found")

    return duplicates

def validate_team_mapping(nfelo_raw, games):
    """Validate that all teams in Week 11 games have nfelo ratings."""
    print("\n" + "="*80)
    print("VALIDATION 2: TEAM NAME MAPPING")
    print("="*80)

    # Get unique teams from Week 11 games
    week11_teams = set(games['home_team'].unique()) | set(games['away_team'].unique())

    # Get unique teams from nfelo snapshot
    nfelo_teams = set(nfelo_raw['team'].unique())

    print(f"\nWeek 11 teams: {len(week11_teams)}")
    print(f"nfelo teams:   {len(nfelo_teams)}")

    # Find teams in Week 11 but not in nfelo
    missing_in_nfelo = week11_teams - nfelo_teams

    # Find teams in nfelo but not in Week 11 (could be aliases)
    extra_in_nfelo = nfelo_teams - week11_teams

    issues_found = False

    if missing_in_nfelo:
        print(f"\n🔴 CRITICAL: Teams in Week 11 games but NOT in nfelo snapshot:")
        for team in sorted(missing_in_nfelo):
            print(f"  - {team}")
        issues_found = True

    if extra_in_nfelo:
        print(f"\n🟡 WARNING: Teams in nfelo snapshot but NOT in Week 11:")
        for team in sorted(extra_in_nfelo):
            print(f"  - {team} (possible alias or inactive team)")
        issues_found = True

    if not issues_found:
        print("\n✓ All teams have matching entries")

    # Check expected teams
    print(f"\n\nExpected 2025 teams: {len(EXPECTED_TEAMS_2025)}")
    missing_from_expected = EXPECTED_TEAMS_2025 - week11_teams

    if missing_from_expected:
        print(f"🟡 Teams expected but not in Week 11 (bye/missing):")
        for team in sorted(missing_from_expected):
            print(f"  - {team}")

    return missing_in_nfelo, extra_in_nfelo

def validate_ratings_sanity(nfelo_raw):
    """Check that nfelo ratings are within reasonable bounds."""
    print("\n" + "="*80)
    print("VALIDATION 3: RATING SANITY CHECKS")
    print("="*80)

    if 'nfelo' not in nfelo_raw.columns:
        print("\n🔴 ERROR: 'nfelo' column not found in snapshot")
        return

    ratings = nfelo_raw['nfelo']

    print(f"\nRating statistics:")
    print(f"  Count:   {len(ratings)}")
    print(f"  Mean:    {ratings.mean():.2f}")
    print(f"  Median:  {ratings.median():.2f}")
    print(f"  Std Dev: {ratings.std():.2f}")
    print(f"  Min:     {ratings.min():.2f}")
    print(f"  Max:     {ratings.max():.2f}")

    # Check for NaN values
    nan_count = ratings.isna().sum()
    if nan_count > 0:
        print(f"\n🔴 CRITICAL: {nan_count} NaN rating values found")
        nan_teams = nfelo_raw[ratings.isna()]['team'].values
        print(f"  Teams with NaN ratings: {nan_teams}")
    else:
        print(f"\n✓ No NaN values")

    # Check for extreme values
    extreme_low = nfelo_raw[ratings < NFELO_RATING_MIN]
    extreme_high = nfelo_raw[ratings > NFELO_RATING_MAX]

    if len(extreme_low) > 0:
        print(f"\n🟡 WARNING: {len(extreme_low)} ratings below {NFELO_RATING_MIN}")
        print(extreme_low[['team', 'nfelo']])

    if len(extreme_high) > 0:
        print(f"\n🟡 WARNING: {len(extreme_high)} ratings above {NFELO_RATING_MAX}")
        print(extreme_high[['team', 'nfelo']])

    # Check for unusual values
    unusual_low = nfelo_raw[(ratings >= NFELO_RATING_MIN) & (ratings < NFELO_TYPICAL_MIN)]
    unusual_high = nfelo_raw[(ratings <= NFELO_RATING_MAX) & (ratings > NFELO_TYPICAL_MAX)]

    if len(unusual_low) > 0:
        print(f"\n🟡 INFO: {len(unusual_low)} ratings unusually low (below {NFELO_TYPICAL_MIN})")
        print(unusual_low[['team', 'nfelo']])

    if len(unusual_high) > 0:
        print(f"\n🟡 INFO: {len(unusual_high)} ratings unusually high (above {NFELO_TYPICAL_MAX})")
        print(unusual_high[['team', 'nfelo']])

def validate_data_freshness(nfelo_raw):
    """Check if the data appears to be current."""
    print("\n" + "="*80)
    print("VALIDATION 4: DATA FRESHNESS")
    print("="*80)

    # Look for date columns
    date_cols = [col for col in nfelo_raw.columns if 'date' in col.lower() or 'week' in col.lower()]

    if date_cols:
        print(f"\nFound date/week columns: {date_cols}")
        for col in date_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {nfelo_raw[col].nunique()}")
            print(f"  Latest value:  {nfelo_raw[col].max()}")
    else:
        print("\n🟡 WARNING: No obvious date/week columns found")
        print("Cannot verify data freshness automatically")

    # Show all columns for inspection
    print(f"\n\nAll columns in nfelo snapshot:")
    for i, col in enumerate(nfelo_raw.columns, 1):
        print(f"  {i}. {col}")

# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_duplicates(nfelo_raw):
    """Remove duplicate team entries, keeping the most appropriate one."""
    print("\n" + "="*80)
    print("CLEANING 1: REMOVING DUPLICATES")
    print("="*80)

    # Check if duplicates exist
    has_duplicates = nfelo_raw.duplicated(subset=['team'], keep=False).any()

    if not has_duplicates:
        print("\n✓ No duplicates to clean")
        return nfelo_raw.copy()

    print(f"\nOriginal rows: {len(nfelo_raw)}")

    # Strategy: Keep first occurrence (assumes most recent if sorted chronologically)
    # Log what we're removing
    duplicates = nfelo_raw[nfelo_raw.duplicated(subset=['team'], keep='first')]
    print(f"\nRemoving {len(duplicates)} duplicate rows:")
    for team in duplicates['team'].unique():
        team_dups = duplicates[duplicates['team'] == team]
        print(f"  {team}: removing {len(team_dups)} duplicate(s)")

    # Remove duplicates
    cleaned = nfelo_raw.drop_duplicates(subset=['team'], keep='first').copy()

    print(f"\nCleaned rows: {len(cleaned)}")
    print(f"✓ Removed {len(nfelo_raw) - len(cleaned)} duplicate rows")

    return cleaned

def clean_team_names(nfelo_cleaned):
    """Apply team name corrections for nflverse compatibility."""
    print("\n" + "="*80)
    print("CLEANING 2: TEAM NAME MAPPING")
    print("="*80)

    print(f"\nApplying team alias corrections:")

    corrections_made = []

    for old_code, new_code in TEAM_ALIAS_CORRECTIONS.items():
        if old_code in nfelo_cleaned['team'].values:
            print(f"  {old_code} -> {new_code}")
            nfelo_cleaned.loc[nfelo_cleaned['team'] == old_code, 'team'] = new_code
            corrections_made.append((old_code, new_code))

    if corrections_made:
        print(f"\n✓ Applied {len(corrections_made)} team name correction(s)")
    else:
        print(f"\n✓ No team name corrections needed")

    return nfelo_cleaned

def verify_cleaned_data(nfelo_cleaned, games):
    """Final validation of cleaned dataset."""
    print("\n" + "="*80)
    print("VERIFICATION: CLEANED DATA QUALITY")
    print("="*80)

    # Check for duplicates
    dup_count = nfelo_cleaned.duplicated(subset=['team']).sum()
    print(f"\nDuplicates: {dup_count}")

    # Check for NaN ratings
    nan_count = nfelo_cleaned['nfelo'].isna().sum()
    print(f"NaN ratings: {nan_count}")

    # Check team coverage
    week11_teams = set(games['home_team'].unique()) | set(games['away_team'].unique())
    nfelo_teams = set(nfelo_cleaned['team'].unique())

    missing_teams = week11_teams - nfelo_teams

    print(f"\nWeek 11 teams: {len(week11_teams)}")
    print(f"nfelo teams:   {len(nfelo_teams)}")
    print(f"Missing teams: {len(missing_teams)}")

    if missing_teams:
        print(f"\n🔴 CRITICAL: Still missing teams after cleaning:")
        for team in sorted(missing_teams):
            print(f"  - {team}")
        return False

    if dup_count > 0 or nan_count > 0:
        print(f"\n🔴 CRITICAL: Data quality issues remain after cleaning")
        return False

    print(f"\n✅ ALL VALIDATION CHECKS PASSED")
    return True

# ============================================================================
# OUTPUT
# ============================================================================

def save_cleaned_data(nfelo_cleaned):
    """Save cleaned nfelo ratings to file."""
    print("\n" + "="*80)
    print("SAVING CLEANED DATA")
    print("="*80)

    # Save to data/cache directory
    cache_dir = config.DATA_DIR / 'cache'
    cache_dir.mkdir(exist_ok=True)

    output_file = cache_dir / 'nfelo_snapshot_cleaned.csv'
    nfelo_cleaned.to_csv(output_file, index=False)

    print(f"\n✓ Saved cleaned data to: {output_file}")
    print(f"  Rows: {len(nfelo_cleaned)}")
    print(f"  Columns: {len(nfelo_cleaned.columns)}")

    return output_file

def generate_report(nfelo_raw, nfelo_cleaned, validation_passed):
    """Generate summary report of data cleaning process."""
    print("\n" + "="*80)
    print("DATA CLEANING SUMMARY REPORT")
    print("="*80)

    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n\nRAW DATA:")
    print(f"  Total rows:       {len(nfelo_raw)}")
    print(f"  Unique teams:     {nfelo_raw['team'].nunique()}")
    print(f"  Duplicate teams:  {nfelo_raw.duplicated(subset=['team']).sum()}")

    print(f"\n\nCLEANED DATA:")
    print(f"  Total rows:       {len(nfelo_cleaned)}")
    print(f"  Unique teams:     {nfelo_cleaned['team'].nunique()}")
    print(f"  Duplicate teams:  {nfelo_cleaned.duplicated(subset=['team']).sum()}")

    print(f"\n\nCHANGES:")
    print(f"  Rows removed:     {len(nfelo_raw) - len(nfelo_cleaned)}")
    print(f"  Teams corrected:  {len(TEAM_ALIAS_CORRECTIONS)}")

    print(f"\n\nVALIDATION STATUS:")
    if validation_passed:
        print(f"  ✅ PASSED - Data ready for predictions")
    else:
        print(f"  🔴 FAILED - Manual intervention required")

    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run full validation and cleaning pipeline."""

    print("\n" + "="*80)
    print("BALL KNOWER - DATA VALIDATION & CLEANING FRAMEWORK")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load raw nfelo snapshot from GitHub")
    print("  2. Validate data quality (duplicates, mappings, sanity)")
    print("  3. Clean and correct issues")
    print("  4. Verify cleaned data")
    print("  5. Save cleaned dataset for prediction pipeline")

    # Load data
    nfelo_raw, games = load_raw_data()

    # Run validations
    duplicates = validate_duplicates(nfelo_raw)
    missing_teams, extra_teams = validate_team_mapping(nfelo_raw, games)
    validate_ratings_sanity(nfelo_raw)
    validate_data_freshness(nfelo_raw)

    # Clean data
    nfelo_cleaned = clean_duplicates(nfelo_raw)
    nfelo_cleaned = clean_team_names(nfelo_cleaned)

    # Verify cleaned data
    validation_passed = verify_cleaned_data(nfelo_cleaned, games)

    # Save results
    if validation_passed:
        output_file = save_cleaned_data(nfelo_cleaned)
    else:
        print("\n🔴 WARNING: Data validation failed. Not saving cleaned data.")
        print("Manual intervention required to resolve remaining issues.")

    # Generate report
    generate_report(nfelo_raw, nfelo_cleaned, validation_passed)

    return nfelo_cleaned if validation_passed else None

if __name__ == '__main__':
    cleaned_data = main()

    if cleaned_data is not None:
        print("\n✅ SUCCESS: Data cleaning complete")
        print("\nNext steps:")
        print("  1. Review the cleaned data in data/cache/nfelo_snapshot_cleaned.csv")
        print("  2. Update predict_current_week.py to use cleaned data")
        print("  3. Re-run Week 11 predictions")
        print("  4. Compare predictions to actual results")
    else:
        print("\n🔴 FAILED: Data cleaning incomplete")
        print("\nManual intervention required:")
        print("  1. Review validation errors above")
        print("  2. Investigate missing teams or data quality issues")
        print("  3. Update TEAM_ALIAS_CORRECTIONS if needed")
        print("  4. Re-run this script after fixes")
