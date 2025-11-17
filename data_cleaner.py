"""
Data Cleaning Module for BK_Build

Provides automated cleaning and fixing of data quality issues identified by data_validator.py

Usage:
    python data_cleaner.py --validate-only  # Check what would be fixed
    python data_cleaner.py --clean          # Actually fix issues (creates backups)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import shutil

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


class DataCleaner:
    """Cleans and fixes data quality issues"""

    def __init__(self, verbose: bool = False, backup: bool = True):
        self.verbose = verbose
        self.backup = backup
        self.fixes_applied = []

    def log_fix(self, category: str, message: str):
        """Log a fix that was applied"""
        self.fixes_applied.append({"category": category, "message": message})
        if self.verbose:
            print(f"🔧 FIX [{category}]: {message}")

    def backup_data(self, file_path: Path) -> Path:
        """Create timestamped backup of data file"""
        if not self.backup:
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"

        shutil.copy2(file_path, backup_path)
        if self.verbose:
            print(f"📦 Created backup: {backup_path}")

        return backup_path

    def fix_nfelo_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate nfelo ratings

        Strategy:
        - For historical data: For each (season, week, team) group with duplicates, keep first
        - For snapshot data: For each team with duplicates, keep first
        """
        team_col = 'team' if 'team' in df.columns else None
        if not team_col:
            return df

        original_len = len(df)

        # Handle historical data with season/week
        if 'season' in df.columns and 'week' in df.columns:
            # Find duplicates
            duplicates = df.groupby(['season', 'week', team_col]).size()
            duplicates = duplicates[duplicates > 1]

            if len(duplicates) == 0:
                if self.verbose:
                    print("✓ No nfelo duplicates found")
                return df

            # For each duplicate group, keep one
            cleaned_rows = []

            for (season, week, team), count in duplicates.items():
                dup_data = df[(df['season'] == season) &
                              (df['week'] == week) &
                              (df[team_col] == team)]

                # Keep the first occurrence (most conservative choice)
                keep_row = dup_data.iloc[0:1]
                cleaned_rows.append(keep_row)

                removed_count = count - 1
                self.log_fix(
                    "NFELO_DUPLICATES",
                    f"Removed {removed_count} duplicate(s) for {team} in {season} Week {week}"
                )

            # Get all non-duplicate rows
            non_dup_mask = ~df.set_index(['season', 'week', team_col]).index.isin(
                duplicates.index
            )
            non_dup_rows = df[non_dup_mask]

            # Combine non-duplicates with chosen rows from duplicate groups
            if cleaned_rows:
                cleaned_df = pd.concat([non_dup_rows] + cleaned_rows, ignore_index=True)
            else:
                cleaned_df = non_dup_rows

            # Sort chronologically
            cleaned_df = cleaned_df.sort_values(['season', 'week', team_col]).reset_index(drop=True)

        else:
            # Snapshot data - simple duplicate removal
            duplicates = df[team_col].value_counts()
            duplicates = duplicates[duplicates > 1]

            if len(duplicates) == 0:
                if self.verbose:
                    print("✓ No nfelo duplicates found")
                return df

            for team, count in duplicates.items():
                removed_count = count - 1
                self.log_fix(
                    "NFELO_DUPLICATES",
                    f"Removed {removed_count} duplicate(s) for {team} in snapshot"
                )

            # Keep first occurrence of each team
            cleaned_df = df.drop_duplicates(subset=[team_col], keep='first').reset_index(drop=True)

        removed = original_len - len(cleaned_df)
        self.log_fix("NFELO_DUPLICATES", f"Total duplicates removed: {removed}")

        return cleaned_df

    def create_team_name_mapping(self) -> Dict[str, str]:
        """
        Create standardized team name mapping

        Handles known team name variations:
        - LA -> LAR (Rams)
        - LAC -> LAC (Chargers)
        - LV -> LV (Raiders, formerly OAK)
        - OAK -> LV (Raiders moved to Las Vegas)
        - SD -> LAC (Chargers moved from San Diego)
        - STL -> LAR (Rams moved from St. Louis)
        """
        mapping = {
            # Current teams (identity mapping)
            'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
            'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
            'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
            'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC',
            'LAC': 'LAC', 'LAR': 'LAR', 'LV': 'LV', 'MIA': 'MIA',
            'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO', 'NYG': 'NYG',
            'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT', 'SF': 'SF',
            'SEA': 'SEA', 'TB': 'TB', 'TEN': 'TEN', 'WAS': 'WAS',

            # Ambiguous "LA" -> assume Rams (more common in betting)
            'LA': 'LAR',

            # Historical relocations
            'OAK': 'LV',    # Raiders moved to Las Vegas
            'SD': 'LAC',    # Chargers moved from San Diego
            'STL': 'LAR',   # Rams moved from St. Louis
        }

        return mapping

    def standardize_team_names(self, df: pd.DataFrame, team_columns: List[str]) -> pd.DataFrame:
        """
        Standardize team names using mapping

        Args:
            df: DataFrame to clean
            team_columns: List of column names containing team names

        Returns:
            DataFrame with standardized team names
        """
        mapping = self.create_team_name_mapping()

        for col in team_columns:
            if col not in df.columns:
                continue

            # Track changes
            before = df[col].value_counts()

            # Apply mapping
            df[col] = df[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)

            # Log changes
            after = df[col].value_counts()
            changed_teams = set(before.index) - set(after.index)

            if changed_teams:
                for team in changed_teams:
                    new_name = mapping.get(team, team)
                    count = before.get(team, 0)
                    self.log_fix(
                        "TEAM_NAMES",
                        f"Standardized {count} occurrences: {team} -> {new_name} (column: {col})"
                    )

        return df

    def validate_and_fix_ratings(self, df: pd.DataFrame, rating_col: str = None) -> pd.DataFrame:
        """
        Validate and fix extreme rating values

        - Ratings should be roughly 1200-1700
        - Values outside 1000-1800 are likely errors
        - Replace with NaN and warn
        """
        # Auto-detect rating column if not specified
        if rating_col is None:
            if 'nfelo' in df.columns:
                rating_col = 'nfelo'
            elif 'elo_rating' in df.columns:
                rating_col = 'elo_rating'
            else:
                return df

        if rating_col not in df.columns:
            return df

        # Find outliers
        mask = (df[rating_col] < 1000) | (df[rating_col] > 1800)
        outlier_count = mask.sum()

        if outlier_count > 0:
            # Log the outliers (include available columns)
            cols_to_show = [rating_col, 'team']
            if 'season' in df.columns:
                cols_to_show.append('season')
            if 'week' in df.columns:
                cols_to_show.append('week')

            outliers = df[mask][cols_to_show].head(10)

            self.log_fix(
                "RATING_OUTLIERS",
                f"Found {outlier_count} extreme ratings (outside 1000-1800), setting to NaN"
            )

            if self.verbose and len(outliers) > 0:
                print(f"   Examples:\n{outliers}")

            # Set outliers to NaN
            df.loc[mask, rating_col] = np.nan

        return df

    def sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is sorted chronologically"""
        if 'season' in df.columns and 'week' in df.columns:
            df = df.sort_values(['season', 'week']).reset_index(drop=True)
            self.log_fix("SORTING", "Sorted data chronologically by season/week")

        return df

    def clean_nfelo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline for nfelo data"""
        if self.verbose:
            print("\n" + "="*80)
            print("CLEANING NFELO DATA")
            print("="*80)

        original_len = len(df)

        # Fix duplicates
        df = self.fix_nfelo_duplicates(df)

        # Standardize team names
        df = self.standardize_team_names(df, ['team'])

        # Validate ratings (auto-detect column)
        df = self.validate_and_fix_ratings(df)

        # Sort chronologically
        df = self.sort_chronologically(df)

        final_len = len(df)
        if self.verbose:
            print(f"\n✓ Cleaning complete: {original_len} -> {final_len} rows")

        return df

    def clean_game_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline for game data"""
        if self.verbose:
            print("\n" + "="*80)
            print("CLEANING GAME DATA")
            print("="*80)

        original_len = len(df)

        # Standardize team names
        df = self.standardize_team_names(df, ['away_team', 'home_team'])

        # Remove games where team plays itself
        same_team_mask = df['away_team'] == df['home_team']
        same_team_count = same_team_mask.sum()

        if same_team_count > 0:
            df = df[~same_team_mask].reset_index(drop=True)
            self.log_fix("GAME_SAME_TEAM", f"Removed {same_team_count} games where team plays itself")

        # Remove duplicate games
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['season', 'week', 'away_team', 'home_team'], keep='first')
        after_dedup = len(df)
        dup_removed = before_dedup - after_dedup

        if dup_removed > 0:
            self.log_fix("GAME_DUPLICATES", f"Removed {dup_removed} duplicate games")

        # Sort chronologically
        df = self.sort_chronologically(df)

        final_len = len(df)
        if self.verbose:
            print(f"\n✓ Cleaning complete: {original_len} -> {final_len} rows")

        return df

    def generate_summary(self) -> str:
        """Generate summary of fixes applied"""
        summary = []
        summary.append("\n" + "="*80)
        summary.append("DATA CLEANING SUMMARY")
        summary.append("="*80)
        summary.append(f"Fixes Applied: {len(self.fixes_applied)}")
        summary.append("")

        if self.fixes_applied:
            for i, fix in enumerate(self.fixes_applied, 1):
                summary.append(f"{i}. [{fix['category']}] {fix['message']}")
        else:
            summary.append("No fixes were necessary - data was already clean!")

        summary.append("="*80)

        return "\n".join(summary)


def main():
    """Run data cleaning pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean BK_Build data")
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate, do not save changes')
    parser.add_argument('--clean', action='store_true',
                       help='Actually clean and save data (creates backups)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backups (not recommended)')

    args = parser.parse_args()

    if not args.validate_only and not args.clean:
        parser.error("Must specify either --validate-only or --clean")

    cleaner = DataCleaner(verbose=args.verbose, backup=not args.no_backup)

    print("\n" + "="*80)
    print("BK_BUILD DATA CLEANING")
    print("="*80)
    print(f"Mode: {'VALIDATE ONLY' if args.validate_only else 'CLEAN AND SAVE'}")

    # Load data
    print("\nLoading data sources...")

    # Load nfelo snapshot
    try:
        nfelo_snapshot_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'
        nfelo_df = pd.read_csv(nfelo_snapshot_url)
        print(f"✓ Loaded nfelo snapshot: {len(nfelo_df)} rows")
    except Exception as e:
        print(f"✗ Failed to load nfelo snapshot: {e}")
        nfelo_df = None

    # Load games from nflverse
    try:
        games_df = nflverse.games()
        print(f"✓ Loaded games: {len(games_df)} rows")
    except Exception as e:
        print(f"✗ Failed to load games: {e}")
        games_df = None

    # Clean nfelo data
    if nfelo_df is not None:
        cleaned_nfelo = cleaner.clean_nfelo_data(nfelo_df)

        if args.clean:
            # Save cleaned data
            output_path = project_root / "data" / "cleaned_nfelo.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup original if it exists
            if output_path.exists():
                cleaner.backup_data(output_path)

            cleaned_nfelo.to_parquet(output_path, index=False)
            print(f"\n💾 Saved cleaned nfelo data to: {output_path}")

    # Clean game data
    if games_df is not None:
        cleaned_games = cleaner.clean_game_data(games_df)

        if args.clean:
            # Save cleaned data
            output_path = project_root / "data" / "cleaned_schedules.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup original if it exists
            if output_path.exists():
                cleaner.backup_data(output_path)

            cleaned_games.to_parquet(output_path, index=False)
            print(f"\n💾 Saved cleaned game data to: {output_path}")

    # Print summary
    summary = cleaner.generate_summary()
    print(summary)

    # Save summary to file
    summary_path = project_root / "data_cleaning_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
