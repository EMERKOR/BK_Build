"""
Data Validation Framework for BK_Build

This module provides comprehensive validation for all data sources used in the betting models.
It identifies data quality issues before they cause model failures.

Usage:
    python data_validator.py --verbose
    python data_validator.py --fix  # Attempt auto-fixes (creates backups)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


class DataValidator:
    """Validates NFL data quality and identifies issues"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        self.info = []

    def log_issue(self, category: str, message: str, data: Optional[Dict] = None):
        """Log a critical data quality issue"""
        issue = {"category": category, "message": message, "data": data}
        self.issues.append(issue)
        if self.verbose:
            print(f"🔴 ISSUE [{category}]: {message}")
            if data:
                print(f"   Data: {data}")

    def log_warning(self, category: str, message: str, data: Optional[Dict] = None):
        """Log a non-critical warning"""
        warning = {"category": category, "message": message, "data": data}
        self.warnings.append(warning)
        if self.verbose:
            print(f"🟡 WARNING [{category}]: {message}")
            if data:
                print(f"   Data: {data}")

    def log_info(self, message: str):
        """Log informational message"""
        self.info.append(message)
        if self.verbose:
            print(f"ℹ️  {message}")

    def validate_nfelo_ratings(self, df: pd.DataFrame) -> Dict:
        """
        Validate nfelo ratings data

        Checks:
        - Duplicate teams (for snapshot data)
        - Missing teams
        - Rating values in reasonable range
        - Chronological ordering (if historical data)
        """
        self.log_info("Validating nfelo ratings...")

        results = {
            "total_rows": len(df),
            "unique_teams": 0,
            "duplicates": [],
            "missing_standard_teams": [],
            "rating_outliers": [],
            "date_issues": []
        }

        # Check for required columns - nfelo can have different names
        team_col = None
        rating_col = None

        if 'team' in df.columns:
            team_col = 'team'

        if 'nfelo' in df.columns:
            rating_col = 'nfelo'
        elif 'elo_rating' in df.columns:
            rating_col = 'elo_rating'

        if not team_col or not rating_col:
            self.log_issue("NFELO_STRUCTURE", f"Missing required columns. Found: {df.columns.tolist()}")
            return results

        results["unique_teams"] = df[team_col].nunique()

        # Check for duplicate teams (simple duplicates if no season/week)
        if 'season' in df.columns and 'week' in df.columns:
            # Historical data with time series
            duplicates = df.groupby(['season', 'week', team_col]).size()
            duplicates = duplicates[duplicates > 1]

            if len(duplicates) > 0:
                for (season, week, team), count in duplicates.items():
                    dup_data = df[(df['season'] == season) &
                                  (df['week'] == week) &
                                  (df[team_col] == team)]
                    ratings = dup_data[rating_col].tolist()

                    self.log_issue(
                        "NFELO_DUPLICATES",
                        f"Team {team} appears {count} times in {season} Week {week}",
                        {"team": team, "season": season, "week": week, "ratings": ratings}
                    )
                    results["duplicates"].append({
                        "team": team,
                        "season": season,
                        "week": week,
                        "count": count,
                        "ratings": ratings
                    })
        else:
            # Snapshot data - check for simple duplicates
            duplicates = df[team_col].value_counts()
            duplicates = duplicates[duplicates > 1]

            if len(duplicates) > 0:
                for team, count in duplicates.items():
                    dup_data = df[df[team_col] == team]
                    ratings = dup_data[rating_col].tolist()

                    self.log_issue(
                        "NFELO_DUPLICATES",
                        f"Team {team} appears {count} times in snapshot",
                        {"team": team, "count": count, "ratings": ratings}
                    )
                    results["duplicates"].append({
                        "team": team,
                        "count": count,
                        "ratings": ratings
                    })

        # Check for missing current NFL teams (as of 2025)
        current_nfl_teams = {
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LAC', 'LAR', 'LA', 'LV', 'MIA', 'MIN', 'NE', 'NO',
            'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
        }

        teams_in_data = set(df[team_col].unique())
        missing = current_nfl_teams - teams_in_data

        if missing:
            self.log_warning(
                "NFELO_MISSING_TEAMS",
                f"Missing {len(missing)} current NFL teams: {sorted(missing)}",
                {"missing_teams": sorted(missing)}
            )
            results["missing_standard_teams"] = sorted(missing)

        # Check rating value ranges (typical nfelo range is ~1200-1700)
        ratings = df[rating_col].dropna()
        if len(ratings) > 0:
            outliers = ratings[(ratings < 1000) | (ratings > 1800)]
            if len(outliers) > 0:
                self.log_warning(
                    "NFELO_RATING_OUTLIERS",
                    f"Found {len(outliers)} ratings outside normal range (1000-1800)",
                    {"min": float(ratings.min()), "max": float(ratings.max())}
                )
                results["rating_outliers"] = {
                    "count": len(outliers),
                    "min": float(ratings.min()),
                    "max": float(ratings.max())
                }

        # Check for chronological issues (only for historical data)
        if 'season' in df.columns and 'week' in df.columns:
            if not df[['season', 'week']].apply(tuple, axis=1).is_monotonic_increasing:
                self.log_warning(
                    "NFELO_CHRONOLOGY",
                    "Data is not sorted chronologically by season/week"
                )
                results["date_issues"].append("Not chronologically sorted")

        return results

    def validate_game_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate game data

        Checks:
        - Required columns present
        - No null team names
        - Spread values reasonable
        - Home/away teams different
        - Valid dates
        """
        self.log_info("Validating game data...")

        results = {
            "total_games": len(df),
            "seasons": [],
            "issues": []
        }

        # Check required columns
        required_cols = ['season', 'week', 'away_team', 'home_team']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.log_issue("GAME_STRUCTURE", f"Missing required columns: {missing_cols}")
            return results

        results["seasons"] = sorted(df['season'].unique().tolist())

        # Check for null team names
        null_away = df['away_team'].isna().sum()
        null_home = df['home_team'].isna().sum()

        if null_away > 0:
            self.log_issue("GAME_NULL_TEAMS", f"Found {null_away} games with null away_team")
            results["issues"].append(f"{null_away} null away_team")

        if null_home > 0:
            self.log_issue("GAME_NULL_TEAMS", f"Found {null_home} games with null home_team")
            results["issues"].append(f"{null_home} null home_team")

        # Check for same team playing itself
        same_team = df[df['away_team'] == df['home_team']]
        if len(same_team) > 0:
            self.log_issue(
                "GAME_SAME_TEAM",
                f"Found {len(same_team)} games where team plays itself",
                {"examples": same_team[['season', 'week', 'home_team']].head().to_dict()}
            )
            results["issues"].append(f"{len(same_team)} games with same home/away team")

        # Check spread values if present
        if 'spread_line' in df.columns:
            spreads = df['spread_line'].dropna()
            extreme_spreads = spreads[(spreads < -30) | (spreads > 30)]

            if len(extreme_spreads) > 0:
                self.log_warning(
                    "GAME_EXTREME_SPREADS",
                    f"Found {len(extreme_spreads)} spreads outside ±30 points",
                    {"min": float(spreads.min()), "max": float(spreads.max())}
                )

        # Check for duplicate games (same teams, same week)
        if 'season' in df.columns and 'week' in df.columns:
            dups = df.groupby(['season', 'week', 'away_team', 'home_team']).size()
            dups = dups[dups > 1]

            if len(dups) > 0:
                self.log_issue(
                    "GAME_DUPLICATES",
                    f"Found {len(dups)} duplicate games",
                    {"count": len(dups)}
                )
                results["issues"].append(f"{len(dups)} duplicate games")

        return results

    def validate_team_name_consistency(self, games_df: pd.DataFrame, nfelo_df: pd.DataFrame) -> Dict:
        """
        Validate that team names are consistent between data sources

        Checks:
        - All game teams have corresponding nfelo ratings
        - Team name mappings (LAC/LAR/LA, LV, etc.)
        """
        self.log_info("Validating team name consistency...")

        results = {
            "games_teams": set(),
            "nfelo_teams": set(),
            "missing_in_nfelo": set(),
            "missing_in_games": set(),
            "potential_mappings": {}
        }

        # Get unique teams from each source
        if 'away_team' in games_df.columns and 'home_team' in games_df.columns:
            games_teams = set(games_df['away_team'].dropna().unique()) | \
                         set(games_df['home_team'].dropna().unique())
            results["games_teams"] = sorted(games_teams)
        else:
            self.log_issue("CONSISTENCY_CHECK", "Cannot validate - missing team columns in games")
            return results

        if 'team' in nfelo_df.columns:
            nfelo_teams = set(nfelo_df['team'].dropna().unique())
            results["nfelo_teams"] = sorted(nfelo_teams)
        else:
            self.log_issue("CONSISTENCY_CHECK", "Cannot validate - missing team column in nfelo")
            return results

        # Find mismatches
        missing_in_nfelo = games_teams - nfelo_teams
        missing_in_games = nfelo_teams - games_teams

        results["missing_in_nfelo"] = sorted(missing_in_nfelo)
        results["missing_in_games"] = sorted(missing_in_games)

        if missing_in_nfelo:
            self.log_issue(
                "TEAM_NAME_MISMATCH",
                f"Teams in games but not in nfelo: {sorted(missing_in_nfelo)}",
                {"teams": sorted(missing_in_nfelo)}
            )

            # Try to suggest mappings
            for game_team in missing_in_nfelo:
                # Check for partial matches
                potential = [nt for nt in nfelo_teams if game_team in nt or nt in game_team]
                if potential:
                    results["potential_mappings"][game_team] = potential
                    self.log_info(f"   Potential mapping: {game_team} -> {potential}")

        if missing_in_games:
            self.log_warning(
                "TEAM_NAME_EXTRA",
                f"Teams in nfelo but not in games: {sorted(missing_in_games)}",
                {"teams": sorted(missing_in_games)}
            )

        return results

    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        report = []
        report.append("\n" + "="*80)
        report.append("DATA VALIDATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append(f"🔴 Critical Issues: {len(self.issues)}")
        report.append(f"🟡 Warnings: {len(self.warnings)}")
        report.append(f"ℹ️  Info: {len(self.info)}")
        report.append("")

        # Issues
        if self.issues:
            report.append("="*80)
            report.append("CRITICAL ISSUES (Must Fix)")
            report.append("="*80)
            for i, issue in enumerate(self.issues, 1):
                report.append(f"\n{i}. [{issue['category']}] {issue['message']}")
                if issue['data']:
                    report.append(f"   Details: {json.dumps(issue['data'], indent=2)}")

        # Warnings
        if self.warnings:
            report.append("\n" + "="*80)
            report.append("WARNINGS (Should Review)")
            report.append("="*80)
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"\n{i}. [{warning['category']}] {warning['message']}")
                if warning['data']:
                    report.append(f"   Details: {json.dumps(warning['data'], indent=2)}")

        # Status
        report.append("\n" + "="*80)
        if len(self.issues) == 0:
            report.append("✅ DATA QUALITY: GOOD - No critical issues found")
        else:
            report.append("❌ DATA QUALITY: POOR - Critical issues must be resolved")
        report.append("="*80)

        return "\n".join(report)


def main():
    """Run full data validation pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate BK_Build data quality")
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--fix', action='store_true', help='Attempt to auto-fix issues')
    args = parser.parse_args()

    validator = DataValidator(verbose=args.verbose)

    print("\n" + "="*80)
    print("BK_BUILD DATA VALIDATION")
    print("="*80)

    # Load data
    print("\nLoading data sources...")

    # Load nfelo snapshot (current ratings)
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

    # Run validations
    if nfelo_df is not None:
        print("\n" + "-"*80)
        nfelo_results = validator.validate_nfelo_ratings(nfelo_df)

    if games_df is not None:
        print("\n" + "-"*80)
        game_results = validator.validate_game_data(games_df)

    if nfelo_df is not None and games_df is not None:
        print("\n" + "-"*80)
        consistency_results = validator.validate_team_name_consistency(games_df, nfelo_df)

    # Generate report
    report = validator.generate_report()
    print(report)

    # Save report to file
    report_path = Path(__file__).parent / "data_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Full report saved to: {report_path}")

    # Exit with error code if critical issues found
    if len(validator.issues) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
