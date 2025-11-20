"""
Build FantasyPoints Features CLI Script

This script builds team-week and matchup-week level features from FantasyPoints data.

Usage:
    python src/build_fantasypoints_features.py

Outputs:
    - data/fantasypoints/fpd_team_week_2025.parquet
    - data/fantasypoints/fpd_matchup_week_2025.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ball_knower.fantasypoints import features, loaders


def load_games_dataset(season: int = 2025) -> pd.DataFrame:
    """
    Load the canonical Ball Knower games dataset.

    Args:
        season: NFL season year

    Returns:
        DataFrame with columns: game_id, season, week, home_team, away_team
    """
    print(f"\nLoading games dataset for {season} season...")

    # Try to load from local schedules.parquet first
    local_path = PROJECT_ROOT / "schedules.parquet"

    if local_path.exists():
        print(f"✓ Loading from local file: {local_path}")
        schedules = pd.read_parquet(local_path)
    else:
        # Fall back to nflverse GitHub data
        print("✓ Loading from nflverse GitHub...")
        url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
        schedules = pd.read_csv(url)

    # Filter to requested season
    schedules = schedules[schedules['season'] == season].copy()

    # Keep only regular season
    if 'game_type' in schedules.columns:
        schedules = schedules[schedules['game_type'] == 'REG'].copy()

    # Ensure we have the required columns
    required_cols = ['game_id', 'season', 'week', 'home_team', 'away_team']

    # Handle different column naming conventions
    if 'team_home' in schedules.columns:
        schedules = schedules.rename(columns={
            'team_home': 'home_team',
            'team_away': 'away_team'
        })

    # Create game_id if it doesn't exist
    if 'game_id' not in schedules.columns:
        schedules['game_id'] = (
            schedules['season'].astype(str) + '_' +
            schedules['week'].astype(str) + '_' +
            schedules['away_team'] + '_' +
            schedules['home_team']
        )

    # Select only required columns
    games_df = schedules[required_cols].copy()

    print(f"✓ Loaded {len(games_df)} games for {season} season")

    return games_df


def print_dataframe_summary(df: pd.DataFrame, name: str) -> None:
    """
    Print a summary of the DataFrame structure.

    Args:
        df: DataFrame to summarize
        name: Name of the dataset
    """
    print(f"\n{'='*80}")
    print(f"{name} Summary")
    print(f"{'='*80}")
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"  {i:3d}. {col:40s} {str(dtype):10s} ({non_null:4d} non-null, {null_pct:5.1f}% null)")

    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    print(f"\nBasic statistics:")
    print(df.describe().to_string())


def main():
    """Main execution function."""
    print("="*80)
    print("FantasyPoints Feature Builder")
    print("="*80)

    season = 2025

    # Step 1: Load games dataset
    games_df = load_games_dataset(season=season)

    # Step 2: Build team-week features
    print(f"\n{'='*80}")
    print("Building Team-Week Features")
    print(f"{'='*80}")

    team_week_df = features.build_fpd_team_week(season=season)

    # Step 3: Build matchup-week features
    print(f"\n{'='*80}")
    print("Building Matchup-Week Features")
    print(f"{'='*80}")

    matchup_week_df = features.build_fpd_matchup_week(team_week_df, games_df)

    # Step 4: Save features
    print(f"\n{'='*80}")
    print("Saving Features")
    print(f"{'='*80}")

    features.save_features(team_week_df, matchup_week_df, season=season)

    # Step 5: Print summaries
    print_dataframe_summary(team_week_df, "Team-Week Features")
    print_dataframe_summary(matchup_week_df, "Matchup-Week Features")

    # Feature column breakdown
    print(f"\n{'='*80}")
    print("Feature Column Breakdown")
    print(f"{'='*80}")

    # Categorize team-week features
    def_cols = [c for c in team_week_df.columns if c.startswith('def_')]
    qb_cols = [c for c in team_week_df.columns if c.startswith('qb_')]
    wr_cols = [c for c in team_week_df.columns if c.startswith('wr_')]
    rush_cols = [c for c in team_week_df.columns if c.startswith('rush_')]
    off_cols = [c for c in team_week_df.columns if c.startswith('off_')]
    ol_cols = [c for c in team_week_df.columns if c.startswith('ol_')]
    other_cols = [c for c in team_week_df.columns if not any(c.startswith(p) for p in ['def_', 'qb_', 'wr_', 'rush_', 'off_', 'ol_', 'season', 'team'])]

    print(f"\nTeam-Week Feature Categories:")
    print(f"  Defensive Coverage:     {len(def_cols):3d} features")
    print(f"  QB Metrics:             {len(qb_cols):3d} features")
    print(f"  WR/TE Room:             {len(wr_cols):3d} features")
    print(f"  Rushing:                {len(rush_cols):3d} features")
    print(f"  Offensive Identity:     {len(off_cols):3d} features")
    print(f"  Offensive Line:         {len(ol_cols):3d} features")
    print(f"  Other:                  {len(other_cols):3d} features")
    print(f"  {'─'*40}")
    print(f"  TOTAL:                  {len(team_week_df.columns) - 2:3d} features (excluding season, team)")

    # Categorize matchup features
    home_cols = [c for c in matchup_week_df.columns if c.startswith('home_') and c not in ['home_team']]
    away_cols = [c for c in matchup_week_df.columns if c.startswith('away_') and c not in ['away_team']]
    diff_cols = [c for c in matchup_week_df.columns if c.startswith('diff_')]
    faces_cols = [c for c in matchup_week_df.columns if 'faces_' in c]

    print(f"\nMatchup-Week Feature Categories:")
    print(f"  Home Team Features:     {len(home_cols):3d} features")
    print(f"  Away Team Features:     {len(away_cols):3d} features")
    print(f"  Differentials:          {len(diff_cols):3d} features")
    print(f"  Coverage Matchups:      {len(faces_cols):3d} features")
    print(f"  {'─'*40}")
    print(f"  TOTAL:                  {len(matchup_week_df.columns) - 5:3d} features (excluding game_id, season, week, home_team, away_team)")

    print(f"\n{'='*80}")
    print("✓ FantasyPoints Feature Build Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
