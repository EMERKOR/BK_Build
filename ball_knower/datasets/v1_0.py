"""
Ball Knower v1.0 Dataset Builder

Builds the canonical v1.0 training dataset for actual-margin prediction.

The v1.0 model is intentionally simple:
- Target: actual game margin (NOT Vegas line)
- Features: nfelo_diff (and potentially simple structural flags)
- Philosophy: Model the game first, then compare to Vegas

This is a clean break from v1.2 which was trained to predict Vegas lines.

Data sources:
- nflverse games.csv: actual scores and game metadata
- nfelo games.csv: nfelo ratings and Vegas lines
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Data source URLs
NFLVERSE_GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
NFELO_GAMES_URL = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'


def build_v1_0_training_frame(min_season: int = 2009, max_season: int | None = None) -> pd.DataFrame:
    """
    Build the canonical v1.0 training dataset for actual-margin prediction.

    Returns one row per game with:
      - season, week
      - home_team, away_team
      - home_score, away_score
      - actual_margin (home_score - away_score)
      - nfelo_diff (starting_nfelo_home - starting_nfelo_away)
      - vegas_line (home_line_close, for later evaluation)
      - structural flags (neutral, playoffs, divisional, etc. if available)

    The target for v1.0 is actual_margin, NOT vegas_line.

    Args:
        min_season: Minimum season year (inclusive), default 2009
        max_season: Maximum season year (inclusive), default None (use all available)

    Returns:
        pd.DataFrame: Clean training frame with one row per game
    """
    print(f"\n[v1.0 Dataset Builder]")

    # ========================================================================
    # LOAD NFLVERSE GAMES (for actual scores)
    # ========================================================================

    print(f"  Loading nflverse games (for actual scores)...")
    nflverse_df = pd.read_csv(NFLVERSE_GAMES_URL)
    print(f"  Loaded {len(nflverse_df):,} games from nflverse")

    # Filter for relevant game types (regular season + playoffs)
    if 'game_type' in nflverse_df.columns:
        nflverse_df = nflverse_df[
            nflverse_df['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])
        ].copy()

    # Filter by season
    nflverse_df = nflverse_df[nflverse_df['season'] >= min_season].copy()
    if max_season is not None:
        nflverse_df = nflverse_df[nflverse_df['season'] <= max_season].copy()

    # Keep only games with final scores
    nflverse_df = nflverse_df[nflverse_df['home_score'].notna()].copy()
    nflverse_df = nflverse_df[nflverse_df['away_score'].notna()].copy()

    # Calculate actual margin
    nflverse_df['actual_margin'] = nflverse_df['home_score'] - nflverse_df['away_score']

    # Create game_id in nfelo format: YYYY_WW_AWAY_HOME
    nflverse_df['game_id_nfelo'] = (
        nflverse_df['season'].astype(str) + '_' +
        nflverse_df['week'].astype(str).str.zfill(2) + '_' +
        nflverse_df['away_team'] + '_' +
        nflverse_df['home_team']
    )

    print(f"  Filtered to {len(nflverse_df):,} games with scores")

    # ========================================================================
    # LOAD NFELO GAMES (for ratings and Vegas lines)
    # ========================================================================

    print(f"  Loading nfelo games (for ratings)...")
    nfelo_df = pd.read_csv(NFELO_GAMES_URL)
    print(f"  Loaded {len(nfelo_df):,} games from nfelo")

    # Keep only games with nfelo ratings and Vegas lines
    nfelo_df = nfelo_df[nfelo_df['starting_nfelo_home'].notna()].copy()
    nfelo_df = nfelo_df[nfelo_df['starting_nfelo_away'].notna()].copy()
    nfelo_df = nfelo_df[nfelo_df['home_line_close'].notna()].copy()

    # Calculate nfelo_diff
    nfelo_df['nfelo_diff'] = nfelo_df['starting_nfelo_home'] - nfelo_df['starting_nfelo_away']

    # Select columns to merge
    nfelo_cols = [
        'game_id',
        'nfelo_diff',
        'home_line_close',
        'div_game_mod'
    ]
    nfelo_df = nfelo_df[nfelo_cols].copy()
    nfelo_df = nfelo_df.rename(columns={'game_id': 'game_id_nfelo'})

    print(f"  Filtered to {len(nfelo_df):,} games with nfelo ratings")

    # ========================================================================
    # MERGE DATASETS
    # ========================================================================

    print(f"  Merging nflverse and nfelo datasets...")

    # Inner join on game_id
    df = nflverse_df.merge(
        nfelo_df,
        on='game_id_nfelo',
        how='inner'
    )

    print(f"  Merged dataset: {len(df):,} games")

    if len(df) == 0:
        print("\n  ⚠ Warning: No games matched between nflverse and nfelo!")
        print("  This might be due to game_id format mismatch.")
        return pd.DataFrame()

    # ========================================================================
    # CREATE FINAL DATASET
    # ========================================================================

    # Rename for clarity
    df = df.rename(columns={'home_line_close': 'vegas_line'})

    # Add structural features
    # Neutral site flag (if available in nflverse)
    if 'location' in df.columns:
        df['is_neutral'] = (df['location'] == 'Neutral').astype(int)
    else:
        df['is_neutral'] = 0

    # Divisional game flag (from nfelo)
    df['is_divisional'] = df['div_game_mod'].fillna(0).astype(bool).astype(int)

    # Playoff game (inferred from week number, typically week > 18)
    df['is_playoff'] = (df['week'] > 18).astype(int)

    # Select and order final columns
    columns_to_keep = [
        'game_id_nfelo',
        'season',
        'week',
        'away_team',
        'home_team',
        'home_score',
        'away_score',
        'nfelo_diff',
        'actual_margin',
        'vegas_line',
        'is_neutral',
        'is_divisional',
        'is_playoff'
    ]

    result = df[columns_to_keep].copy()
    result = result.rename(columns={'game_id_nfelo': 'game_id'})
    result = result.reset_index(drop=True)

    print(f"  Final dataset: {len(result):,} games, {len(result.columns)} columns")
    print(f"  Season range: {result['season'].min()}-{result['season'].max()}")
    print(f"  ✓ v1.0 training frame built successfully\n")

    return result


def save_v1_0_training_frame(
    path: str | Path,
    min_season: int = 2009,
    max_season: int | None = None
) -> None:
    """
    Build and save the v1.0 training frame to a file.

    Args:
        path: Output file path (CSV or Parquet)
        min_season: Minimum season year (inclusive)
        max_season: Maximum season year (inclusive)
    """
    df = build_v1_0_training_frame(min_season=min_season, max_season=max_season)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"\n  Saved to: {path}")
    print(f"  File size: {path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    # Demo: Build and display sample
    print("\n" + "="*80)
    print("BALL KNOWER v1.0 DATASET BUILDER - DEMO")
    print("="*80)

    df = build_v1_0_training_frame(min_season=2009, max_season=2023)

    print("\n" + "="*80)
    print("SAMPLE DATA (first 5 games)")
    print("="*80)
    print(df.head().to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df[['nfelo_diff', 'actual_margin', 'vegas_line']].describe())

    print("\n" + "="*80)
    print("SEASON BREAKDOWN")
    print("="*80)
    season_counts = df.groupby('season').size()
    print(season_counts.to_string())

    print("\n")
