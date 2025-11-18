"""
Ball Knower v1.2 Dataset Builder

Provides the canonical v1.2 training/benchmark frame using nflverse games data.

This module:
- Loads nflverse historical games (with actual scores)
- Loads nfelo ratings and situational adjustments
- Merges them to create a complete dataset with:
  * Actual game results (home_score, away_score, home_margin)
  * Vegas closing spreads
  * ELO ratings and derived features
  * Situational adjustments (rest, divisional, QB, etc.)

The result is suitable for:
- Training ML models
- Backtesting predictions
- Benchmarking against external sources (PredictionTracker, etc.)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import sys

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def build_training_frame(
    start_season: int = 2009,
    end_season: int = 2024,
    nflverse_url: str = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    nfelo_url: str = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv',
) -> pd.DataFrame:
    """
    Build the canonical v1.2 training frame combining nflverse games (with scores)
    and nfelo ratings/adjustments.

    Parameters
    ----------
    start_season : int
        First season to include (default: 2009)
    end_season : int
        Last season to include (default: 2024)
    nflverse_url : str
        URL to nflverse games CSV (contains actual scores and spreads)
    nfelo_url : str
        URL to nfelo games CSV (contains ELO ratings and situational adjustments)

    Returns
    -------
    df : DataFrame
        Canonical game-level frame with columns:
            - game_id: unique game identifier (season_week_away_home)
            - season: NFL season year
            - week: NFL week number
            - gameday: game date
            - away_team: away team code
            - home_team: home team code
            - away_score: away team final score
            - home_score: home team final score
            - home_margin: home_score - away_score
            - vegas_closing_spread: Vegas closing line (home referenced, negative = home favored)
            - nfelo_diff: starting ELO differential (home - away)
            - rest_advantage: combined bye/rest adjustment
            - div_game: divisional game flag
            - surface_mod: surface adjustment
            - time_advantage: home time zone advantage
            - qb_diff: QB adjustment differential (home - away)

    Notes
    -----
    This function:
    1. Loads nflverse games to get actual scores and Vegas spreads
    2. Loads nfelo data to get ELO ratings and situational adjustments
    3. Merges on (season, week, away_team, home_team)
    4. Filters to games with complete data (scores and Vegas lines)

    The nflverse dataset uses consistent team abbreviations (e.g., 'LAC', 'LAR').
    The nfelo dataset may use different abbreviations - we normalize them during merge.
    """
    print(f"\n[1/4] Loading nflverse games ({start_season}-{end_season})...")

    # Load nflverse games (has actual scores and spreads)
    nflverse_df = pd.read_csv(nflverse_url)

    # Filter to requested seasons and regular season only
    nflverse_df = nflverse_df[
        (nflverse_df['season'] >= start_season) &
        (nflverse_df['season'] <= end_season)
    ].copy()

    # Keep only regular season games (filter out playoffs for now)
    if 'game_type' in nflverse_df.columns:
        nflverse_df = nflverse_df[nflverse_df['game_type'] == 'REG'].copy()

    # Filter to games with complete data (scores and Vegas lines)
    nflverse_df = nflverse_df[nflverse_df['home_score'].notna()].copy()
    nflverse_df = nflverse_df[nflverse_df['away_score'].notna()].copy()
    nflverse_df = nflverse_df[nflverse_df['spread_line'].notna()].copy()

    # Compute home margin
    nflverse_df['home_margin'] = nflverse_df['home_score'] - nflverse_df['away_score']

    # Rename spread_line to vegas_closing_spread for clarity
    nflverse_df['vegas_closing_spread'] = nflverse_df['spread_line']

    # Normalize team column names if needed
    if 'team_home' in nflverse_df.columns:
        nflverse_df.rename(columns={
            'team_home': 'home_team',
            'team_away': 'away_team',
        }, inplace=True)

    print(f"✓ Loaded {len(nflverse_df):,} nflverse games with scores and spreads")

    print(f"\n[2/4] Loading nfelo data for ELO ratings and adjustments...")

    # Load nfelo historical games (has ELO ratings and situational features)
    nfelo_df = pd.read_csv(nfelo_url)

    # Extract season/week/teams from nfelo game_id
    nfelo_df[['season', 'week', 'away_team', 'home_team']] = nfelo_df['game_id'].str.extract(
        r'(\d{4})_(\d+)_(\w+)_(\w+)'
    )
    nfelo_df['season'] = nfelo_df['season'].astype(int)
    nfelo_df['week'] = nfelo_df['week'].astype(int)

    # Filter to requested seasons
    nfelo_df = nfelo_df[
        (nfelo_df['season'] >= start_season) &
        (nfelo_df['season'] <= end_season)
    ].copy()

    # Filter to games with ELO ratings
    nfelo_df = nfelo_df[nfelo_df['starting_nfelo_home'].notna()].copy()
    nfelo_df = nfelo_df[nfelo_df['starting_nfelo_away'].notna()].copy()

    # Compute ELO differential
    nfelo_df['nfelo_diff'] = nfelo_df['starting_nfelo_home'] - nfelo_df['starting_nfelo_away']

    # Extract situational adjustments
    nfelo_df['home_bye_mod'] = nfelo_df['home_bye_mod'].fillna(0)
    nfelo_df['away_bye_mod'] = nfelo_df['away_bye_mod'].fillna(0)
    nfelo_df['rest_advantage'] = nfelo_df['home_bye_mod'] + nfelo_df['away_bye_mod']

    nfelo_df['div_game'] = nfelo_df['div_game_mod'].fillna(0)
    nfelo_df['surface_mod'] = nfelo_df['dif_surface_mod'].fillna(0)
    nfelo_df['time_advantage'] = nfelo_df['home_time_advantage_mod'].fillna(0)

    # QB adjustments
    nfelo_df['qb_diff'] = (
        nfelo_df['home_538_qb_adj'].fillna(0) - nfelo_df['away_538_qb_adj'].fillna(0)
    )

    # Select nfelo columns to merge
    nfelo_cols = [
        'season', 'week', 'away_team', 'home_team',
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff'
    ]
    nfelo_df = nfelo_df[nfelo_cols].copy()

    print(f"✓ Loaded {len(nfelo_df):,} nfelo games with ELO and adjustments")

    print(f"\n[3/4] Merging nflverse and nfelo datasets...")

    # Merge nflverse (scores) + nfelo (ratings/features)
    # Merge on (season, week, away_team, home_team)
    merged_df = nflverse_df.merge(
        nfelo_df,
        on=['season', 'week', 'away_team', 'home_team'],
        how='inner',  # Only keep games that exist in both datasets
        suffixes=('_nflverse', '_nfelo')
    )

    print(f"✓ Merged to {len(merged_df):,} games with complete data")

    print(f"\n[4/4] Finalizing canonical dataset...")

    # Create canonical game_id for consistency
    merged_df['game_id'] = (
        merged_df['season'].astype(str) + '_' +
        merged_df['week'].astype(str) + '_' +
        merged_df['away_team'] + '_' +
        merged_df['home_team']
    )

    # Select and order canonical columns
    canonical_cols = [
        'game_id',
        'season',
        'week',
        'gameday',
        'away_team',
        'home_team',
        'away_score',
        'home_score',
        'home_margin',
        'vegas_closing_spread',
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
    ]

    # Add optional columns if they exist
    optional_cols = ['gametime', 'location', 'roof', 'surface', 'temp', 'wind']
    for col in optional_cols:
        if col in merged_df.columns:
            canonical_cols.append(col)

    # Filter to columns that exist
    available_cols = [c for c in canonical_cols if c in merged_df.columns]
    result_df = merged_df[available_cols].copy()

    result_df = result_df.reset_index(drop=True)

    print(f"✓ Canonical v1.2 dataset ready: {len(result_df):,} games")
    print(f"  Season range: {result_df['season'].min()}-{result_df['season'].max()}")
    print(f"  Columns: {len(result_df.columns)}")
    print(f"  Missing scores: {result_df['home_margin'].isna().sum()}")
    print(f"  Missing Vegas lines: {result_df['vegas_closing_spread'].isna().sum()}")

    return result_df


def load_v1_2_model(model_path: Optional[str] = None) -> tuple[dict, float]:
    """
    Load the Ball Knower v1.2 model coefficients and intercept.

    Parameters
    ----------
    model_path : str, optional
        Path to model JSON file. If None, uses default location:
        output/ball_knower_v1_2_model.json

    Returns
    -------
    coefficients : dict
        Model coefficients mapping feature names to values
    intercept : float
        Model intercept

    Raises
    ------
    FileNotFoundError
        If model file doesn't exist at specified path
    ValueError
        If model file is malformed or missing required keys
    """
    import json

    if model_path is None:
        # Use default model location
        model_path = _PROJECT_ROOT / "output" / "ball_knower_v1_2_model.json"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Ball Knower v1.2 model not found at: {model_path}\n"
            f"Please ensure the model file exists or provide a valid path."
        )

    with open(model_path, 'r') as f:
        model_data = json.load(f)

    # Validate model structure
    if 'coefficients' not in model_data:
        raise ValueError("Model file missing 'coefficients' key")
    if 'intercept' not in model_data:
        raise ValueError("Model file missing 'intercept' key")

    return model_data['coefficients'], model_data['intercept']


def add_bk_predictions(
    df: pd.DataFrame,
    model_coef: dict,
    intercept: float,
) -> pd.DataFrame:
    """
    Add Ball Knower v1.2 predictions to a game-level DataFrame.

    Parameters
    ----------
    df : DataFrame
        Game-level frame from build_training_frame()
    model_coef : dict
        Model coefficients mapping feature names to values
    intercept : float
        Model intercept

    Returns
    -------
    df : DataFrame
        Input frame with added 'bk_line' column

    Notes
    -----
    This function computes the BK v1.2 predicted line using the Ridge model:
        bk_line = intercept + sum(coef[feat] * df[feat] for feat in features)

    If a required feature is missing from the dataframe, it will be treated as 0.
    This is consistent with Ridge regression handling of missing features.
    """
    df = df.copy()

    feature_cols = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
    ]

    # Build prediction from coefficients
    df['bk_line'] = intercept

    for feat in feature_cols:
        if feat in model_coef:
            if feat in df.columns:
                df['bk_line'] += model_coef[feat] * df[feat]
            else:
                # Missing feature - treat as 0 (no contribution)
                # This handles cases where div_game or other features are missing
                pass

    return df


def build_training_frame_with_predictions(
    start_season: int = 2009,
    end_season: int = 2024,
    model_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Build the canonical v1.2 training frame WITH Ball Knower predictions.

    This is a convenience function that:
    1. Builds the canonical training frame
    2. Loads the v1.2 model
    3. Adds bk_line predictions

    Parameters
    ----------
    start_season : int
        First season to include (default: 2009)
    end_season : int
        Last season to include (default: 2024)
    model_path : str, optional
        Path to model JSON file. If None, uses default location.
    **kwargs
        Additional arguments passed to build_training_frame()

    Returns
    -------
    df : DataFrame
        Canonical game-level frame with 'bk_line' column added

    Examples
    --------
    >>> df = build_training_frame_with_predictions(start_season=2020, end_season=2024)
    >>> print(df[['season', 'week', 'home_team', 'away_team', 'bk_line']].head())
    """
    # Build base training frame
    df = build_training_frame(start_season=start_season, end_season=end_season, **kwargs)

    # Load model
    coefficients, intercept = load_v1_2_model(model_path)

    # Add predictions
    df = add_bk_predictions(df, coefficients, intercept)

    return df


if __name__ == "__main__":
    # Test the dataset builder
    print("="*80)
    print("Ball Knower v1.2 Canonical Dataset Builder")
    print("="*80)

    df = build_training_frame(start_season=2020, end_season=2024)

    print("\n" + "="*80)
    print("DATASET PREVIEW")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nFirst 3 games:")
    print(df.head(3).to_string())

    print(f"\nLast 3 games:")
    print(df.tail(3).to_string())

    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"\nGames by season:")
    print(df.groupby('season').size())

    print(f"\nActual margin statistics:")
    print(df['home_margin'].describe())

    print(f"\nVegas spread statistics:")
    print(df['vegas_closing_spread'].describe())
