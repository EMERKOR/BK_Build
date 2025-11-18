"""
Ball Knower v1.2 Dataset Builder

Provides reusable functions to build the canonical v1.2 training frame
from nfelo historical data.

This module:
- Loads nfelo historical games
- Extracts season/week/team information
- Engineers features (ELO diff, rest advantage, etc.)
- Computes game outcomes (home_margin)
- Returns a clean DataFrame for training or benchmarking
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def build_training_frame(
    nfelo_url: str = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv',
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build the canonical v1.2 training frame from nfelo historical data.

    Parameters
    ----------
    nfelo_url : str
        URL or file path to nfelo games CSV
    min_season : int, optional
        Minimum season to include (default: no filter)
    max_season : int, optional
        Maximum season to include (default: no filter)

    Returns
    -------
    df : DataFrame
        Canonical game-level frame with columns:
            - game_id: unique game identifier (season_week_away_home)
            - season: NFL season year
            - week: NFL week number
            - away_team: away team code
            - home_team: home team code
            - away_score: away team final score
            - home_score: home team final score
            - home_margin: home_score - away_score
            - vegas_line: Vegas closing line (home referenced)
            - nfelo_diff: starting ELO differential (home - away)
            - rest_advantage: combined bye/rest adjustment
            - div_game: divisional game flag
            - surface_mod: surface adjustment
            - time_advantage: home time zone advantage
            - qb_diff: QB adjustment differential (home - away)
            - bk_line: Ball Knower v1.2 predicted line (if available)

    Notes
    -----
    This function filters to games with complete data (non-null closing lines and ELO ratings).
    """
    # Load nfelo historical games
    df = pd.read_csv(nfelo_url)

    # Extract season/week/teams from game_id
    df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(
        r'(\d{4})_(\d+)_(\w+)_(\w+)'
    )
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to complete data
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()

    # Apply season filters if specified
    if min_season is not None:
        df = df[df['season'] >= min_season].copy()
    if max_season is not None:
        df = df[df['season'] <= max_season].copy()

    # Rename and compute basic columns
    df['vegas_line'] = df['home_line_close']

    # Compute home margin from scores if available
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['home_margin'] = df['home_score'] - df['away_score']
    elif 'score_home' in df.columns and 'score_away' in df.columns:
        # Alternative column names
        df['home_score'] = df['score_home']
        df['away_score'] = df['score_away']
        df['home_margin'] = df['home_score'] - df['away_score']
    else:
        # If no score columns, set to NaN
        df['home_score'] = np.nan
        df['away_score'] = np.nan
        df['home_margin'] = np.nan

    # Feature engineering
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Situational adjustments
    df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
    df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
    df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

    # QB adjustments
    df['qb_diff'] = (
        df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0)
    )

    # Select and order canonical columns
    canonical_cols = [
        'game_id',
        'season',
        'week',
        'away_team',
        'home_team',
        'away_score',
        'home_score',
        'home_margin',
        'vegas_line',
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
    ]

    # Add any additional columns that exist
    optional_cols = ['gameday', 'gametime', 'location', 'roof']
    for col in optional_cols:
        if col in df.columns:
            canonical_cols.append(col)

    # Filter to columns that exist
    available_cols = [c for c in canonical_cols if c in df.columns]
    df = df[available_cols].copy()

    df = df.reset_index(drop=True)

    return df


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
        if feat in model_coef and feat in df.columns:
            df['bk_line'] += model_coef[feat] * df[feat]

    return df
