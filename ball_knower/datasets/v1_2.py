"""
Ball Knower v1.2 Dataset Builder

Builds training dataset for enhanced Vegas spread prediction model.

Features:
- nfelo ratings (ELO differential)
- Situational adjustments (rest, division, surface, time zone)
- QB adjustments (538 QB EPA)

Target:
- vegas_closing_spread (market consensus)

Use Case:
- Market-calibrated model for identifying betting edges
- Compare predictions to Vegas lines
"""

import pandas as pd
import numpy as np
from pathlib import Path

from ball_knower.features import engineering as features


def build_training_frame(
    start_year: int = 2009,
    end_year: int = 2024,
    data_url: str = None
) -> pd.DataFrame:
    """
    Build v1.2 training dataset from nfelo historical data.

    Args:
        start_year: Start season year (default: 2009)
        end_year: End season year (default: 2024)
        data_url: Optional custom nfelo data URL

    Returns:
        DataFrame with columns:
            - game_id: Unique identifier
            - season, week: Temporal identifiers
            - away_team, home_team: Team identifiers
            - nfelo_diff: Primary feature (home - away)
            - rest_advantage: Combined bye week effects
            - div_game: Division game modifier
            - surface_mod: Surface differential modifier
            - time_advantage: Time zone advantage modifier
            - qb_diff: QB adjustment differential
            - vegas_closing_spread: Target variable
            - home_score, away_score: Actual scores
            - actual_margin: Actual outcome
            - home_points, away_points, home_margin: Intentionally unused

    Expected shape:
        - Rows: 2000-4500 games (depending on year range)
        - Columns: 26
    """
    if data_url is None:
        data_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'

    # Load nfelo historical data
    df = pd.read_csv(data_url)

    # Extract season/week/teams from game_id
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to requested year range
    df = df[(df['season'] >= start_year) & (df['season'] <= end_year)].copy()

    # Filter to complete data
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    # Primary feature: ELO differential
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Situational adjustments
    # Use canonical rest advantage calculation from ball_knower.features.engineering
    df['rest_advantage'] = features.compute_rest_advantage_from_nfelo(df)

    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

    # QB adjustments
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) -
                     df['away_538_qb_adj'].fillna(0))

    # Target: Vegas closing spread
    df['vegas_closing_spread'] = df['home_line_close']

    # Actual outcomes
    df['home_score'] = df['home_score'].fillna(0)
    df['away_score'] = df['away_score'].fillna(0)
    df['actual_margin'] = df['home_score'] - df['away_score']

    # Intentionally unused columns (for leak detection)
    df['home_points'] = df['home_score']
    df['away_points'] = df['away_score']
    df['home_margin'] = df['actual_margin']

    # Remove rows with NaN in critical feature columns
    feature_cols = [
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff', 'vegas_closing_spread'
    ]
    mask = df[feature_cols].notna().all(axis=1)
    df = df[mask].copy()

    # Select final columns
    output_cols = [
        'game_id',
        'season',
        'week',
        'away_team',
        'home_team',
        # Features
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
        # Targets
        'vegas_closing_spread',
        'home_score',
        'away_score',
        'actual_margin',
        # Intentionally unused (leak detection)
        'home_points',
        'away_points',
        'home_margin',
    ]

    # Add any missing columns that nfelo might have
    for col in output_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[output_cols].reset_index(drop=True)
