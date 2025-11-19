"""
Ball Knower v1.0 Dataset Builder

Builds training dataset for baseline actual margin prediction model.

Features:
- nfelo ratings (ELO differential)
- Basic game metadata

Target:
- actual_margin (home_score - away_score)

Use Case:
- Foundation model that ignores betting markets
- Establishes "football truth" baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_training_frame(
    start_year: int = 2009,
    end_year: int = 2024,
    data_url: str = None,
    schedules_path: str = None
) -> pd.DataFrame:
    """
    Build v1.0 training dataset from nfelo historical data + actual scores.

    Args:
        start_year: Start season year (default: 2009)
        end_year: End season year (default: 2024)
        data_url: Optional custom nfelo data URL
        schedules_path: Path to schedules.parquet (default: ./schedules.parquet)

    Returns:
        DataFrame with columns:
            - game_id: Unique identifier (YYYY_WW_AWAY_HOME)
            - season: NFL season year
            - week: Week number
            - away_team: Away team abbreviation
            - home_team: Home team abbreviation
            - nfelo_diff: ELO differential (home - away)
            - home_line_close: Vegas closing line
            - home_score: Actual home score
            - away_score: Actual away score
            - actual_margin: Target variable (home_score - away_score)
            - nfelo_diff: Primary feature (home ELO - away ELO)
            - home_points: Intentionally unused (for leak detection)
            - away_points: Intentionally unused (for leak detection)
            - home_margin: Intentionally unused (for leak detection)

    Expected shape:
        - Rows: 2000-4500 games (depending on year range)
        - Columns: 13
    """
    if data_url is None:
        data_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'

    if schedules_path is None:
        schedules_path = Path(__file__).parent.parent.parent / 'schedules.parquet'

    # Load nfelo historical data (for ratings and Vegas lines)
    df = pd.read_csv(data_url)

    # Extract season/week/teams from game_id
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to requested year range
    df = df[(df['season'] >= start_year) & (df['season'] <= end_year)].copy()

    # Filter to complete data (has ELO ratings and Vegas lines)
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()

    # Calculate primary feature: ELO differential
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Load actual game results from schedules
    schedules = pd.read_parquet(schedules_path)

    # Filter schedules to regular season games only (game_type == 'REG')
    schedules = schedules[schedules['game_type'] == 'REG'].copy()

    # Select needed columns from schedules (including spread_line for actual Vegas line)
    schedules_cols = schedules[['game_id', 'home_score', 'away_score', 'spread_line']].copy()

    # Merge in actual scores and spread line
    df = df.merge(schedules_cols, on='game_id', how='left')

    # Prefer spread_line from schedules (actual Vegas) over home_line_close from nfelo
    # spread_line is from home perspective (negative = home favored)
    df['home_line_close'] = df['spread_line'].fillna(df['home_line_close'])

    # Filter to games with actual scores (completed games only)
    df = df[df['home_score'].notna()].copy()
    df = df[df['away_score'].notna()].copy()

    # Calculate target: actual margin (home perspective)
    df['actual_margin'] = df['home_score'] - df['away_score']

    # Add intentionally unused columns (for leak detection tests)
    df['home_points'] = df['home_score']
    df['away_points'] = df['away_score']
    df['home_margin'] = df['actual_margin']

    # Select final columns
    output_cols = [
        'game_id',
        'season',
        'week',
        'away_team',
        'home_team',
        'nfelo_diff',
        'home_line_close',
        'actual_margin',
        'home_score',
        'away_score',
        'home_points',
        'away_points',
        'home_margin'
    ]

    return df[output_cols].reset_index(drop=True)
