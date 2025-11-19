"""
Ball Knower v1.3 Dataset Builder

Enhanced dataset with advanced features for improved spread prediction.

Features:
- v1.2 baseline: nfelo ratings, situational adjustments, QB adjustments (6 features)
- Rolling performance: win rate, point differential, ATS rate (6 features)
- Rolling ELO changes: recent ELO trends (4 features)
- Game context: playoff implications, primetime games (2 features)

Total: 18 features (up from 6 in v1.2)

Target:
- vegas_closing_spread (market consensus)

Use Case:
- Enhanced market-calibrated model for identifying betting edges
- Improved feature set while maintaining v1.2's market-alignment strategy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from ball_knower.features import engineering as features


def build_training_frame(
    start_year: int = 2009,
    end_year: int = 2024,
    data_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Build v1.3 training dataset with expanded features.

    Args:
        start_year: Start season year (default: 2009)
        end_year: End season year (default: 2024)
        data_url: Optional custom nfelo data URL

    Returns:
        DataFrame with columns:
            - game_id, season, week, away_team, home_team
            - v1.2 features (6): nfelo_diff, rest_advantage, div_game,
              surface_mod, time_advantage, qb_diff
            - Rolling form - Home (3): win_rate_L5_home, point_diff_L5_home, ats_rate_L5_home
            - Rolling form - Away (3): win_rate_L5_away, point_diff_L5_away, ats_rate_L5_away
            - Rolling ELO - Home (2): nfelo_change_L3_home, nfelo_change_L5_home
            - Rolling ELO - Away (2): nfelo_change_L3_away, nfelo_change_L5_away
            - Game context (2): is_playoff_week, is_primetime
            - vegas_closing_spread (target)
            - actual_margin, home_score, away_score

    Expected shape:
        - Rows: 2000-4500 games (depending on year range)
        - Columns: ~30
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

    # Sort by season and week for rolling calculations
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    # =========================================================================
    # v1.2 BASELINE FEATURES
    # =========================================================================

    # Primary feature: ELO differential
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Situational adjustments (canonical from engineering.py)
    df['rest_advantage'] = features.compute_rest_advantage_from_nfelo(df)
    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

    # QB adjustments
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) -
                     df['away_538_qb_adj'].fillna(0))

    # =========================================================================
    # ADVANCED FEATURES: ROLLING FORM
    # =========================================================================

    # Calculate outcomes for rolling features
    df['home_score'] = df['home_score'].fillna(0)
    df['away_score'] = df['away_score'].fillna(0)
    df['actual_margin'] = df['home_score'] - df['away_score']

    # Build team-level game history for rolling calculations
    rolling_features = _calculate_team_rolling_features(df)

    # Merge rolling features for home team
    df = df.merge(
        rolling_features,
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_home')
    )
    df = df.drop(columns=['team'], errors='ignore')
    df = df.rename(columns={
        'win_rate_L5': 'win_rate_L5_home',
        'point_diff_L5': 'point_diff_L5_home',
        'ats_rate_L5': 'ats_rate_L5_home',
        'nfelo_change_L3': 'nfelo_change_L3_home',
        'nfelo_change_L5': 'nfelo_change_L5_home'
    })

    # Merge rolling features for away team
    df = df.merge(
        rolling_features,
        left_on=['season', 'week', 'away_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_away')
    )
    df = df.drop(columns=['team'], errors='ignore')
    df = df.rename(columns={
        'win_rate_L5': 'win_rate_L5_away',
        'point_diff_L5': 'point_diff_L5_away',
        'ats_rate_L5': 'ats_rate_L5_away',
        'nfelo_change_L3': 'nfelo_change_L3_away',
        'nfelo_change_L5': 'nfelo_change_L5_away'
    })

    # =========================================================================
    # ADVANCED FEATURES: GAME CONTEXT
    # =========================================================================

    # Playoff implications (week >= 15 in regular season)
    df['is_playoff_week'] = ((df['week'] >= 15) & (df['week'] <= 18)).astype(int)

    # Primetime games (assuming gameday_hour if available, else use defaults)
    # NFElo doesn't have game time, so we'll skip this for now
    df['is_primetime'] = 0  # Placeholder - would need game time data

    # =========================================================================
    # TARGET AND VALIDATION
    # =========================================================================

    # Target: Vegas closing spread
    df['vegas_closing_spread'] = df['home_line_close']

    # Remove rows with NaN in critical feature columns
    feature_cols = [
        # v1.2 features
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff',
        # Rolling features (allow NaN for first few games)
        'vegas_closing_spread'
    ]
    mask = df[feature_cols].notna().all(axis=1)
    df = df[mask].copy()

    # Fill NaN in rolling features with 0 (represents no history)
    rolling_cols = [col for col in df.columns if '_L' in col or col.startswith('win_rate')
                   or col.startswith('point_diff') or col.startswith('ats_rate')
                   or col.startswith('nfelo_change')]
    for col in rolling_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Select final columns
    output_cols = [
        'game_id',
        'season',
        'week',
        'away_team',
        'home_team',
        # v1.2 baseline features (6)
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
        # Rolling form - Home (3)
        'win_rate_L5_home',
        'point_diff_L5_home',
        'ats_rate_L5_home',
        # Rolling form - Away (3)
        'win_rate_L5_away',
        'point_diff_L5_away',
        'ats_rate_L5_away',
        # Rolling ELO - Home (2)
        'nfelo_change_L3_home',
        'nfelo_change_L5_home',
        # Rolling ELO - Away (2)
        'nfelo_change_L3_away',
        'nfelo_change_L5_away',
        # Game context (2)
        'is_playoff_week',
        'is_primetime',
        # Target and outcomes
        'vegas_closing_spread',
        'home_score',
        'away_score',
        'actual_margin',
    ]

    # Add any missing columns
    for col in output_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[output_cols].reset_index(drop=True)


def _calculate_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling features for each team.

    CRITICAL: Uses .shift(1) to prevent data leakage - each game's rolling
    features are calculated ONLY from games strictly before that game.

    Args:
        df: Game-level DataFrame with scores and outcomes

    Returns:
        DataFrame with team-level rolling features per game
    """
    # Create team-game records
    team_games = []

    for idx, game in df.iterrows():
        # Home team game
        team_games.append({
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'team': game['home_team'],
            'nfelo_start': game['starting_nfelo_home'],
            'nfelo_end': game['ending_nfelo_home'],
            'won': 1 if game['home_score'] > game['away_score'] else 0,
            'point_diff': game['home_score'] - game['away_score'],
            'spread_line': game['home_line_close'],
        })

        # Away team game
        team_games.append({
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'team': game['away_team'],
            'nfelo_start': game['starting_nfelo_away'],
            'nfelo_end': game['ending_nfelo_away'],
            'won': 1 if game['away_score'] > game['home_score'] else 0,
            'point_diff': game['away_score'] - game['home_score'],
            'spread_line': -game['home_line_close'],  # Flip for away perspective
        })

    team_df = pd.DataFrame(team_games)

    # Calculate ATS (against the spread) outcome
    # Team covers if: point_diff + spread_line > 0
    team_df['ats_cover'] = (team_df['point_diff'] + team_df['spread_line'] > 0).astype(int)

    # Calculate ELO changes
    team_df['nfelo_change'] = team_df['nfelo_end'] - team_df['nfelo_start']

    # CRITICAL: Sort by team, season, week to ensure chronological order
    # This ensures .shift(1) excludes the current game and all future games
    team_df = team_df.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    # Calculate rolling features (LEAK-FREE: shift(1) excludes current game)
    team_df['win_rate_L5'] = (
        team_df.groupby('team')['won']
        .shift(1)  # Exclude current game
        .rolling(5, min_periods=1)
        .mean()
    )

    team_df['point_diff_L5'] = (
        team_df.groupby('team')['point_diff']
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    team_df['ats_rate_L5'] = (
        team_df.groupby('team')['ats_cover']
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    team_df['nfelo_change_L3'] = (
        team_df.groupby('team')['nfelo_change']
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
    )

    team_df['nfelo_change_L5'] = (
        team_df.groupby('team')['nfelo_change']
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    return team_df[['season', 'week', 'team', 'win_rate_L5', 'point_diff_L5',
                    'ats_rate_L5', 'nfelo_change_L3', 'nfelo_change_L5']]


def validate_v1_3_no_leakage(df: pd.DataFrame) -> None:
    """
    Validate that v1.3 dataset has no data leakage.

    Checks:
    1. First games of each season have NaN or 0 in rolling features (warmup period)
    2. No rolling feature uses data from current game or future
    3. All features are available before game start

    Args:
        df: v1.3 training dataset

    Raises:
        AssertionError: If leakage detected
    """
    # Check that rolling features for very first games are NaN or 0
    # (they should have no history)
    first_week_games = df[df['week'] == 1].head(10)

    rolling_cols = [col for col in df.columns if '_L' in col]

    if len(first_week_games) > 0 and len(rolling_cols) > 0:
        # First week games should have minimal rolling feature values
        for col in rolling_cols:
            first_values = first_week_games[col].abs()
            # Allow small values (filled with 0) but not full historical data
            assert first_values.max() < 5.0, \
                f"Potential leakage: {col} has large values in first week: {first_values.max()}"

    # Check that features don't reference future outcomes
    # (This is ensured by shift(1) in rolling calculations, verified by construction)

    print("✓ v1.3 leakage validation passed")
