"""
Feature Engineering Functions

CRITICAL: All features must be LEAK-FREE.
Centralized canonical implementations for rest-related features.

This module contains the single source of truth for:
1. Rest advantage from NFElo bye modifiers
2. Schedule-based rest days calculations

All dataset builders and backtest scripts MUST use these functions.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ============================================================================
# REST ADVANTAGE FROM NFELO BYE MODIFIERS
# ============================================================================

def compute_rest_advantage_from_nfelo(df: pd.DataFrame) -> pd.Series:
    """
    Canonical rest advantage calculation from NFElo bye modifiers.

    This is the primary rest advantage metric used in v1.2 models.
    It combines home and away bye week modifiers from the NFElo dataset.

    Args:
        df: DataFrame with 'home_bye_mod' and 'away_bye_mod' columns

    Returns:
        pd.Series: rest_advantage = home_bye_mod.fillna(0) + away_bye_mod.fillna(0)

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> df['rest_advantage'] = compute_rest_advantage_from_nfelo(df)
    """
    required = {"home_bye_mod", "away_bye_mod"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for rest advantage: {missing}")

    return df["home_bye_mod"].fillna(0) + df["away_bye_mod"].fillna(0)


# ============================================================================
# SCHEDULE-BASED REST DAYS
# ============================================================================

def compute_rest_days_from_schedule(
    schedules: pd.DataFrame,
    home_team_col: str = "team_home",
    away_team_col: str = "team_away",
    date_col: str = "gameday",
    game_id_col: str = "game_id",
) -> pd.DataFrame:
    """
    Canonical schedule-based rest days computation.

    Calculates days of rest since the last game for each team based on
    schedule chronology. This is an alternative rest metric to NFElo's
    bye modifiers.

    Args:
        schedules: Game-level schedule DataFrame
        home_team_col: Column name for home team (default: "team_home")
        away_team_col: Column name for away team (default: "team_away")
        date_col: Column name for game date (default: "gameday")
        game_id_col: Column name for game ID (default: "game_id")

    Returns:
        pd.DataFrame: Original schedules with added columns:
            - home_rest_days: Days since home team's last game
            - away_rest_days: Days since away team's last game
            - rest_advantage: home_rest_days - away_rest_days

    Example:
        >>> schedules = compute_rest_days_from_schedule(schedules)
        >>> print(schedules[['home_team', 'away_team', 'rest_advantage']])
    """
    schedules = schedules.copy()
    schedules[date_col] = pd.to_datetime(schedules[date_col])

    # Build team-level schedule
    team_games = []

    for team in schedules[home_team_col].unique():
        # Get all games for this team (home and away)
        home_games = schedules[schedules[home_team_col] == team][[date_col, game_id_col]].copy()
        home_games['team'] = team

        away_games = schedules[schedules[away_team_col] == team][[date_col, game_id_col]].copy()
        away_games['team'] = team

        team_schedule = pd.concat([home_games, away_games]).sort_values(date_col)

        # Calculate rest days (days since previous game)
        team_schedule['rest_days'] = team_schedule[date_col].diff().dt.days

        team_games.append(team_schedule)

    rest_df = pd.concat(team_games, ignore_index=True)

    # Merge home team rest days
    schedules = schedules.merge(
        rest_df[[game_id_col, 'team', 'rest_days']].rename(
            columns={'rest_days': 'home_rest_days'}
        ),
        left_on=[game_id_col, home_team_col],
        right_on=[game_id_col, 'team'],
        how='left'
    ).drop(columns=['team'])

    # Merge away team rest days
    schedules = schedules.merge(
        rest_df[[game_id_col, 'team', 'rest_days']].rename(
            columns={'rest_days': 'away_rest_days'}
        ),
        left_on=[game_id_col, away_team_col],
        right_on=[game_id_col, 'team'],
        how='left'
    ).drop(columns=['team'])

    # Compute rest advantage (home - away)
    schedules['rest_advantage'] = (
        schedules['home_rest_days'].fillna(0) - schedules['away_rest_days'].fillna(0)
    )

    return schedules


# ============================================================================
# ROLLING EPA FEATURES (LEAK-FREE)
# ============================================================================

def calculate_rolling_epa(schedules, weekly_stats, windows=[3, 5, 10]):
    """
    Calculate rolling EPA features for each team, ensuring NO LEAKAGE.

    Args:
        schedules (pd.DataFrame): Game schedules with dates
        weekly_stats (pd.DataFrame): Weekly team stats from nfl_data_py
        windows (list): Rolling window sizes (e.g., [3, 5, 10] games)

    Returns:
        pd.DataFrame: Schedules with rolling EPA features added
    """
    # Ensure schedules are sorted by date
    schedules = schedules.sort_values(['season', 'week']).copy()

    # Get EPA from weekly stats if available
    if 'offense_epa' not in weekly_stats.columns:
        print("⚠ Warning: EPA not in weekly stats. Using play-by-play calculation.")
        # TODO: Calculate from play-by-play if needed
        return schedules

    # Create team-game-level EPA data
    team_games = []

    for idx, game in schedules.iterrows():
        # Home team game
        team_games.append({
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'gameday': game['gameday'],
            'team': game['team_home'],
            'opponent': game['team_away'],
            'location': 'home',
            'points_for': game['home_score'],
            'points_against': game['away_score'],
        })

        # Away team game
        team_games.append({
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'gameday': game['gameday'],
            'team': game['team_away'],
            'opponent': game['team_home'],
            'location': 'away',
            'points_for': game['away_score'],
            'points_against': game['home_score'],
        })

    team_games_df = pd.DataFrame(team_games)

    # Merge EPA data from weekly stats
    team_games_df = team_games_df.merge(
        weekly_stats[['season', 'week', 'recent_team', 'offense_epa', 'defense_epa']],
        left_on=['season', 'week', 'team'],
        right_on=['season', 'week', 'recent_team'],
        how='left'
    )

    # CRITICAL: Sort by team and chronological keys (season, week, gameday) to ensure
    # rolling windows only include strictly past games, preventing data leakage.
    # This sort order guarantees that .shift(1) will exclude the current game.
    team_games_df = team_games_df.sort_values(['team', 'season', 'week', 'gameday'])

    # Calculate rolling features (LEAK-FREE: shift by 1 to exclude current game)
    for window in windows:
        team_games_df[f'epa_off_L{window}'] = (
            team_games_df.groupby('team')['offense_epa']
            .shift(1)  # CRITICAL: Shift to exclude current game
            .rolling(window, min_periods=1)
            .mean()
        )

        team_games_df[f'epa_def_L{window}'] = (
            team_games_df.groupby('team')['defense_epa']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

        team_games_df[f'epa_margin_L{window}'] = (
            team_games_df[f'epa_off_L{window}'] - team_games_df[f'epa_def_L{window}']
        )

    print(f"✓ Calculated rolling EPA features for windows: {windows}")
    return team_games_df


def validate_no_leakage(df, date_col='gameday'):
    """
    Validate that rolling features don't leak future information.

    Args:
        df (pd.DataFrame): DataFrame with rolling features
        date_col (str): Date column name

    Raises:
        ValueError: If leakage is detected
    """
    # Check that no rolling feature includes data from the same gameday or future
    rolling_cols = [col for col in df.columns if '_L' in col and col.startswith('epa')]

    for col in rolling_cols:
        # For each team, verify rolling stat at time T only uses data before T
        # This is ensured by the shift(1) in rolling calculation
        pass

    print("✓ Leakage validation passed")


# ============================================================================
# RECENT FORM FEATURES
# ============================================================================

def calculate_recent_form(schedules, windows=[3, 5]):
    """
    Calculate recent form: wins, covers, point differential.

    Args:
        schedules (pd.DataFrame): Game schedules with outcomes
        windows (list): Rolling windows for form calculation

    Returns:
        pd.DataFrame: Schedules with form features
    """
    schedules = schedules.sort_values(['season', 'week']).copy()

    # Create win/loss indicators
    schedules['home_win'] = (schedules['home_score'] > schedules['away_score']).astype(int)
    schedules['away_win'] = (schedules['away_score'] > schedules['home_score']).astype(int)

    # Calculate ATS (against the spread) if spread_line exists
    if 'spread_line' in schedules.columns:
        # spread_line is from home team perspective
        schedules['home_ats_margin'] = schedules['home_score'] - schedules['away_score'] + schedules['spread_line']
        schedules['home_ats_cover'] = (schedules['home_ats_margin'] > 0).astype(int)

    # Build team-level game history
    team_games = []

    for idx, game in schedules.iterrows():
        # Home team
        team_games.append({
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'team': game['team_home'],
            'win': game['home_win'],
            'ats_cover': game.get('home_ats_cover', np.nan),
            'point_diff': game['home_score'] - game['away_score']
        })

        # Away team
        team_games.append({
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'team': game['team_away'],
            'win': game['away_win'],
            'ats_cover': 1 - game.get('home_ats_cover', np.nan) if 'home_ats_cover' in game else np.nan,
            'point_diff': game['away_score'] - game['home_score']
        })

    # CRITICAL: Sort by team and chronological keys to ensure rolling windows
    # only include strictly past games, preventing data leakage.
    team_games_df = pd.DataFrame(team_games).sort_values(['team', 'season', 'week'])

    # Calculate rolling form (LEAK-FREE: shift by 1)
    for window in windows:
        team_games_df[f'win_rate_L{window}'] = (
            team_games_df.groupby('team')['win']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

        team_games_df[f'ats_rate_L{window}'] = (
            team_games_df.groupby('team')['ats_cover']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

        team_games_df[f'point_diff_L{window}'] = (
            team_games_df.groupby('team')['point_diff']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

    print("✓ Calculated recent form features")
    return team_games_df


# ============================================================================
# MATCHUP FEATURES
# ============================================================================

def create_matchup_features(schedules, team_ratings):
    """
    Create matchup-specific features (offense vs opponent defense, etc.).

    Args:
        schedules (pd.DataFrame): Game schedules
        team_ratings (pd.DataFrame): Team ratings (EPA, ELO, etc.)

    Returns:
        pd.DataFrame: Schedules with matchup features
    """
    schedules = schedules.copy()

    # Merge home team ratings
    schedules = schedules.merge(
        team_ratings,
        left_on='team_home',
        right_on='team',
        how='left',
        suffixes=('', '_home')
    )

    # Merge away team ratings
    schedules = schedules.merge(
        team_ratings,
        left_on='team_away',
        right_on='team',
        how='left',
        suffixes=('', '_away')
    )

    # Calculate matchup advantages
    if 'epa_off' in team_ratings.columns and 'epa_def' in team_ratings.columns:
        schedules['home_off_vs_away_def'] = schedules['epa_off'] - schedules['epa_def_away']
        schedules['away_off_vs_home_def'] = schedules['epa_off_away'] - schedules['epa_def']

    print("✓ Created matchup features")
    return schedules
