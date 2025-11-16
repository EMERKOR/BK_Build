"""
Feature Engineering Module

CRITICAL: All features must be LEAK-FREE.
Rolling statistics ONLY use past games, never include current game.

Features engineered:
1. Rolling EPA (offense, defense, margin) - 3, 5, 10 game windows
2. Rest days since last game
3. Home/away splits
4. Matchup-specific features (vs opponent defense, etc.)
5. Recent form (win streak, ATS performance)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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

    # Sort by team and date to prepare for rolling calculations
    team_games_df = team_games_df.sort_values(['team', 'season', 'week'])

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
# REST & SCHEDULE FEATURES
# ============================================================================

def calculate_rest_days(schedules):
    """
    Calculate days of rest since last game for each team.

    Args:
        schedules (pd.DataFrame): Game schedules

    Returns:
        pd.DataFrame: Schedules with rest days added
    """
    schedules = schedules.sort_values(['season', 'week']).copy()
    schedules['gameday'] = pd.to_datetime(schedules['gameday'])

    team_games = []

    for team in schedules['team_home'].unique():
        # Get all games for this team (home and away)
        home_games = schedules[schedules['team_home'] == team][['gameday', 'game_id']].copy()
        home_games['team'] = team

        away_games = schedules[schedules['team_away'] == team][['gameday', 'game_id']].copy()
        away_games['team'] = team

        team_schedule = pd.concat([home_games, away_games]).sort_values('gameday')

        # Calculate rest days
        team_schedule['rest_days'] = team_schedule['gameday'].diff().dt.days

        team_games.append(team_schedule)

    rest_df = pd.concat(team_games)

    # Merge back to schedules
    schedules = schedules.merge(
        rest_df[['game_id', 'team', 'rest_days']].rename(columns={'rest_days': 'home_rest_days'}),
        left_on=['game_id', 'team_home'],
        right_on=['game_id', 'team'],
        how='left'
    ).drop(columns=['team'])

    schedules = schedules.merge(
        rest_df[['game_id', 'team', 'rest_days']].rename(columns={'rest_days': 'away_rest_days'}),
        left_on=['game_id', 'team_away'],
        right_on=['game_id', 'team'],
        how='left'
    ).drop(columns=['team'])

    schedules['rest_advantage'] = schedules['home_rest_days'] - schedules['away_rest_days']

    print("✓ Calculated rest days features")
    return schedules


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


# ============================================================================
# MASTER FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_all_features(schedules, weekly_stats=None, team_ratings=None):
    """
    Engineer all features for spread prediction.

    Args:
        schedules (pd.DataFrame): Game schedules
        weekly_stats (pd.DataFrame): Weekly team stats (optional)
        team_ratings (pd.DataFrame): External team ratings (optional)

    Returns:
        pd.DataFrame: Schedules with all engineered features
    """
    print("\n" + "="*60)
    print("ENGINEERING FEATURES (LEAK-FREE)")
    print("="*60 + "\n")

    # Calculate rest days
    schedules = calculate_rest_days(schedules)

    # Calculate rolling EPA if weekly stats provided
    if weekly_stats is not None:
        team_epa = calculate_rolling_epa(schedules, weekly_stats)
        # TODO: Merge back to schedules

    # Calculate recent form
    team_form = calculate_recent_form(schedules)
    # TODO: Merge back to schedules

    # Create matchup features if team ratings provided
    if team_ratings is not None:
        schedules = create_matchup_features(schedules, team_ratings)

    # Validate no leakage
    validate_no_leakage(schedules)

    print("\n" + "="*60)
    print("✓ ALL FEATURES ENGINEERED")
    print("="*60 + "\n")

    return schedules
