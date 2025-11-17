"""
Advanced Feature Engineering for Ball Knower v2.0
==================================================

This module provides enhanced features for NFL spread prediction:

1. QB Adjustments:
   - Identify starting QBs
   - Calculate rolling QB performance (EPA, QBR)
   - Detect QB injuries/changes
   - Adjust team ratings for backup QBs

2. Recent Team Performance:
   - Rolling team EPA (offense/defense)
   - Momentum indicators
   - Form-weighted ratings (recent games matter more)

3. Next Gen Stats Integration:
   - QB efficiency metrics
   - Pressure stats
   - Advanced passing metrics

Author: Ball Knower Team
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TEAM NAME MAPPING (QBR uses full names, nflverse uses abbreviations)
# ============================================================================

TEAM_ABB_TO_FULL = {
    'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills',
    'CAR': 'Panthers', 'CHI': 'Bears', 'CIN': 'Bengals', 'CLE': 'Browns',
    'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GB': 'Packers',
    'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'KC': 'Chiefs',
    'LA': 'Rams', 'LAC': 'Chargers', 'LV': 'Raiders', 'MIA': 'Dolphins',
    'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints', 'NYG': 'Giants',
    'NYJ': 'Jets', 'PHI': 'Eagles', 'PIT': 'Steelers', 'SEA': 'Seahawks',
    'SF': '49ers', 'TB': 'Buccaneers', 'TEN': 'Titans', 'WAS': 'Commanders'
}

# ============================================================================
# QB PERFORMANCE FEATURES
# ============================================================================

def load_qb_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load QB performance data from all available sources.

    Returns:
        (qbr_weekly, ngs_passing, injuries)
    """
    # ESPN QBR weekly
    qbr_week = pd.read_parquet('espn_qbr_week.parquet')

    # Next Gen Stats passing
    ngs_passing = pd.read_parquet('ngs_passing.parquet')

    # Injuries data
    injuries = pd.read_parquet('injuries.parquet')

    return qbr_week, ngs_passing, injuries


def get_starting_qb(
    team: str,
    season: int,
    week: int,
    qbr_data: pd.DataFrame,
    ngs_data: pd.DataFrame
) -> Optional[str]:
    """
    Identify the starting QB for a team in a given week.

    Strategy:
    1. Look at NGS passing data (only starters get significant snaps)
    2. Fall back to QBR data
    3. Return QB with most dropbacks

    Args:
        team: Team abbreviation (e.g., 'BUF')
        season: Season year
        week: Week number
        qbr_data: ESPN QBR weekly data
        ngs_data: Next Gen Stats passing data

    Returns:
        Starting QB name, or None if not found
    """
    # Try NGS data first (more accurate for starters)
    ngs_qbs = ngs_data[
        (ngs_data['season'] == season) &
        (ngs_data['week'] == week) &
        (ngs_data['team_abbr'] == team)
    ].copy()

    if len(ngs_qbs) > 0:
        # Return QB with most attempts
        starter = ngs_qbs.sort_values('attempts', ascending=False).iloc[0]
        return starter['player_display_name']

    # Fall back to QBR data (convert team abbreviation to full name)
    team_full = TEAM_ABB_TO_FULL.get(team, team)
    qbr_qbs = qbr_data[
        (qbr_data['season'] == season) &
        (qbr_data['game_week'] == week) &
        (qbr_data['team'] == team_full)
    ].copy()

    if len(qbr_qbs) > 0:
        # Return QB with most plays
        starter = qbr_qbs.sort_values('qb_plays', ascending=False).iloc[0]
        return starter['name_display']

    return None


def get_qb_rolling_stats(
    team: str,
    season: int,
    week: int,
    qbr_data: pd.DataFrame,
    ngs_data: pd.DataFrame,
    lookback: int = 3
) -> Dict[str, float]:
    """
    Calculate rolling QB performance metrics (last N games).

    Args:
        team: Team abbreviation
        season: Season year
        week: Current week
        qbr_data: ESPN QBR weekly data
        ngs_data: Next Gen Stats passing data
        lookback: Number of games to look back (default: 3)

    Returns:
        Dictionary of rolling stats:
        - qb_rolling_epa: Average EPA per play (last N games)
        - qb_rolling_qbr: Average QBR (last N games)
        - qb_rolling_cpoe: Completion % over expected (last N games)
        - qb_rolling_pressure_pct: Pressure rate (last N games)
        - qb_recent_form: Trend indicator (-1 to +1)
    """
    # Get QBR data for this team (up to current week)
    team_full = TEAM_ABB_TO_FULL.get(team, team)
    team_qbr = qbr_data[
        (qbr_data['season'] == season) &
        (qbr_data['team'] == team_full) &
        (qbr_data['game_week'] < week)  # Only prior weeks
    ].copy()

    # Get NGS data for this team
    team_ngs = ngs_data[
        (ngs_data['season'] == season) &
        (ngs_data['team_abbr'] == team) &
        (ngs_data['week'] < week)  # Only prior weeks
    ].copy()

    # Get last N weeks
    recent_qbr = team_qbr.sort_values('game_week', ascending=False).head(lookback)
    recent_ngs = team_ngs.sort_values('week', ascending=False).head(lookback)

    # Calculate rolling averages
    stats = {
        'qb_rolling_epa': 0.0,
        'qb_rolling_qbr': 50.0,  # League average
        'qb_rolling_cpoe': 0.0,
        'qb_rolling_pressure_pct': 0.25,  # Typical pressure rate
        'qb_recent_form': 0.0
    }

    if len(recent_qbr) > 0:
        # EPA per play
        total_plays = recent_qbr['qb_plays'].sum()
        total_epa = recent_qbr['epa_total'].sum()
        stats['qb_rolling_epa'] = total_epa / total_plays if total_plays > 0 else 0.0

        # QBR
        stats['qb_rolling_qbr'] = recent_qbr['qbr_total'].mean()

        # Form trend (compare first half to second half of lookback window)
        if len(recent_qbr) >= 2:
            mid = len(recent_qbr) // 2
            first_half_qbr = recent_qbr.iloc[mid:]['qbr_total'].mean()
            second_half_qbr = recent_qbr.iloc[:mid]['qbr_total'].mean()
            # Normalize to -1 to +1
            stats['qb_recent_form'] = (second_half_qbr - first_half_qbr) / 100.0

    if len(recent_ngs) > 0:
        # Completion % over expected
        stats['qb_rolling_cpoe'] = recent_ngs['completion_percentage_above_expectation'].mean()

    return stats


def detect_qb_change(
    team: str,
    season: int,
    week: int,
    qbr_data: pd.DataFrame,
    ngs_data: pd.DataFrame
) -> Tuple[bool, float]:
    """
    Detect if team has changed QBs recently and estimate impact.

    Args:
        team: Team abbreviation
        season: Season year
        week: Current week
        qbr_data: ESPN QBR weekly data
        ngs_data: Next Gen Stats passing data

    Returns:
        (qb_changed: bool, qb_change_penalty: float)
        - qb_changed: True if QB changed in last 2 weeks
        - qb_change_penalty: Estimated point impact (negative = worse)
    """
    if week < 3:
        return False, 0.0

    # Get last 3 weeks of starters
    starters = []
    for w in range(max(1, week - 3), week):
        starter = get_starting_qb(team, season, w, qbr_data, ngs_data)
        if starter:
            starters.append(starter)

    if len(starters) < 2:
        return False, 0.0

    # Check if current starter is different from previous weeks
    current_starter = starters[-1] if len(starters) > 0 else None
    previous_starters = starters[:-1] if len(starters) > 1 else []

    if not current_starter or not previous_starters:
        return False, 0.0

    # QB changed if current != most recent previous
    qb_changed = current_starter != previous_starters[-1]

    if not qb_changed:
        return False, 0.0

    # Estimate impact: compare QB performance
    # Get current QB's recent stats
    current_qb_stats = qbr_data[
        (qbr_data['season'] == season) &
        (qbr_data['name_display'] == current_starter) &
        (qbr_data['game_week'] < week)
    ]

    # Get previous QB's recent stats
    prev_qb_name = previous_starters[-1]
    prev_qb_stats = qbr_data[
        (qbr_data['season'] == season) &
        (qbr_data['name_display'] == prev_qb_name) &
        (qbr_data['game_week'] < week)
    ]

    # Calculate QBR difference
    current_qbr = current_qb_stats['qbr_total'].mean() if len(current_qb_stats) > 0 else 50.0
    prev_qbr = prev_qb_stats['qbr_total'].mean() if len(prev_qb_stats) > 0 else 50.0

    # Convert QBR difference to point impact
    # Rule of thumb: 10 QBR points ~ 1-2 points in spread
    qbr_diff = current_qbr - prev_qbr
    qb_change_penalty = (qbr_diff / 10.0) * 1.5

    # Cap the impact
    qb_change_penalty = np.clip(qb_change_penalty, -5.0, 5.0)

    return qb_changed, qb_change_penalty


# ============================================================================
# ROLLING TEAM EPA FEATURES
# ============================================================================

def load_team_epa() -> pd.DataFrame:
    """Load team EPA data."""
    return pd.read_csv('team_week_epa_2013_2024.csv')


def get_team_rolling_epa(
    team: str,
    season: int,
    week: int,
    epa_data: pd.DataFrame,
    lookback: int = 5
) -> Dict[str, float]:
    """
    Calculate rolling team EPA metrics (last N games).

    Args:
        team: Team abbreviation
        season: Season year
        week: Current week
        epa_data: Team EPA weekly data
        lookback: Number of games to look back (default: 5)

    Returns:
        Dictionary of rolling stats:
        - rolling_off_epa: Offensive EPA per play (last N games)
        - rolling_def_epa: Defensive EPA per play (last N games)
        - rolling_off_success_rate: Offensive success rate
        - rolling_def_success_rate: Defensive success rate
        - team_momentum: Trend indicator (-1 to +1)
    """
    # Get team data up to current week
    team_data = epa_data[
        (epa_data['season'] == season) &
        (epa_data['team'] == team) &
        (epa_data['week'] < week)
    ].copy()

    # Get last N weeks
    recent = team_data.sort_values('week', ascending=False).head(lookback)

    stats = {
        'rolling_off_epa': 0.0,
        'rolling_def_epa': 0.0,
        'rolling_off_success_rate': 0.50,
        'rolling_def_success_rate': 0.50,
        'team_momentum': 0.0
    }

    if len(recent) == 0:
        return stats

    # Calculate rolling averages
    stats['rolling_off_epa'] = recent['off_epa_per_play'].mean()
    stats['rolling_def_epa'] = recent['def_epa_per_play'].mean()
    stats['rolling_off_success_rate'] = recent['off_success_rate'].mean()
    stats['rolling_def_success_rate'] = recent['def_success_rate'].mean()

    # Momentum: compare recent vs earlier performance
    if len(recent) >= lookback:
        mid = lookback // 2
        early_epa = recent.iloc[mid:]['off_epa_per_play'].mean()
        late_epa = recent.iloc[:mid]['off_epa_per_play'].mean()
        # Normalize to -1 to +1
        stats['team_momentum'] = np.clip((late_epa - early_epa) / 0.2, -1.0, 1.0)

    return stats


# ============================================================================
# NEXT GEN STATS FEATURES
# ============================================================================

def get_ngs_qb_features(
    team: str,
    season: int,
    week: int,
    ngs_data: pd.DataFrame,
    lookback: int = 3
) -> Dict[str, float]:
    """
    Calculate Next Gen Stats QB features (last N games).

    Args:
        team: Team abbreviation
        season: Season year
        week: Current week
        ngs_data: Next Gen Stats passing data
        lookback: Number of games to look back

    Returns:
        Dictionary of NGS features:
        - avg_time_to_throw: Average time to throw (seconds)
        - completion_pct_above_exp: Completion % over expected
        - aggressiveness: Deep ball tendency (0-1)
        - avg_air_yards: Average depth of target
    """
    # Get team NGS data
    team_ngs = ngs_data[
        (ngs_data['season'] == season) &
        (ngs_data['team_abbr'] == team) &
        (ngs_data['week'] < week)
    ].copy()

    # Get last N weeks
    recent = team_ngs.sort_values('week', ascending=False).head(lookback)

    stats = {
        'avg_time_to_throw': 2.5,  # League average
        'completion_pct_above_exp': 0.0,
        'aggressiveness': 0.10,  # League average
        'avg_air_yards': 7.5  # League average
    }

    if len(recent) == 0:
        return stats

    # Calculate weighted averages (weight by attempts)
    total_attempts = recent['attempts'].sum()

    if total_attempts > 0:
        stats['avg_time_to_throw'] = (
            (recent['avg_time_to_throw'] * recent['attempts']).sum() / total_attempts
        )
        stats['completion_pct_above_exp'] = (
            (recent['completion_percentage_above_expectation'] * recent['attempts']).sum() / total_attempts
        )
        stats['aggressiveness'] = (
            (recent['aggressiveness'] * recent['attempts']).sum() / total_attempts
        )
        stats['avg_air_yards'] = (
            (recent['avg_intended_air_yards'] * recent['attempts']).sum() / total_attempts
        )

    return stats


# ============================================================================
# MATCHUP FEATURES
# ============================================================================

def calculate_matchup_features(
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    qbr_data: pd.DataFrame,
    ngs_data: pd.DataFrame,
    epa_data: pd.DataFrame,
    lookback_qb: int = 3,
    lookback_team: int = 5
) -> Dict[str, float]:
    """
    Calculate all advanced features for a matchup.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: Season year
        week: Week number
        qbr_data: ESPN QBR data
        ngs_data: Next Gen Stats data
        epa_data: Team EPA data
        lookback_qb: QB rolling window (games)
        lookback_team: Team rolling window (games)

    Returns:
        Dictionary of all advanced features
    """
    features = {}

    # QB features (home)
    home_qb_stats = get_qb_rolling_stats(home_team, season, week, qbr_data, ngs_data, lookback_qb)
    for key, val in home_qb_stats.items():
        features[f'home_{key}'] = val

    # QB features (away)
    away_qb_stats = get_qb_rolling_stats(away_team, season, week, qbr_data, ngs_data, lookback_qb)
    for key, val in away_qb_stats.items():
        features[f'away_{key}'] = val

    # QB change detection
    home_qb_changed, home_qb_penalty = detect_qb_change(home_team, season, week, qbr_data, ngs_data)
    away_qb_changed, away_qb_penalty = detect_qb_change(away_team, season, week, qbr_data, ngs_data)

    features['home_qb_changed'] = 1.0 if home_qb_changed else 0.0
    features['away_qb_changed'] = 1.0 if away_qb_changed else 0.0
    features['qb_change_diff'] = home_qb_penalty - away_qb_penalty

    # Team EPA features (home)
    home_epa = get_team_rolling_epa(home_team, season, week, epa_data, lookback_team)
    for key, val in home_epa.items():
        features[f'home_{key}'] = val

    # Team EPA features (away)
    away_epa = get_team_rolling_epa(away_team, season, week, epa_data, lookback_team)
    for key, val in away_epa.items():
        features[f'away_{key}'] = val

    # Differential features (home - away)
    features['rolling_epa_diff'] = features['home_rolling_off_epa'] - features['away_rolling_off_epa']
    features['rolling_def_epa_diff'] = features['home_rolling_def_epa'] - features['away_rolling_def_epa']
    features['qb_rolling_epa_diff'] = features['home_qb_rolling_epa'] - features['away_qb_rolling_epa']
    features['qb_rolling_qbr_diff'] = features['home_qb_rolling_qbr'] - features['away_qb_rolling_qbr']
    features['momentum_diff'] = features['home_team_momentum'] - features['away_team_momentum']

    # NGS features (home)
    home_ngs = get_ngs_qb_features(home_team, season, week, ngs_data, lookback_qb)
    for key, val in home_ngs.items():
        features[f'home_{key}'] = val

    # NGS features (away)
    away_ngs = get_ngs_qb_features(away_team, season, week, ngs_data, lookback_qb)
    for key, val in away_ngs.items():
        features[f'away_{key}'] = val

    # NGS differential features
    features['cpoe_diff'] = features['home_completion_pct_above_exp'] - features['away_completion_pct_above_exp']

    return features


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def add_advanced_features_to_games(
    games_df: pd.DataFrame,
    season: int,
    week: int,
    qb_lookback: int = 3,
    team_lookback: int = 5
) -> pd.DataFrame:
    """
    Add all advanced features to a games dataframe.

    Args:
        games_df: DataFrame with columns ['home_team', 'away_team']
        season: Season year
        week: Week number
        qb_lookback: QB rolling window
        team_lookback: Team rolling window

    Returns:
        DataFrame with advanced features added
    """
    # Load data
    qbr_data, ngs_data, _ = load_qb_data()
    epa_data = load_team_epa()

    # Calculate features for each game
    all_features = []

    for idx, game in games_df.iterrows():
        features = calculate_matchup_features(
            home_team=game['home_team'],
            away_team=game['away_team'],
            season=season,
            week=week,
            qbr_data=qbr_data,
            ngs_data=ngs_data,
            epa_data=epa_data,
            lookback_qb=qb_lookback,
            lookback_team=team_lookback
        )
        all_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Combine with original games
    result = pd.concat([games_df.reset_index(drop=True), features_df], axis=1)

    return result
