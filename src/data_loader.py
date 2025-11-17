"""
Data Loading Module

DEPRECATED: New code should use `from ball_knower.io import loaders` instead.

This module provides compatibility with legacy code and historical data loading.
Current-week data loaders now forward to the unified ball_knower.io.loaders API
when available.

IMPORTANT: Week and season are now passed as function parameters (no longer from config).
Archive scripts still use hardcoded paths for reproducibility.

Handles loading and initial cleaning of all data sources:
- nfl_data_py historical data (still handled here)
- nfelo ratings and stats (forwarded to unified loader)
- Substack ratings and projections (forwarded to unified loader)
- Reference data (coaches, AV)

All data is normalized to use standard team abbreviations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from .team_mapping import normalize_team_name, normalize_team_column
from .config import NFL_HEAD_COACHES, CURRENT_SEASON_DIR

warnings.filterwarnings('ignore', category=FutureWarning)

# Default season and week for legacy compatibility
# Archive scripts should pass these explicitly
DEFAULT_SEASON = 2025
DEFAULT_WEEK = 11

# Try to import unified loader module
try:
    from ball_knower.io import loaders as new_loaders
    NEW_LOADERS_AVAILABLE = True
except ImportError:
    NEW_LOADERS_AVAILABLE = False


# ============================================================================
# NFL_DATA_PY LOADERS
# ============================================================================

def load_historical_schedules(start_year, end_year):
    """
    Load historical NFL schedules from nfl_data_py.

    Args:
        start_year (int): Start year (e.g., 2015)
        end_year (int): End year (e.g., 2024)

    Returns:
        pd.DataFrame: Schedule data with columns:
            - game_id, season, week, gameday, home_team, away_team
            - home_score, away_score, spread_line, total_line
            - roof, surface, temp, wind
    """
    print(f"Loading schedules {start_year}-{end_year} from nflverse...")

    # Use nflverse GitHub raw data URL (works, doesn't get 403)
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

    schedules = pd.read_csv(url)

    # Filter for requested years
    schedules = schedules[
        (schedules['season'] >= start_year) &
        (schedules['season'] <= end_year)
    ].copy()

    # Keep only regular season and playoffs (drop preseason)
    if 'game_type' in schedules.columns:
        schedules = schedules[schedules['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])].copy()

    # Rename columns for clarity if needed
    if 'home_team' in schedules.columns:
        schedules.rename(columns={
            'home_team': 'team_home',
            'away_team': 'team_away',
        }, inplace=True)

    print(f"✓ Loaded {len(schedules):,} games from {start_year}-{end_year}")
    return schedules


def load_historical_team_stats(start_year, end_year, stat_type='weekly'):
    """
    Load historical team-level stats from nfl_data_py.

    Args:
        start_year (int): Start year
        end_year (int): End year
        stat_type (str): 'weekly' or 'seasonal'

    Returns:
        pd.DataFrame: Team stats by week/season
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError("nfl_data_py not installed")

    print(f"Loading {stat_type} team stats {start_year}-{end_year}...")
    years = list(range(start_year, end_year + 1))

    if stat_type == 'weekly':
        stats = nfl.import_weekly_data(years, downcast=False)
    else:
        stats = nfl.import_seasonal_data(years)

    print(f"✓ Loaded {len(stats):,} team-week records")
    return stats


# ============================================================================
# NFELO LOADERS
# ============================================================================

def _legacy_load_nfelo_power_ratings(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    LEGACY: Load nfelo power ratings from hardcoded file path.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Team power ratings with standardized team column
    """
    # Hardcoded path for legacy archive compatibility
    legacy_path = CURRENT_SEASON_DIR / f'power_ratings_nfelo_{season}_week_{week}.csv'
    df = pd.read_csv(legacy_path)

    # Standardize team names
    df = normalize_team_column(df, column_name='Team', new_column_name='team')

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    print(f"✓ Loaded nfelo power ratings: {len(df)} teams")
    return df


def load_nfelo_power_ratings(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load nfelo power ratings.

    Forwards to unified loader when available, otherwise uses legacy implementation.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Team power ratings with standardized team column
    """
    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_power_ratings("nfelo", season, week)
    return _legacy_load_nfelo_power_ratings(season, week)


def _legacy_load_nfelo_epa_tiers(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    LEGACY: Load nfelo EPA tiers from hardcoded file path.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: EPA metrics by team
    """
    legacy_path = CURRENT_SEASON_DIR / f'epa_tiers_nfelo_{season}_week_{week}.csv'
    df = pd.read_csv(legacy_path)

    # Standardize team names
    df = normalize_team_column(df, column_name='Team', new_column_name='team')

    # Rename EPA columns for clarity
    df.rename(columns={
        'EPA/Play': 'epa_off',
        'EPA/Play Against': 'epa_def'
    }, inplace=True)

    # Calculate EPA margin
    df['epa_margin'] = df['epa_off'] - df['epa_def']

    print(f"✓ Loaded nfelo EPA tiers: {len(df)} teams")
    return df


def load_nfelo_epa_tiers(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load nfelo EPA tiers (offensive/defensive EPA per play).

    Forwards to unified loader when available, otherwise uses legacy implementation.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: EPA metrics by team
    """
    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_epa_tiers("nfelo", season, week)
    return _legacy_load_nfelo_epa_tiers(season, week)


def load_nfelo_qb_rankings():
    """
    Load nfelo QB rankings.

    Returns:
        pd.DataFrame: QB rankings by team
    """
    df = pd.read_csv(NFELO_QB_RANKINGS)

    # Check if Team column exists
    if 'Team' in df.columns:
        df = normalize_team_column(df, column_name='Team', new_column_name='team')
    elif 'QB' in df.columns:
        # QB rankings may have QB name instead of team
        # We'll handle this later when we have more info
        pass

    print(f"✓ Loaded nfelo QB rankings: {len(df)} records")
    return df


def _legacy_load_nfelo_sos(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    LEGACY: Load nfelo strength of schedule data from hardcoded file path.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: SOS metrics by team
    """
    legacy_path = CURRENT_SEASON_DIR / f'strength_of_schedule_nfelo_{season}_week_{week}.csv'
    df = pd.read_csv(legacy_path)

    df = normalize_team_column(df, column_name='Team', new_column_name='team')

    print(f"✓ Loaded nfelo SOS: {len(df)} teams")
    return df


def load_nfelo_sos(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load nfelo strength of schedule data.

    Forwards to unified loader when available, otherwise uses legacy implementation.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: SOS metrics by team
    """
    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_strength_of_schedule("nfelo", season, week)
    return _legacy_load_nfelo_sos(season, week)


# ============================================================================
# SUBSTACK LOADERS
# ============================================================================

def _legacy_load_substack_power_ratings(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    LEGACY: Load Substack power ratings from hardcoded file path.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Power ratings with Off/Def/Ovr scores
    """
    legacy_path = CURRENT_SEASON_DIR / f'power_ratings_substack_{season}_week_{week}.csv'
    # File has 2 header rows - skip first, use second as column names
    df = pd.read_csv(legacy_path, encoding='utf-8-sig', skiprows=1)

    # Remove weird header artifacts (X.1, X.2, etc.)
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Standardize team names (should be 'Team' column now)
    if 'Team' in df.columns:
        df = normalize_team_column(df, column_name='Team', new_column_name='team')
    else:
        # First column should be team
        first_col = df.columns[0]
        df = normalize_team_column(df, column_name=first_col, new_column_name='team')

    # Convert numeric columns
    for col in ['Off.', 'Def.', 'Ovr.']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"✓ Loaded Substack power ratings: {len(df)} teams")
    return df


def load_substack_power_ratings(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load Substack power ratings.

    Forwards to unified loader when available, otherwise uses legacy implementation.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Power ratings with Off/Def/Ovr scores
    """
    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_power_ratings("substack", season, week)
    return _legacy_load_substack_power_ratings(season, week)


def _legacy_load_substack_qb_epa(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    LEGACY: Load Substack QB EPA data from hardcoded file path.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: QB-level EPA metrics
    """
    legacy_path = CURRENT_SEASON_DIR / f'qb_epa_substack_{season}_week_{week}.csv'
    # File has 2 header rows - skip first, use second as column names
    df = pd.read_csv(legacy_path, encoding='utf-8-sig', skiprows=1)

    # Remove weird header artifacts
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # The 'Tms' column contains team abbreviations (lowercase 3-letter codes)
    # Some QBs have multiple teams (e.g., "cle, cin") - take the first team
    if 'Tms' in df.columns:
        df['Tms'] = df['Tms'].str.split(',').str[0].str.strip()
        df = normalize_team_column(df, column_name='Tms', new_column_name='team')
    elif 'Team' in df.columns:
        df['Team'] = df['Team'].str.split(',').str[0].str.strip()
        df = normalize_team_column(df, column_name='Team', new_column_name='team')

    print(f"✓ Loaded Substack QB EPA: {len(df)} QBs")
    return df


def load_substack_qb_epa(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load Substack QB EPA data.

    Forwards to unified loader when available, otherwise uses legacy implementation.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: QB-level EPA metrics
    """
    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_qb_epa("substack", season, week)
    return _legacy_load_substack_qb_epa(season, week)


def _legacy_load_substack_weekly_projections(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    LEGACY: Load Substack weekly game projections from hardcoded file path.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Weekly matchups with projected spreads
    """
    legacy_path = CURRENT_SEASON_DIR / f'weekly_projections_ppg_substack_{season}_week_{week}.csv'
    # Try PPG file first
    df = pd.read_csv(legacy_path, encoding='utf-8-sig')

    # Remove weird header artifacts
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Parse matchups - handle both "at" and "vs" formats row-by-row
    # "Team1 at Team2" = Team1 is away, Team2 is home
    # "Team1 vs Team2" = neutral or unclear, assume first is away
    if 'Matchup' in df.columns:
        def parse_matchup(matchup):
            if ' at ' in matchup:
                teams = matchup.split(' at ')
                return pd.Series({'team_away_full': teams[0], 'team_home_full': teams[1]})
            elif ' vs ' in matchup:
                teams = matchup.split(' vs ')
                return pd.Series({'team_away_full': teams[0], 'team_home_full': teams[1]})
            else:
                return pd.Series({'team_away_full': None, 'team_home_full': None})

        df[['team_away_full', 'team_home_full']] = df['Matchup'].apply(parse_matchup)

        # Normalize team names
        df['team_away'] = df['team_away_full'].apply(normalize_team_name)
        df['team_home'] = df['team_home_full'].apply(normalize_team_name)

    # Parse favorite column (e.g., "ATL -5.5")
    if 'Favorite' in df.columns:
        df['substack_spread_line'] = df['Favorite'].str.extract(r'([-+]?\d+\.?\d*)')[0].astype(float)

    print(f"✓ Loaded Substack weekly projections: {len(df)} games")
    return df


def load_substack_weekly_projections(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load Substack weekly game projections (spreads and win probabilities).

    Forwards to unified loader when available, otherwise uses legacy implementation.

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Weekly matchups with projected spreads
    """
    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_weekly_projections_ppg("substack", season, week)
    return _legacy_load_substack_weekly_projections(season, week)


# ============================================================================
# REFERENCE DATA LOADERS
# ============================================================================

def load_head_coaches():
    """
    Load NFL head coach data.

    Returns:
        pd.DataFrame: Coach stats and tenure info
    """
    df = pd.read_csv(NFL_HEAD_COACHES)

    # The Coach column has coach names
    # This file may not have explicit team mapping, will need to cross-reference

    print(f"✓ Loaded head coach data: {len(df)} coaches")
    return df


# ============================================================================
# COMBINED LOADERS
# ============================================================================

def load_all_current_week_data(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Load all external ratings data for a given season and week.

    NOTE: This is a DEPRECATED compatibility function.
    New code should use: ball_knower.io.loaders.load_all_sources(season, week)

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        dict: Dictionary with all loaded DataFrames
    """
    data = {}

    print("\n" + "="*60)
    print(f"LOADING DATA (Season {season}, Week {week})")
    print("="*60 + "\n")

    # Load nfelo data
    data['nfelo_power'] = load_nfelo_power_ratings(season, week)
    data['nfelo_epa'] = load_nfelo_epa_tiers(season, week)
    data['nfelo_sos'] = load_nfelo_sos(season, week)

    # Load Substack data
    data['substack_power'] = load_substack_power_ratings(season, week)
    data['substack_qb_epa'] = load_substack_qb_epa(season, week)
    data['substack_weekly'] = load_substack_weekly_projections(season, week)

    # Load reference data
    data['coaches'] = load_head_coaches()

    print("\n" + "="*60)
    print(f"✓ ALL DATA LOADED FOR {season} WEEK {week}")
    print("="*60 + "\n")

    return data


def merge_current_week_ratings(season=DEFAULT_SEASON, week=DEFAULT_WEEK):
    """
    Merge all ratings for a given season and week into a single team-level DataFrame.

    NOTE: This is a DEPRECATED compatibility function.
    New code should use: ball_knower.io.loaders.load_all_sources(season, week)['merged_ratings']

    Args:
        season (int): Season year (default: 2025)
        week (int): Week number (default: 11)

    Returns:
        pd.DataFrame: Combined ratings with all features per team
    """
    data = load_all_current_week_data(season, week)

    # Start with nfelo power ratings as base
    merged = data['nfelo_power'][['team', 'nfelo', 'QB Adj', 'Value']].copy()

    # Merge nfelo EPA
    merged = merged.merge(
        data['nfelo_epa'][['team', 'epa_off', 'epa_def', 'epa_margin']],
        on='team',
        how='left'
    )

    # Merge Substack power ratings
    merged = merged.merge(
        data['substack_power'][['team', 'Off.', 'Def.', 'Ovr.']],
        on='team',
        how='left',
        suffixes=('', '_substack')
    )

    print(f"✓ Merged ratings for {season} Week {week}: {len(merged)} teams")
    return merged
