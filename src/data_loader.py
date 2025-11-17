"""
Data Loading Module

Handles loading and initial cleaning of all data sources:
- nfl_data_py historical data
- nfelo ratings and stats
- Substack ratings and projections
- Reference data (coaches, AV)

All data is normalized to use standard team abbreviations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from .team_mapping import normalize_team_name, normalize_team_column
from .config import (
    NFELO_POWER_RATINGS, NFELO_SOS, NFELO_EPA_TIERS,
    NFELO_QB_RANKINGS, SUBSTACK_POWER_RATINGS, SUBSTACK_QB_EPA,
    SUBSTACK_WEEKLY_PROJ_ELO, SUBSTACK_WEEKLY_PROJ_PPG,
    NFL_HEAD_COACHES, CURRENT_SEASON, CURRENT_WEEK
)

# Try to import new unified loaders
try:
    from ball_knower.io import loaders as new_loaders
    NEW_LOADERS_AVAILABLE = True
except ImportError:
    NEW_LOADERS_AVAILABLE = False
    warnings.warn(
        "ball_knower.io.loaders not available; using legacy data_loader implementations.",
        UserWarning,
    )

warnings.filterwarnings('ignore', category=FutureWarning)


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

def _legacy_load_nfelo_power_ratings():
    """Legacy implementation of load_nfelo_power_ratings."""
    df = pd.read_csv(NFELO_POWER_RATINGS)

    # Standardize team names
    df = normalize_team_column(df, column_name='Team', new_column_name='team')

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    print(f"✓ Loaded nfelo power ratings: {len(df)} teams")
    return df


def load_nfelo_power_ratings():
    """
    Load nfelo power ratings (current week).

    DEPRECATED: Use ball_knower.io.loaders.load_power_ratings('nfelo', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: Team power ratings with standardized team column
    """
    warnings.warn(
        "load_nfelo_power_ratings() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_power_ratings('nfelo', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_power_ratings(
            provider="nfelo",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_nfelo_power_ratings()


def _legacy_load_nfelo_epa_tiers():
    """Legacy implementation of load_nfelo_epa_tiers."""
    df = pd.read_csv(NFELO_EPA_TIERS)

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


def load_nfelo_epa_tiers():
    """
    Load nfelo EPA tiers (offensive/defensive EPA per play).

    DEPRECATED: Use ball_knower.io.loaders.load_epa_tiers('nfelo', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: EPA metrics by team
    """
    warnings.warn(
        "load_nfelo_epa_tiers() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_epa_tiers('nfelo', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_epa_tiers(
            provider="nfelo",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_nfelo_epa_tiers()


def _legacy_load_nfelo_qb_rankings():
    """Legacy implementation of load_nfelo_qb_rankings."""
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


def load_nfelo_qb_rankings():
    """
    Load nfelo QB rankings.

    DEPRECATED: Use ball_knower.io.loaders.load_qb_rankings('nfelo', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: QB rankings by team
    """
    warnings.warn(
        "load_nfelo_qb_rankings() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_qb_rankings('nfelo', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_qb_rankings(
            provider="nfelo",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_nfelo_qb_rankings()


def _legacy_load_nfelo_sos():
    """Legacy implementation of load_nfelo_sos."""
    df = pd.read_csv(NFELO_SOS)

    df = normalize_team_column(df, column_name='Team', new_column_name='team')

    print(f"✓ Loaded nfelo SOS: {len(df)} teams")
    return df


def load_nfelo_sos():
    """
    Load nfelo strength of schedule data.

    DEPRECATED: Use ball_knower.io.loaders.load_strength_of_schedule('nfelo', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: SOS metrics by team
    """
    warnings.warn(
        "load_nfelo_sos() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_strength_of_schedule('nfelo', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_strength_of_schedule(
            provider="nfelo",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_nfelo_sos()


# ============================================================================
# SUBSTACK LOADERS
# ============================================================================

def _legacy_load_substack_power_ratings():
    """Legacy implementation of load_substack_power_ratings."""
    # File has 2 header rows - skip first, use second as column names
    df = pd.read_csv(SUBSTACK_POWER_RATINGS, encoding='utf-8-sig', skiprows=1)

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


def load_substack_power_ratings():
    """
    Load Substack power ratings.

    DEPRECATED: Use ball_knower.io.loaders.load_power_ratings('substack', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: Power ratings with Off/Def/Ovr scores
    """
    warnings.warn(
        "load_substack_power_ratings() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_power_ratings('substack', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_power_ratings(
            provider="substack",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_substack_power_ratings()


def _legacy_load_substack_qb_epa():
    """Legacy implementation of load_substack_qb_epa."""
    # File has 2 header rows - skip first, use second as column names
    df = pd.read_csv(SUBSTACK_QB_EPA, encoding='utf-8-sig', skiprows=1)

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


def load_substack_qb_epa():
    """
    Load Substack QB EPA data.

    DEPRECATED: Use ball_knower.io.loaders.load_qb_epa('substack', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: QB-level EPA metrics
    """
    warnings.warn(
        "load_substack_qb_epa() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_qb_epa('substack', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_qb_epa(
            provider="substack",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_substack_qb_epa()


def _legacy_load_substack_weekly_projections():
    """Legacy implementation of load_substack_weekly_projections."""
    # Try PPG file first
    df = pd.read_csv(SUBSTACK_WEEKLY_PROJ_PPG, encoding='utf-8-sig')

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


def load_substack_weekly_projections():
    """
    Load Substack weekly game projections (spreads and win probabilities).

    DEPRECATED: Use ball_knower.io.loaders.load_weekly_projections_ppg('substack', season, week) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: Weekly matchups with projected spreads
    """
    warnings.warn(
        "load_substack_weekly_projections() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_weekly_projections_ppg('substack', season, week) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        return new_loaders.load_weekly_projections_ppg(
            provider="substack",
            season=CURRENT_SEASON,
            week=CURRENT_WEEK,
        )

    return _legacy_load_substack_weekly_projections()


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

def _legacy_load_all_current_week_data():
    """Legacy implementation of load_all_current_week_data."""
    data = {}

    print("\n" + "="*60)
    print("LOADING CURRENT WEEK DATA (Week 11, 2025)")
    print("="*60 + "\n")

    # Load nfelo data
    data['nfelo_power'] = _legacy_load_nfelo_power_ratings()
    data['nfelo_epa'] = _legacy_load_nfelo_epa_tiers()
    data['nfelo_sos'] = _legacy_load_nfelo_sos()

    # Load Substack data
    data['substack_power'] = _legacy_load_substack_power_ratings()
    data['substack_qb_epa'] = _legacy_load_substack_qb_epa()
    data['substack_weekly'] = _legacy_load_substack_weekly_projections()

    # Load reference data
    data['coaches'] = load_head_coaches()

    print("\n" + "="*60)
    print("✓ ALL CURRENT WEEK DATA LOADED")
    print("="*60 + "\n")

    return data


def load_all_current_week_data():
    """
    Load all current week external ratings data.

    DEPRECATED: Use ball_knower.io.loaders.load_all_sources(week, season) instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        dict: Dictionary with all loaded DataFrames (keys differ from new loaders)
    """
    warnings.warn(
        "load_all_current_week_data() is deprecated and will be removed in a future version. "
        "Use ball_knower.io.loaders.load_all_sources(week, season) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        # Load data using new loaders
        new_data = new_loaders.load_all_sources(
            week=CURRENT_WEEK,
            season=CURRENT_SEASON,
        )

        # Map new keys to legacy keys for backward compatibility
        legacy_data = {
            'nfelo_power': new_data.get('power_ratings_nfelo'),
            'nfelo_epa': new_data.get('epa_tiers_nfelo'),
            'nfelo_sos': new_data.get('strength_of_schedule_nfelo'),
            'substack_power': new_data.get('power_ratings_substack'),
            'substack_qb_epa': new_data.get('qb_epa_substack'),
            'substack_weekly': new_data.get('weekly_projections_ppg_substack'),
            'coaches': load_head_coaches(),  # Still load coaches with legacy function
        }

        return legacy_data

    return _legacy_load_all_current_week_data()


def _legacy_merge_current_week_ratings():
    """Legacy implementation of merge_current_week_ratings."""
    data = _legacy_load_all_current_week_data()

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

    print(f"✓ Merged current week ratings: {len(merged)} teams")
    return merged


def merge_current_week_ratings():
    """
    Merge all current week ratings into a single team-level DataFrame.

    DEPRECATED: Use ball_knower.io.loaders.load_all_sources(...)['merged_ratings'] instead.
    This compatibility wrapper will be removed in a future version.

    Returns:
        pd.DataFrame: Combined ratings with all features per team
    """
    warnings.warn(
        "merge_current_week_ratings() is deprecated and will be removed in a future version. "
        'Use ball_knower.io.loaders.load_all_sources(...)["merged_ratings"] instead.',
        DeprecationWarning,
        stacklevel=2,
    )

    if NEW_LOADERS_AVAILABLE:
        data = new_loaders.load_all_sources(
            week=CURRENT_WEEK,
            season=CURRENT_SEASON,
        )
        return data["merged_ratings"]

    return _legacy_merge_current_week_ratings()
