"""
Unified Data Loaders for Ball Knower

This module provides a unified interface for loading NFL data from multiple sources.
It supports both the new category-first naming convention and legacy provider-first
filenames via automatic fallback.

Target naming convention (category-first):
    {category}_{provider}_{season}_week_{week}.csv

Legacy naming convention (provider-first):
    {provider}_{category_variant}_{season}_week_{week}.csv

The loaders automatically try the new pattern first, then fall back to known legacy
patterns for the current dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import warnings
import sys
import importlib.util

# Import config and team_mapping directly to avoid circular imports
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

# Import config module directly
_config_spec = importlib.util.spec_from_file_location("config", _project_root / "src" / "config.py")
_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config)

# Import team_mapping module directly
_team_mapping_spec = importlib.util.spec_from_file_location("team_mapping", _project_root / "src" / "team_mapping.py")
_team_mapping = importlib.util.module_from_spec(_team_mapping_spec)
_team_mapping_spec.loader.exec_module(_team_mapping)

# Extract needed values
CURRENT_SEASON = _config.CURRENT_SEASON
CURRENT_WEEK = _config.CURRENT_WEEK
CURRENT_SEASON_DIR = _config.CURRENT_SEASON_DIR
normalize_team_name = _team_mapping.normalize_team_name
normalize_team_column = _team_mapping.normalize_team_column

# Default data directory
DEFAULT_DATA_DIR = CURRENT_SEASON_DIR

# Fallback filename patterns for current dataset (provider-first legacy naming)
# Maps (category, provider) -> legacy filename pattern
FALLBACK_FILENAMES = {
    # nfelo files (provider-first)
    ("power_ratings", "nfelo"): "nfelo_power_ratings_{season}_week_{week}.csv",
    ("epa_tiers", "nfelo"): "nfelo_epa_tiers_off_def_{season}_week_{week}.csv",
    ("strength_of_schedule", "nfelo"): "nfelo_strength_of_schedule_{season}_week_{week}.csv",
    ("qb_rankings", "nfelo"): "nfelo_qb_rankings_{season}_week_{week}.csv",
    ("win_totals", "nfelo"): "nfelo_nfl_win_totals_{season}_week_{week} (1).csv",
    ("receiving_leaders", "nfelo"): "nfelo_nfl_receiving_leaders_{season}_week_{week}.csv",

    # substack files (provider-first)
    ("power_ratings", "substack"): "substack_power_ratings_{season}_week_{week}.csv",
    ("qb_epa", "substack"): "substack_qb_epa_{season}_week_{week}.csv",
    ("weekly_projections_elo", "substack"): "substack_weekly_proj_elo_{season}_week_{week}.csv",
    ("weekly_projections_ppg", "substack"): "substack_weekly_proj_ppg_{season}_week_{week}.csv",
}


def _resolve_file_path(
    category: str,
    provider: str,
    season: int,
    week: int,
    data_dir: Path
) -> Path:
    """
    Resolve the file path for a given category and provider.

    Tries category-first naming first, then falls back to legacy provider-first naming.
    Issues a deprecation warning when using legacy filenames.

    Args:
        category: Data category (e.g., 'power_ratings', 'epa_tiers')
        provider: Data provider (e.g., 'nfelo', 'substack')
        season: Season year
        week: Week number
        data_dir: Data directory path

    Returns:
        Path: Resolved file path

    Raises:
        FileNotFoundError: If neither category-first nor legacy pattern exists
    """
    # Try category-first naming first (target pattern)
    category_first_filename = f"{category}_{provider}_{season}_week_{week}.csv"
    category_first_path = data_dir / category_first_filename

    if category_first_path.exists():
        return category_first_path

    # Fall back to legacy provider-first naming
    legacy_key = (category, provider)
    if legacy_key in FALLBACK_FILENAMES:
        legacy_pattern = FALLBACK_FILENAMES[legacy_key]
        legacy_filename = legacy_pattern.format(season=season, week=week)
        legacy_path = data_dir / legacy_filename

        if legacy_path.exists():
            warnings.warn(
                f"Using legacy filename '{legacy_filename}'. "
                f"Consider renaming to '{category_first_filename}' for consistency.",
                DeprecationWarning,
                stacklevel=3,
            )
            return legacy_path

    # Neither pattern exists
    raise FileNotFoundError(
        f"Could not find data file for {category}/{provider}. "
        f"Tried:\n  - {category_first_filename}\n"
        f"  - {FALLBACK_FILENAMES.get(legacy_key, 'no legacy pattern')}"
    )


def _normalize_team_column_inplace(df: pd.DataFrame, column_name: str = 'Team') -> pd.DataFrame:
    """
    Normalize team names in a DataFrame using src.team_mapping.

    Args:
        df: DataFrame with team names
        column_name: Name of the column containing team names

    Returns:
        DataFrame with normalized 'team' column
    """
    return normalize_team_column(df, column_name=column_name, new_column_name='team')


# ============================================================================
# INDIVIDUAL CATEGORY LOADERS
# ============================================================================

def load_power_ratings(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load power ratings for a given provider.

    Args:
        provider: Data provider ('nfelo' or 'substack')
        season: Season year (default: CURRENT_SEASON from config)
        week: Week number (default: CURRENT_WEEK from config)
        data_dir: Data directory (default: CURRENT_SEASON_DIR from config)

    Returns:
        pd.DataFrame with columns including 'team' and provider-specific ratings
    """
    file_path = _resolve_file_path("power_ratings", provider, season, week, data_dir)

    if provider == "nfelo":
        df = pd.read_csv(file_path)
        df = _normalize_team_column_inplace(df, column_name='Team')
        df.columns = [col.strip() for col in df.columns]
        return df

    elif provider == "substack":
        # Substack has 2 header rows
        df = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=1)
        df = df.loc[:, ~df.columns.str.startswith('X.')]

        if 'Team' in df.columns:
            df = _normalize_team_column_inplace(df, column_name='Team')
        else:
            first_col = df.columns[0]
            df = _normalize_team_column_inplace(df, column_name=first_col)

        # Convert numeric columns
        for col in ['Off.', 'Def.', 'Ovr.']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_epa_tiers(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load EPA tiers (offensive/defensive EPA per play).

    Args:
        provider: Data provider ('nfelo')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with columns: team, epa_off, epa_def, epa_margin
    """
    file_path = _resolve_file_path("epa_tiers", provider, season, week, data_dir)
    df = pd.read_csv(file_path)

    df = _normalize_team_column_inplace(df, column_name='Team')

    # Rename EPA columns for clarity
    df.rename(columns={
        'EPA/Play': 'epa_off',
        'EPA/Play Against': 'epa_def'
    }, inplace=True)

    # Calculate EPA margin
    df['epa_margin'] = df['epa_off'] - df['epa_def']

    return df


def load_strength_of_schedule(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load strength of schedule data.

    Args:
        provider: Data provider ('nfelo')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with SOS metrics
    """
    file_path = _resolve_file_path("strength_of_schedule", provider, season, week, data_dir)
    df = pd.read_csv(file_path)
    df = _normalize_team_column_inplace(df, column_name='Team')
    return df


def load_qb_rankings(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load QB rankings.

    Args:
        provider: Data provider ('nfelo')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with QB rankings
    """
    file_path = _resolve_file_path("qb_rankings", provider, season, week, data_dir)
    df = pd.read_csv(file_path)

    # Check if Team column exists
    if 'Team' in df.columns:
        df = _normalize_team_column_inplace(df, column_name='Team')

    return df


def load_qb_epa(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load QB EPA data.

    Args:
        provider: Data provider ('substack')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with QB-level EPA metrics
    """
    file_path = _resolve_file_path("qb_epa", provider, season, week, data_dir)

    # Substack has 2 header rows
    df = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=1)
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # The 'Tms' column contains team abbreviations
    if 'Tms' in df.columns:
        df['Tms'] = df['Tms'].str.split(',').str[0].str.strip()
        df = _normalize_team_column_inplace(df, column_name='Tms')
    elif 'Team' in df.columns:
        df['Team'] = df['Team'].str.split(',').str[0].str.strip()
        df = _normalize_team_column_inplace(df, column_name='Team')

    return df


def load_weekly_projections_ppg(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load weekly game projections (PPG-based).

    Args:
        provider: Data provider ('substack')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with weekly matchups and projected spreads
    """
    file_path = _resolve_file_path("weekly_projections_ppg", provider, season, week, data_dir)
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Parse matchups
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
        df['team_away'] = df['team_away_full'].apply(normalize_team_name)
        df['team_home'] = df['team_home_full'].apply(normalize_team_name)

    # Parse favorite column
    if 'Favorite' in df.columns:
        df['substack_spread_line'] = df['Favorite'].str.extract(r'([-+]?\d+\.?\d*)')[0].astype(float)

    return df


def load_weekly_projections_elo(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load weekly game projections (ELO-based).

    Args:
        provider: Data provider ('substack')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with ELO-based weekly projections
    """
    file_path = _resolve_file_path("weekly_projections_elo", provider, season, week, data_dir)
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df = df.loc[:, ~df.columns.str.startswith('X.')]
    return df


def load_win_totals(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load win totals data.

    Args:
        provider: Data provider ('nfelo')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with win totals
    """
    file_path = _resolve_file_path("win_totals", provider, season, week, data_dir)
    df = pd.read_csv(file_path)
    if 'Team' in df.columns:
        df = _normalize_team_column_inplace(df, column_name='Team')
    return df


def load_receiving_leaders(
    provider: str,
    season: int = CURRENT_SEASON,
    week: int = CURRENT_WEEK,
    data_dir: Path = DEFAULT_DATA_DIR
) -> pd.DataFrame:
    """
    Load receiving leaders data.

    Args:
        provider: Data provider ('nfelo')
        season: Season year
        week: Week number
        data_dir: Data directory

    Returns:
        pd.DataFrame with receiving leaders
    """
    file_path = _resolve_file_path("receiving_leaders", provider, season, week, data_dir)
    df = pd.read_csv(file_path)
    if 'Team' in df.columns:
        df = _normalize_team_column_inplace(df, column_name='Team')
    return df


# ============================================================================
# ORCHESTRATOR FUNCTIONS
# ============================================================================

def load_all_sources(
    week: int = CURRENT_WEEK,
    season: int = CURRENT_SEASON,
    data_dir: Path = DEFAULT_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Load all available data sources for a given week/season.

    Returns a dictionary with keys matching the Phase 2 compatibility layer expectations.

    Args:
        week: Week number (default: CURRENT_WEEK from config)
        season: Season year (default: CURRENT_SEASON from config)
        data_dir: Data directory (default: CURRENT_SEASON_DIR from config)

    Returns:
        dict with keys:
            - 'power_ratings_nfelo'
            - 'epa_tiers_nfelo'
            - 'strength_of_schedule_nfelo'
            - 'power_ratings_substack'
            - 'qb_epa_substack'
            - 'weekly_projections_ppg_substack'
            - 'merged_ratings'
    """
    data = {}

    # Load nfelo data
    try:
        data['power_ratings_nfelo'] = load_power_ratings("nfelo", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load nfelo power ratings: {e}")

    try:
        data['epa_tiers_nfelo'] = load_epa_tiers("nfelo", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load nfelo EPA tiers: {e}")

    try:
        data['strength_of_schedule_nfelo'] = load_strength_of_schedule("nfelo", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load nfelo SOS: {e}")

    # Load Substack data
    try:
        data['power_ratings_substack'] = load_power_ratings("substack", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load Substack power ratings: {e}")

    try:
        data['qb_epa_substack'] = load_qb_epa("substack", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load Substack QB EPA: {e}")

    try:
        data['weekly_projections_ppg_substack'] = load_weekly_projections_ppg("substack", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load Substack weekly projections: {e}")

    # Create merged ratings (matches legacy merge_current_week_ratings logic)
    data['merged_ratings'] = merge_team_ratings(data)

    return data


def merge_team_ratings(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all team-level ratings into a single DataFrame.

    This matches the logic from src.data_loader.merge_current_week_ratings()
    to ensure backward compatibility.

    Args:
        data_dict: Dictionary of DataFrames from load_all_sources()

    Returns:
        pd.DataFrame with merged team ratings
    """
    # Start with nfelo power ratings as base (if available)
    if 'power_ratings_nfelo' in data_dict:
        nfelo_power = data_dict['power_ratings_nfelo']
        merged = nfelo_power[['team', 'nfelo', 'QB Adj', 'Value']].copy()
    else:
        # Fallback: create empty DataFrame with team column
        merged = pd.DataFrame(columns=['team'])

    # Merge nfelo EPA (if available)
    if 'epa_tiers_nfelo' in data_dict:
        nfelo_epa = data_dict['epa_tiers_nfelo']
        merged = merged.merge(
            nfelo_epa[['team', 'epa_off', 'epa_def', 'epa_margin']],
            on='team',
            how='outer'
        )

    # Merge Substack power ratings (if available)
    if 'power_ratings_substack' in data_dict:
        substack_power = data_dict['power_ratings_substack']
        merged = merged.merge(
            substack_power[['team', 'Off.', 'Def.', 'Ovr.']],
            on='team',
            how='outer',
            suffixes=('', '_substack')
        )

    return merged


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    """Sanity check: Load all data and display summary."""
    print("="*60)
    print("BALL KNOWER UNIFIED LOADERS - SANITY CHECK")
    print("="*60)
    print(f"\nSeason: {CURRENT_SEASON}, Week: {CURRENT_WEEK}")
    print(f"Data directory: {DEFAULT_DATA_DIR}\n")

    # Load all sources
    data = load_all_sources()

    print("\nLoaded datasets:")
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  ✓ {key}: {len(df)} rows, {len(df.columns)} columns")

    # Display merged ratings
    if 'merged_ratings' in data:
        print("\nMerged ratings (first 10 teams):")
        print(data['merged_ratings'].head(10))

    print("\n" + "="*60)
    print("✓ SANITY CHECK COMPLETE")
    print("="*60)
