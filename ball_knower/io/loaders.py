"""
Ball Knower Unified Data Loader Module

This module provides a unified, category-first API for loading NFL data from multiple providers.

Key Features:
- Category-first naming convention (e.g., 'power_ratings', 'epa_tiers')
- Dual-pattern file resolution (category-first + provider-first fallback)
- Automatic team name normalization
- Comprehensive error handling and warnings
- Orchestrator functions for loading all sources

Usage:
    from ball_knower.io import loaders

    # Load individual datasets
    df = loaders.load_power_ratings('nfelo', 2024, 11, './data')

    # Load all sources at once
    data = loaders.load_all_sources(week=11, season=2024, data_dir='./data')

    # Get merged team ratings
    merged = loaders.merge_team_ratings(data)

Author: Ball Knower Team
Date: 2024-11-17
Version: 1.0.0
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# Supported data categories
CATEGORIES = {
    'power_ratings',
    'epa_tiers',
    'strength_of_schedule',
    'qb_epa',
    'weekly_projections_ppg',
    'rest_days',
    'team_stats',
    'injuries',
    'vegas_lines',
}

# Supported providers
PROVIDERS = {
    'nfelo',
    'substack',
    'nflverse',
}

# Default data directory
DEFAULT_DATA_DIR = './data'


# ============================================================================
# TEAM NAME NORMALIZATION
# ============================================================================

def normalize_team_name(name: str) -> str:
    """
    Normalize team names to standard abbreviations.

    Args:
        name: Team name or abbreviation

    Returns:
        Standardized team abbreviation
    """
    # Lazy import to avoid circular dependency
    try:
        from src.team_mapping import normalize_team_name as _normalize
        return _normalize(name)
    except ImportError:
        warnings.warn("Could not import team_mapping module; using as-is")
        return name


# ============================================================================
# FILE RESOLUTION
# ============================================================================

def resolve_data_file(
    category: str,
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[Path]:
    """
    Resolve the file path for a given data category and provider.

    Implements dual-pattern resolution:
    1. Category-first: {category}_{provider}_{season}_week_{week}.csv
    2. Provider-first (legacy): {provider}_{category}_{season}_week_{week}.csv

    Args:
        category: Data category (e.g., 'power_ratings')
        provider: Data provider (e.g., 'nfelo')
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        Path to the file if found, None otherwise
    """
    data_path = Path(data_dir)

    # Pattern 1: Category-first (preferred)
    category_first = data_path / f"{category}_{provider}_{season}_week_{week}.csv"
    if category_first.exists():
        return category_first

    # Pattern 2: Provider-first (legacy, deprecated)
    provider_first = data_path / f"{provider}_{category}_{season}_week_{week}.csv"
    if provider_first.exists():
        warnings.warn(
            f"Using legacy filename pattern: {provider_first.name}. "
            f"Please rename to category-first: {category_first.name}",
            DeprecationWarning,
            stacklevel=3
        )
        return provider_first

    return None


# ============================================================================
# CATEGORY-SPECIFIC LOADERS
# ============================================================================

def load_power_ratings(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load power ratings data.

    Args:
        provider: Data provider ('nfelo', 'substack')
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with power ratings, or None if not found
    """
    file_path = resolve_data_file('power_ratings', provider, season, week, data_dir)

    if file_path is None:
        warnings.warn(f"Power ratings not found for {provider} season {season} week {week}")
        return None

    df = pd.read_csv(file_path)

    # Normalize team names if 'team' column exists
    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_epa_tiers(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load EPA tier rankings.

    Args:
        provider: Data provider ('nfelo')
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with EPA tiers, or None if not found
    """
    file_path = resolve_data_file('epa_tiers', provider, season, week, data_dir)

    if file_path is None:
        warnings.warn(f"EPA tiers not found for {provider} season {season} week {week}")
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_strength_of_schedule(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load strength of schedule data.

    Args:
        provider: Data provider ('nfelo')
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with SOS data, or None if not found
    """
    file_path = resolve_data_file('strength_of_schedule', provider, season, week, data_dir)

    if file_path is None:
        warnings.warn(f"Strength of schedule not found for {provider} season {season} week {week}")
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_qb_epa(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load quarterback EPA data.

    Args:
        provider: Data provider ('substack')
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with QB EPA, or None if not found
    """
    file_path = resolve_data_file('qb_epa', provider, season, week, data_dir)

    if file_path is None:
        warnings.warn(f"QB EPA not found for {provider} season {season} week {week}")
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_weekly_projections_ppg(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load weekly points-per-game projections.

    Args:
        provider: Data provider ('substack')
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with PPG projections, or None if not found
    """
    file_path = resolve_data_file('weekly_projections_ppg', provider, season, week, data_dir)

    if file_path is None:
        warnings.warn(f"Weekly projections not found for {provider} season {season} week {week}")
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_rest_days(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load rest days data.

    Args:
        provider: Data provider
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with rest days, or None if not found
    """
    file_path = resolve_data_file('rest_days', provider, season, week, data_dir)

    if file_path is None:
        # Rest days is optional, don't warn
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_team_stats(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load team statistics.

    Args:
        provider: Data provider
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with team stats, or None if not found
    """
    file_path = resolve_data_file('team_stats', provider, season, week, data_dir)

    if file_path is None:
        # Team stats is optional
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_injuries(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load injury data.

    Args:
        provider: Data provider
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with injuries, or None if not found
    """
    file_path = resolve_data_file('injuries', provider, season, week, data_dir)

    if file_path is None:
        # Injuries is optional
        return None

    df = pd.read_csv(file_path)

    if 'team' in df.columns:
        df['team'] = df['team'].apply(normalize_team_name)

    return df


def load_vegas_lines(
    provider: str,
    season: int,
    week: int,
    data_dir: str = DEFAULT_DATA_DIR
) -> Optional[pd.DataFrame]:
    """
    Load Vegas betting lines.

    Args:
        provider: Data provider
        season: NFL season year
        week: NFL week number
        data_dir: Directory containing data files

    Returns:
        DataFrame with Vegas lines, or None if not found
    """
    file_path = resolve_data_file('vegas_lines', provider, season, week, data_dir)

    if file_path is None:
        warnings.warn(f"Vegas lines not found for {provider} season {season} week {week}")
        return None

    df = pd.read_csv(file_path)

    # Normalize both team columns if they exist
    for col in ['team', 'home_team', 'away_team']:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)

    return df


# ============================================================================
# ORCHESTRATOR FUNCTIONS
# ============================================================================

def load_all_sources(
    week: int,
    season: int = 2024,
    data_dir: str = DEFAULT_DATA_DIR
) -> Dict[str, Any]:
    """
    Load all available data sources for a given week.

    This orchestrator function loads:
    - NFelo: power_ratings, epa_tiers, strength_of_schedule
    - Substack: power_ratings, qb_epa, weekly_projections_ppg
    - Merged ratings (via merge_team_ratings)

    Args:
        week: NFL week number
        season: NFL season year (default: 2024)
        data_dir: Directory containing data files

    Returns:
        Dictionary containing all loaded data:
        {
            'power_ratings_nfelo': DataFrame,
            'epa_tiers_nfelo': DataFrame,
            'strength_of_schedule_nfelo': DataFrame,
            'power_ratings_substack': DataFrame,
            'qb_epa_substack': DataFrame,
            'weekly_projections_ppg_substack': DataFrame,
            'merged_ratings': DataFrame,
        }
    """
    data = {}

    # NFelo sources
    data['power_ratings_nfelo'] = load_power_ratings('nfelo', season, week, data_dir)
    data['epa_tiers_nfelo'] = load_epa_tiers('nfelo', season, week, data_dir)
    data['strength_of_schedule_nfelo'] = load_strength_of_schedule('nfelo', season, week, data_dir)

    # Substack sources
    data['power_ratings_substack'] = load_power_ratings('substack', season, week, data_dir)
    data['qb_epa_substack'] = load_qb_epa('substack', season, week, data_dir)
    data['weekly_projections_ppg_substack'] = load_weekly_projections_ppg('substack', season, week, data_dir)

    # Merge team ratings
    data['merged_ratings'] = merge_team_ratings(data)

    return data


def merge_team_ratings(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Merge team ratings from multiple sources into a unified DataFrame.

    Combines:
    - NFelo power ratings
    - NFelo EPA tiers
    - NFelo strength of schedule
    - Substack power ratings
    - Substack QB EPA
    - Substack weekly projections

    Args:
        data: Dictionary of loaded data from load_all_sources()

    Returns:
        DataFrame with 32 teams × merged features
    """
    # Start with NFelo power ratings as base
    merged = data.get('power_ratings_nfelo')

    if merged is None:
        warnings.warn("NFelo power ratings not available; cannot merge team ratings")
        return pd.DataFrame()

    merged = merged.copy()

    # Merge NFelo EPA tiers
    if data.get('epa_tiers_nfelo') is not None:
        merged = merged.merge(
            data['epa_tiers_nfelo'],
            on='team',
            how='left',
            suffixes=('', '_epa')
        )

    # Merge NFelo SOS
    if data.get('strength_of_schedule_nfelo') is not None:
        merged = merged.merge(
            data['strength_of_schedule_nfelo'],
            on='team',
            how='left',
            suffixes=('', '_sos')
        )

    # Merge Substack power ratings
    if data.get('power_ratings_substack') is not None:
        merged = merged.merge(
            data['power_ratings_substack'],
            on='team',
            how='left',
            suffixes=('_nfelo', '_substack')
        )

    # Merge Substack QB EPA
    if data.get('qb_epa_substack') is not None:
        merged = merged.merge(
            data['qb_epa_substack'],
            on='team',
            how='left',
            suffixes=('', '_qb')
        )

    # Merge Substack weekly projections
    if data.get('weekly_projections_ppg_substack') is not None:
        merged = merged.merge(
            data['weekly_projections_ppg_substack'],
            on='team',
            how='left',
            suffixes=('', '_proj')
        )

    return merged


# ============================================================================
# SANITY CHECK (for direct module execution)
# ============================================================================

if __name__ == '__main__':
    """
    Sanity check: Load all sources for Week 11, 2024.
    """
    print("=" * 60)
    print("BALL KNOWER UNIFIED LOADER - SANITY CHECK")
    print("=" * 60)

    week = 11
    season = 2024
    data_dir = './data'

    print(f"\nLoading all sources for Week {week}, {season}...")
    print("-" * 60)

    data = load_all_sources(week, season, data_dir)

    print("\nLoaded datasets:")
    for key, df in data.items():
        if df is not None and not df.empty:
            print(f"  ✓ {key}: {df.shape[0]} rows × {df.shape[1]} columns")
        else:
            print(f"  ✗ {key}: Not available")

    print("\n" + "=" * 60)
    print("✓ SANITY CHECK COMPLETE")
    print("=" * 60)

    # Show merged ratings summary
    merged = data.get('merged_ratings')
    if merged is not None and not merged.empty:
        print(f"\n✓ Merged ratings: {merged.shape[0]} teams × {merged.shape[1]} features")
        print(f"✓ Columns: {', '.join(merged.columns[:10])}...")
    else:
        print("\n✗ Merged ratings not available")
