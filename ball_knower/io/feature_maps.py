"""
Feature Mapping Layer

Provides provider-agnostic canonical feature names for Ball Knower models.

This module abstracts away provider-specific column names (e.g., 'nfelo', 'Ovr.')
and exposes a unified API using semantic feature names (e.g., 'overall_rating').

Usage:
    from ball_knower.io import loaders, feature_maps

    # Load data using unified loader
    data = loaders.load_all_sources(season=2025, week=11)

    # Get canonical feature view
    canonical_ratings = feature_maps.get_canonical_features(data['merged_ratings'])

    # Now use semantic names:
    canonical_ratings[['team', 'overall_rating', 'qb_adjustment', 'epa_margin']]

Adding new providers (v1.2/v1.3):
    1. Add provider mappings to CANONICAL_FEATURE_MAP
    2. Update PROVIDER_PRIORITY if needed
    3. No code changes required in models or scripts
"""

import pandas as pd
import warnings
from typing import Optional, List, Dict, Any

# ============================================================================
# CANONICAL FEATURE SCHEMA
# ============================================================================

CANONICAL_FEATURE_MAP = {
    # Overall power/quality ratings
    'overall_rating': {
        'description': 'Overall team power rating/quality score',
        'nfelo': 'nfelo',
        'substack': 'Ovr.',
        'preferred_provider': 'nfelo',  # Use nfelo when available
    },

    # QB-specific adjustments
    'qb_adjustment': {
        'description': 'QB quality adjustment to base rating',
        'nfelo': 'QB Adj',
        'substack': None,  # Not available
        'preferred_provider': 'nfelo',
    },

    # Offensive ratings
    'offensive_rating': {
        'description': 'Offensive strength rating',
        'nfelo': None,  # Not directly available (use epa_offense instead)
        'substack': 'Off.',
        'preferred_provider': 'substack',
    },

    # Defensive ratings
    'defensive_rating': {
        'description': 'Defensive strength rating',
        'nfelo': None,  # Not directly available (use epa_defense instead)
        'substack': 'Def.',
        'preferred_provider': 'substack',
    },

    # EPA metrics (Expected Points Added)
    'epa_offense': {
        'description': 'Offensive EPA per play',
        'nfelo': 'epa_off',
        'substack': None,
        'preferred_provider': 'nfelo',
    },

    'epa_defense': {
        'description': 'Defensive EPA per play (opponent EPA against)',
        'nfelo': 'epa_def',
        'substack': None,
        'preferred_provider': 'nfelo',
    },

    'epa_margin': {
        'description': 'EPA differential (offense - defense)',
        'nfelo': 'epa_margin',
        'substack': None,
        'preferred_provider': 'nfelo',
    },

    # Trend metrics
    'value_rating': {
        'description': 'Value/efficiency rating',
        'nfelo': 'Value',
        'substack': None,
        'preferred_provider': 'nfelo',
    },

    'week_over_week_change': {
        'description': 'Rating change from previous week',
        'nfelo': 'WoW',
        'substack': None,
        'preferred_provider': 'nfelo',
    },

    'year_to_date_performance': {
        'description': 'Season-to-date performance metric',
        'nfelo': 'YTD',
        'substack': None,
        'preferred_provider': 'nfelo',
    },
}

# Provider priority when multiple sources have the same feature
# Higher priority = preferred source
PROVIDER_PRIORITY = ['nfelo', 'substack']


# ============================================================================
# FEATURE EXTRACTION UTILITIES
# ============================================================================

def get_canonical_features(
    merged_ratings: pd.DataFrame,
    features: Optional[List[str]] = None,
    include_team: bool = True,
    strict: bool = False
) -> pd.DataFrame:
    """
    Extract canonical features from merged ratings DataFrame.

    Converts provider-specific column names to semantic canonical names.

    Args:
        merged_ratings: DataFrame from loaders.load_all_sources()['merged_ratings']
        features: List of canonical feature names to extract (default: all available)
        include_team: Include 'team' column in output (default: True)
        strict: Raise error if requested feature not available (default: False, warns instead)

    Returns:
        DataFrame with canonical feature names as columns

    Example:
        >>> data = loaders.load_all_sources(season=2025, week=11)
        >>> canonical = get_canonical_features(data['merged_ratings'])
        >>> canonical[['team', 'overall_rating', 'epa_margin']].head()
    """
    result = pd.DataFrame()

    # Always include team column if requested
    if include_team:
        if 'team' not in merged_ratings.columns:
            raise ValueError("Expected 'team' column in merged_ratings DataFrame")
        result['team'] = merged_ratings['team']

    # Determine which features to extract
    if features is None:
        # Extract all available features
        features_to_extract = list(CANONICAL_FEATURE_MAP.keys())
    else:
        # Extract only requested features
        features_to_extract = features

    # Extract each canonical feature
    for canonical_name in features_to_extract:
        if canonical_name not in CANONICAL_FEATURE_MAP:
            msg = f"Unknown canonical feature: '{canonical_name}'"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning)
            continue

        feature_config = CANONICAL_FEATURE_MAP[canonical_name]

        # Try to find the feature from available providers (in priority order)
        found = False
        for provider in PROVIDER_PRIORITY:
            provider_col = feature_config.get(provider)

            if provider_col and provider_col in merged_ratings.columns:
                result[canonical_name] = merged_ratings[provider_col]
                found = True
                break

        if not found:
            msg = f"Canonical feature '{canonical_name}' not available from any provider"
            if strict:
                raise ValueError(msg)
            # Create null column as placeholder
            result[canonical_name] = None

    return result


def get_feature_differential(
    canonical_ratings: pd.DataFrame,
    home_teams: pd.Series,
    away_teams: pd.Series,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate feature differentials (home - away) for matchups.

    Useful for model input where you need rating differentials.

    Args:
        canonical_ratings: DataFrame from get_canonical_features()
        home_teams: Series of home team abbreviations
        away_teams: Series of away team abbreviations
        features: List of canonical features to compute differentials for
                 (default: all numeric features)

    Returns:
        DataFrame with columns like 'overall_rating_diff', 'epa_margin_diff', etc.

    Example:
        >>> canonical = get_canonical_features(merged_ratings)
        >>> diffs = get_feature_differential(
        ...     canonical,
        ...     games['home_team'],
        ...     games['away_team'],
        ...     features=['overall_rating', 'epa_margin']
        ... )
    """
    if 'team' not in canonical_ratings.columns:
        raise ValueError("canonical_ratings must have 'team' column")

    # Determine which features to use
    if features is None:
        # Use all numeric columns except 'team'
        features = [col for col in canonical_ratings.columns
                   if col != 'team' and pd.api.types.is_numeric_dtype(canonical_ratings[col])]

    # Build matchups dataframe
    matchups = pd.DataFrame({
        'home_team': home_teams,
        'away_team': away_teams
    }).reset_index(drop=True)

    # Merge home team features
    for feature in features:
        if feature not in canonical_ratings.columns:
            warnings.warn(f"Feature '{feature}' not in canonical_ratings, skipping", UserWarning)
            continue

        # Merge home team
        matchups = matchups.merge(
            canonical_ratings[['team', feature]],
            left_on='home_team',
            right_on='team',
            how='left'
        ).drop(columns=['team']).rename(columns={feature: f'{feature}_home'})

        # Merge away team
        matchups = matchups.merge(
            canonical_ratings[['team', feature]],
            left_on='away_team',
            right_on='team',
            how='left'
        ).drop(columns=['team']).rename(columns={feature: f'{feature}_away'})

        # Calculate differential
        matchups[f'{feature}_diff'] = matchups[f'{feature}_home'] - matchups[f'{feature}_away']

    return matchups


def get_available_features(merged_ratings: pd.DataFrame) -> Dict[str, Any]:
    """
    Report which canonical features are available from the merged ratings.

    Args:
        merged_ratings: DataFrame from loaders.load_all_sources()['merged_ratings']

    Returns:
        Dict mapping canonical feature names to:
            - 'available': bool
            - 'provider': str (which provider supplies this feature)
            - 'column': str (actual column name in merged_ratings)

    Example:
        >>> data = loaders.load_all_sources(season=2025, week=11)
        >>> available = get_available_features(data['merged_ratings'])
        >>> print(available['overall_rating'])
        {'available': True, 'provider': 'nfelo', 'column': 'nfelo'}
    """
    availability = {}

    for canonical_name, feature_config in CANONICAL_FEATURE_MAP.items():
        found = False
        provider_used = None
        column_used = None

        # Check providers in priority order
        for provider in PROVIDER_PRIORITY:
            provider_col = feature_config.get(provider)

            if provider_col and provider_col in merged_ratings.columns:
                found = True
                provider_used = provider
                column_used = provider_col
                break

        availability[canonical_name] = {
            'available': found,
            'provider': provider_used,
            'column': column_used,
            'description': feature_config.get('description', '')
        }

    return availability


def print_feature_availability(merged_ratings: pd.DataFrame):
    """
    Print a report of available canonical features.

    Useful for debugging and understanding which features can be used.

    Args:
        merged_ratings: DataFrame from loaders.load_all_sources()['merged_ratings']
    """
    availability = get_available_features(merged_ratings)

    print("\n" + "="*80)
    print("CANONICAL FEATURE AVAILABILITY")
    print("="*80 + "\n")

    available_features = []
    unavailable_features = []

    for name, info in availability.items():
        if info['available']:
            available_features.append((name, info))
        else:
            unavailable_features.append((name, info))

    print(f"Available Features ({len(available_features)}):")
    print("-" * 80)
    for name, info in available_features:
        print(f"  {name:30s} ← {info['provider']:10s} ({info['column']})")
        print(f"    {info['description']}")

    if unavailable_features:
        print(f"\nUnavailable Features ({len(unavailable_features)}):")
        print("-" * 80)
        for name, info in unavailable_features:
            print(f"  {name:30s} - {info['description']}")

    print("\n" + "="*80 + "\n")


# ============================================================================
# FEATURE METADATA
# ============================================================================

def get_feature_info(canonical_name: str) -> Dict[str, Any]:
    """
    Get metadata about a canonical feature.

    Args:
        canonical_name: Canonical feature name

    Returns:
        Dict with feature configuration and metadata

    Raises:
        ValueError: If canonical_name is not recognized
    """
    if canonical_name not in CANONICAL_FEATURE_MAP:
        raise ValueError(f"Unknown canonical feature: '{canonical_name}'")

    return CANONICAL_FEATURE_MAP[canonical_name].copy()


def list_canonical_features() -> List[str]:
    """
    List all available canonical feature names.

    Returns:
        List of canonical feature names
    """
    return list(CANONICAL_FEATURE_MAP.keys())
