"""
Ball Knower v2.0 - Dataset Loading and Training Frame Builder

This module provides the unified v2.0 dataset pipeline for loading, cleaning,
and merging all data sources into a canonical team-week training frame.

Core responsibilities:
- Load all raw source datasets defined in dataset_roles
- Apply column_name_mapping and drop ignored_columns
- Enforce canonical_schema (dtypes, required columns)
- Merge sources into a canonical team-week frame
- Build training frames with only safe features (T0/T1/T2)

Usage:
    >>> from ball_knower.v2.datasets import load_raw_sources, build_training_frame
    >>> sources = load_raw_sources("data")
    >>> training_frame = build_training_frame(sources, include_market=True)
"""

import os
import glob
import warnings
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

# Import configuration from Phase 2A
from ball_knower.v2.data_config import (
    STRUCTURAL_KEYS,
    TEAM_STRENGTH_FEATURES,
    MARKET_FEATURES,
    EXPERIMENTAL_FEATURES,
    FORBIDDEN_FEATURES,
    SAFE_FEATURES,
    column_name_mapping,
    dataset_roles,
    canonical_schema,
    ignored_columns,
    validate_feature_tier,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_dataset_file(data_dir: str, dataset_name: str) -> Optional[str]:
    """
    Find the file path for a given dataset using dataset_roles configuration.

    Args:
        data_dir: Root data directory
        dataset_name: Name of dataset from dataset_roles

    Returns:
        Full path to the dataset file, or None if not found
    """
    dataset_info = dataset_roles.get(dataset_name)
    if not dataset_info:
        return None

    # Check if there's a file pattern (for current season data)
    if 'file_pattern' in dataset_info:
        pattern = os.path.join(data_dir, dataset_info['file_pattern'])
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent file (assuming naming convention includes date/week)
            return sorted(matches)[-1]
        return None

    # For historical data, try standard naming
    possible_names = [
        f"{dataset_name}.csv",
        f"{dataset_name}.parquet",
    ]

    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    return None


def _apply_column_mapping(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Apply column name mapping to standardize raw column names.

    Args:
        df: Raw dataframe
        dataset_name: Name of the dataset (for logging)

    Returns:
        DataFrame with canonical column names
    """
    # Create a mapping for columns that exist in both the df and the mapping
    rename_map = {
        old: new for old, new in column_name_mapping.items()
        if old in df.columns
    }

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _drop_ignored_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are explicitly marked as ignored.

    Args:
        df: DataFrame with canonical column names

    Returns:
        DataFrame with ignored columns removed
    """
    cols_to_drop = [col for col in ignored_columns if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def _enforce_dtype(df: pd.DataFrame, col: str, target_dtype: str) -> pd.DataFrame:
    """
    Attempt to cast a column to the target dtype.

    Args:
        df: DataFrame
        col: Column name
        target_dtype: Target dtype from canonical_schema

    Returns:
        DataFrame with column cast if possible
    """
    if col not in df.columns:
        return df

    try:
        if target_dtype == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        elif target_dtype == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif target_dtype == 'str':
            df[col] = df[col].astype(str)
        # Add more dtype conversions as needed
    except Exception as e:
        warnings.warn(f"Could not cast column '{col}' to {target_dtype}: {e}")

    return df


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def load_raw_sources(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load all v2.0 source datasets using dataset_roles and column_name_mapping.

    This function:
    1. Finds each dataset file using file patterns from dataset_roles
    2. Loads the CSV with appropriate skip_rows settings
    3. Applies column_name_mapping to standardize names
    4. Drops ignored_columns

    Args:
        data_dir: Root directory containing data files

    Returns:
        Dictionary mapping dataset_name -> cleaned DataFrame (canonical column names)

    Example:
        >>> sources = load_raw_sources("data")
        >>> print(sources.keys())
        dict_keys(['team_week_epa_2013_2024', 'power_ratings_nfelo', ...])
    """
    loaded_sources = {}

    for dataset_name, dataset_info in dataset_roles.items():
        file_path = _find_dataset_file(data_dir, dataset_name)

        if file_path is None:
            warnings.warn(
                f"Dataset '{dataset_name}' not found. Expected pattern: "
                f"{dataset_info.get('file_pattern', dataset_name + '.csv')}"
            )
            continue

        try:
            # Get skip_rows setting if specified
            skip_rows = dataset_info.get('skip_rows', None)

            # Load the dataset
            df = pd.read_csv(file_path, skiprows=skip_rows)

            # Apply column mapping
            df = _apply_column_mapping(df, dataset_name)

            # Drop ignored columns
            df = _drop_ignored_columns(df)

            # Add season if not present (for current season data)
            if 'season' not in df.columns and 'time_range' in dataset_info:
                season_year = dataset_info['time_range'][0]
                df['season'] = season_year

            loaded_sources[dataset_name] = df

            print(f"✓ Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")

        except Exception as e:
            warnings.warn(f"Failed to load '{dataset_name}' from {file_path}: {e}")
            continue

    return loaded_sources


def enforce_canonical_schema(
    df: pd.DataFrame,
    dataset_name: str,
    strict: bool = False
) -> pd.DataFrame:
    """
    Enforce canonical_schema constraints for a given dataset.

    This function:
    1. Checks that expected columns exist
    2. Casts dtypes where possible
    3. Warns or raises on missing/extra columns based on strict mode

    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset (for informative warnings)
        strict: If True, raise errors on schema violations. If False, warn only.

    Returns:
        DataFrame with enforced schema

    Raises:
        ValueError: If strict=True and schema violations are found
    """
    dataset_info = dataset_roles.get(dataset_name)
    if not dataset_info:
        warnings.warn(f"No dataset_roles entry for '{dataset_name}'")
        return df

    expected_columns = set(dataset_info.get('expected_columns', []))
    actual_columns = set(df.columns)

    # Check for missing expected columns
    missing_cols = expected_columns - actual_columns
    if missing_cols:
        msg = f"Dataset '{dataset_name}' missing expected columns: {missing_cols}"
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)

    # Check for extra columns (informational only)
    extra_cols = actual_columns - expected_columns - set(ignored_columns)
    if extra_cols:
        # Filter to only show columns that aren't in any tier (truly unknown)
        unknown_cols = [col for col in extra_cols if validate_feature_tier(col) == 'UNKNOWN']
        if unknown_cols:
            warnings.warn(
                f"Dataset '{dataset_name}' has columns not in canonical_schema: {unknown_cols[:5]}"
                + (f" (and {len(unknown_cols) - 5} more)" if len(unknown_cols) > 5 else "")
            )

    # Enforce dtypes for columns present in canonical_schema
    for col in actual_columns:
        if col in canonical_schema:
            target_dtype = canonical_schema[col]['dtype']
            df = _enforce_dtype(df, col, target_dtype)

    return df


def build_team_week_frame(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all cleaned sources into a canonical team-week frame.

    This function creates a unified dataset keyed by [season, week, team] and includes
    only features from T0/T1/T2 (safe features). TX_FORBIDDEN and T3_EXPERIMENTAL
    features are excluded.

    Merge strategy:
    1. Start with team_week_epa_2013_2024 (historical weekly data)
    2. Join power ratings by [season, team] (current season)
    3. Join schedule/SOS data by [season, team]
    4. Join QB data by team (latest season)
    5. Exclude all forbidden and experimental features

    Args:
        sources: Dictionary of cleaned DataFrames from load_raw_sources()

    Returns:
        DataFrame with canonical team-week structure and only T0/T1/T2 features

    Example:
        >>> team_week = build_team_week_frame(sources)
        >>> print(team_week.columns.tolist())
        ['season', 'week', 'team', 'off_epa_per_play', ...]
    """
    # Start with historical EPA data as the base
    if 'team_week_epa_2013_2024' not in sources:
        raise ValueError("Missing required dataset: team_week_epa_2013_2024")

    base_df = sources['team_week_epa_2013_2024'].copy()
    print(f"Base frame: {base_df.shape[0]} rows (historical team-week EPA data)")

    # Define safe column sets
    safe_cols = set(STRUCTURAL_KEYS + TEAM_STRENGTH_FEATURES + MARKET_FEATURES)
    forbidden_cols = set(FORBIDDEN_FEATURES + EXPERIMENTAL_FEATURES)

    # Helper function to filter columns to safe features only
    def filter_safe_columns(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        """Keep only key columns and safe feature columns."""
        available_cols = [
            col for col in df.columns
            if col in keys or (col in safe_cols and col not in forbidden_cols)
        ]
        return df[available_cols]

    # Join power ratings (nfelo) by [season, team]
    if 'power_ratings_nfelo' in sources:
        nfelo_df = filter_safe_columns(
            sources['power_ratings_nfelo'],
            keys=['season', 'team']
        )
        # For current season data, broadcast to all weeks
        # Merge on season + team (will duplicate for each week)
        base_df = base_df.merge(
            nfelo_df,
            on=['season', 'team'],
            how='left',
            suffixes=('', '_nfelo')
        )
        print(f"After nfelo merge: {base_df.shape[0]} rows, {base_df.shape[1]} columns")

    # Join power ratings (substack) by [team] for current season only
    if 'power_ratings_substack' in sources:
        substack_df = filter_safe_columns(
            sources['power_ratings_substack'],
            keys=['team']
        )
        # Add season to match current year
        if 'season' not in substack_df.columns:
            substack_df['season'] = 2025  # Current season

        base_df = base_df.merge(
            substack_df,
            on=['season', 'team'],
            how='left',
            suffixes=('', '_substack')
        )
        print(f"After substack merge: {base_df.shape[0]} rows, {base_df.shape[1]} columns")

    # Join EPA tiers (nfelo) by [season, team]
    if 'epa_tiers_nfelo' in sources:
        epa_tiers_df = filter_safe_columns(
            sources['epa_tiers_nfelo'],
            keys=['season', 'team']
        )
        # These columns likely overlap with team_week_epa, so use outer join and coalesce
        base_df = base_df.merge(
            epa_tiers_df,
            on=['season', 'team'],
            how='left',
            suffixes=('', '_tiers')
        )
        # Coalesce overlapping EPA columns (prefer original)
        for col in ['off_epa_per_play', 'def_epa_per_play']:
            if f"{col}_tiers" in base_df.columns:
                base_df[col] = base_df[col].fillna(base_df[f"{col}_tiers"])
                base_df = base_df.drop(columns=[f"{col}_tiers"])
        print(f"After EPA tiers merge: {base_df.shape[0]} rows, {base_df.shape[1]} columns")

    # Join strength of schedule by [season, team]
    if 'strength_of_schedule_nfelo' in sources:
        sos_df = filter_safe_columns(
            sources['strength_of_schedule_nfelo'],
            keys=['season', 'team']
        )
        base_df = base_df.merge(
            sos_df,
            on=['season', 'team'],
            how='left',
            suffixes=('', '_sos')
        )
        print(f"After SOS merge: {base_df.shape[0]} rows, {base_df.shape[1]} columns")

    # Final cleanup: remove any forbidden or experimental features that slipped through
    final_cols = [
        col for col in base_df.columns
        if col not in forbidden_cols
    ]
    base_df = base_df[final_cols]

    # Remove duplicate columns from suffixes
    base_df = base_df.loc[:, ~base_df.columns.duplicated()]

    print(f"\nFinal team-week frame: {base_df.shape[0]} rows, {base_df.shape[1]} columns")

    return base_df


def build_training_frame(
    sources: Dict[str, pd.DataFrame],
    include_market: bool = True,
) -> pd.DataFrame:
    """
    Construct the v2.0 training frame for modeling.

    This function creates the final training dataset by:
    1. Building the team-week frame from all sources
    2. Optionally including/excluding T2_MARKET features
    3. Ensuring NO TX_FORBIDDEN or T3_EXPERIMENTAL features are present
    4. Validating the final feature set

    Args:
        sources: Dictionary of cleaned DataFrames from load_raw_sources()
        include_market: If True, include T2_MARKET features. If False, only T0+T1.

    Returns:
        DataFrame ready for model training with validated feature set

    Example:
        >>> training_frame = build_training_frame(sources, include_market=True)
        >>> print(training_frame.shape)
        (5234, 47)  # Example: 5234 team-week observations, 47 features
    """
    # Build the base team-week frame
    df = build_team_week_frame(sources)

    # Define allowed feature sets
    allowed_features = set(STRUCTURAL_KEYS + TEAM_STRENGTH_FEATURES)
    if include_market:
        allowed_features.update(MARKET_FEATURES)

    forbidden_features = set(FORBIDDEN_FEATURES + EXPERIMENTAL_FEATURES)

    # Filter to only allowed columns
    final_cols = [
        col for col in df.columns
        if col in allowed_features or col not in (allowed_features | forbidden_features)
    ]
    df = df[final_cols]

    # Validate: ensure no forbidden features are present
    actual_cols = set(df.columns)
    forbidden_present = actual_cols & forbidden_features
    if forbidden_present:
        raise ValueError(
            f"CRITICAL: Forbidden features found in training frame: {forbidden_present}"
        )

    # Report feature tier breakdown
    t0_cols = [col for col in df.columns if col in STRUCTURAL_KEYS]
    t1_cols = [col for col in df.columns if col in TEAM_STRENGTH_FEATURES]
    t2_cols = [col for col in df.columns if col in MARKET_FEATURES]

    print("\n" + "="*70)
    print("TRAINING FRAME FEATURE BREAKDOWN")
    print("="*70)
    print(f"T0 (Structural):      {len(t0_cols):3d} columns")
    print(f"T1 (Core Strength):   {len(t1_cols):3d} columns")
    print(f"T2 (Market):          {len(t2_cols):3d} columns")
    print(f"Total:                {len(df.columns):3d} columns")
    print(f"Observations:         {len(df):,} team-week records")
    print("="*70)

    # Drop rows with missing keys
    key_cols = ['season', 'week', 'team']
    missing_keys = df[key_cols].isna().any(axis=1)
    if missing_keys.sum() > 0:
        warnings.warn(f"Dropping {missing_keys.sum()} rows with missing keys")
        df = df[~missing_keys]

    # Sort by season, week, team for consistency
    df = df.sort_values(['season', 'week', 'team']).reset_index(drop=True)

    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get a summary of features by tier in the given DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary mapping tier name to list of column names

    Example:
        >>> summary = get_feature_summary(training_frame)
        >>> print(summary['T1'][:5])
        ['off_epa_per_play', 'def_epa_per_play', 'nfelo', ...]
    """
    summary = {
        'T0': [],
        'T1': [],
        'T2': [],
        'T3': [],
        'TX': [],
        'UNKNOWN': [],
    }

    for col in df.columns:
        tier = validate_feature_tier(col)
        summary[tier].append(col)

    return summary


def validate_training_frame(df: pd.DataFrame, strict: bool = True) -> bool:
    """
    Validate that a training frame meets v2.0 requirements.

    Checks:
    1. Has required key columns [season, week, team]
    2. Contains no TX_FORBIDDEN features
    3. Contains no T3_EXPERIMENTAL features
    4. All features are either T0/T1/T2 or explicitly allowed

    Args:
        df: Training frame to validate
        strict: If True, raise errors. If False, return True/False.

    Returns:
        True if valid, False otherwise (or raises if strict=True)

    Raises:
        ValueError: If strict=True and validation fails
    """
    issues = []

    # Check for required keys
    required_keys = ['season', 'week', 'team']
    missing_keys = [k for k in required_keys if k not in df.columns]
    if missing_keys:
        issues.append(f"Missing required keys: {missing_keys}")

    # Check for forbidden features
    forbidden_present = [col for col in df.columns if col in FORBIDDEN_FEATURES]
    if forbidden_present:
        issues.append(f"Forbidden features present: {forbidden_present}")

    # Check for experimental features
    experimental_present = [col for col in df.columns if col in EXPERIMENTAL_FEATURES]
    if experimental_present:
        issues.append(f"Experimental features present: {experimental_present}")

    if issues:
        msg = "Training frame validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return False

    return True


# ============================================================================
# MODULE INFO
# ============================================================================

__all__ = [
    'load_raw_sources',
    'enforce_canonical_schema',
    'build_team_week_frame',
    'build_training_frame',
    'get_feature_summary',
    'validate_training_frame',
]

if __name__ == '__main__':
    # Quick test when run directly
    print("\n" + "="*70)
    print("BALL KNOWER v2.0 - DATASET PIPELINE TEST")
    print("="*70 + "\n")

    print("Loading sources...")
    sources = load_raw_sources("data")

    print("\nBuilding training frame...")
    training_frame = build_training_frame(sources, include_market=True)

    print("\nValidating...")
    is_valid = validate_training_frame(training_frame, strict=False)
    print(f"✓ Validation {'passed' if is_valid else 'failed'}")

    print("\nFeature summary:")
    summary = get_feature_summary(training_frame)
    for tier in ['T0', 'T1', 'T2']:
        print(f"\n{tier} features ({len(summary[tier])}):")
        print("  " + ", ".join(summary[tier][:10]))
        if len(summary[tier]) > 10:
            print(f"  ... and {len(summary[tier]) - 10} more")
