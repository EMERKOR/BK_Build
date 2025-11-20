"""
Data Schema Validation Module

Defines lightweight schema validators for all core data sources.
Validates that DataFrames have required columns and reasonable dtypes.
Raises clear ValueError messages when validation fails.

All validators follow the pattern:
    validate_xxx_df(df: pd.DataFrame) -> None

If validation fails, raises ValueError with helpful message.
If validation succeeds, returns None (fail-fast approach).
"""

import pandas as pd
from typing import List, Dict, Optional, Set


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

# nfelo historical game data (from nfelo_games.csv)
NFELO_HISTORICAL_SCHEMA = {
    'required_columns': [
        'game_id',
        'season',
        'week',
        'home_team',
        'away_team',
        'starting_nfelo_home',
        'starting_nfelo_away',
        'home_line_close',  # Vegas closing line
    ],
    'numeric_columns': [
        'season',
        'week',
        'starting_nfelo_home',
        'starting_nfelo_away',
        'home_line_close',
    ]
}

# nfelo current-season power ratings (weekly)
NFELO_POWER_RATINGS_SCHEMA = {
    'required_columns': [
        'team',  # Team abbreviation (normalized)
        'nfelo',  # Main ELO rating
    ],
    'numeric_columns': [
        'nfelo',
    ],
    'optional_columns': [
        'QB Adj',    # QB adjustment
        'Value',     # Overall value
        'WoW',       # Week over week change
        'YTD',       # Year to date performance
    ]
}

# nfelo EPA tiers (weekly)
NFELO_EPA_TIERS_SCHEMA = {
    'required_columns': [
        'team',
        'epa_off',   # Offensive EPA per play
        'epa_def',   # Defensive EPA per play
    ],
    'numeric_columns': [
        'epa_off',
        'epa_def',
    ],
    'optional_columns': [
        'epa_margin',  # EPA differential (off - def)
    ]
}

# nfelo strength of schedule (weekly)
NFELO_SOS_SCHEMA = {
    'required_columns': [
        'team',
    ],
    'numeric_columns': [
        # SOS columns vary, so we don't enforce specific numeric columns
    ],
    'optional_columns': [
        'SOS',
        'Remaining SOS',
    ]
}

# Substack power ratings (weekly)
SUBSTACK_POWER_RATINGS_SCHEMA = {
    'required_columns': [
        'team',
        'Off.',   # Offensive rating
        'Def.',   # Defensive rating
        'Ovr.',   # Overall rating
    ],
    'numeric_columns': [
        'Off.',
        'Def.',
        'Ovr.',
    ]
}

# Substack QB EPA (weekly)
SUBSTACK_QB_EPA_SCHEMA = {
    'required_columns': [
        'team',  # Team abbreviation
    ],
    'numeric_columns': [
        # QB EPA columns vary by provider
    ],
    'optional_columns': [
        'EPA',
        'EPA/Play',
        'EPA/Att',
        'CPOE',
        'Player',  # QB name
        'Pass',    # Pass attempts
    ]
}

# Substack weekly projections (weekly)
SUBSTACK_WEEKLY_PROJ_SCHEMA = {
    'required_columns': [
        # Note: This can be either matchup-based or team-based
        # We only require that it's a non-empty DataFrame
    ],
    'numeric_columns': [],
    'optional_columns': [
        'team_away',
        'team_home',
        'Favorite',
        'Win Prob.',
        'substack_spread_line',
    ]
}

# Team-week EPA aggregated statistics (from team_week_epa_2013_2024.csv)
TEAM_WEEK_EPA_SCHEMA = {
    'required_columns': [
        'season',
        'week',
        'team',
    ],
    'numeric_columns': [
        'season',
        'week',
    ],
    'optional_columns': [
        'off_epa_per_play',    # Offensive EPA per play
        'def_epa_per_play',    # Defensive EPA per play
        'off_success_rate',    # Offensive success rate
        'def_success_rate',    # Defensive success rate
        'off_epa_total',       # Total offensive EPA
        'def_epa_total',       # Total defensive EPA
        'off_plays',           # Number of offensive plays
        'def_plays',           # Number of defensive plays
    ]
}


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _check_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    data_source_name: str
) -> None:
    """
    Check that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        data_source_name: Human-readable name for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {data_source_name}: {sorted(missing_columns)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def _check_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: List[str],
    data_source_name: str
) -> None:
    """
    Check that specified columns contain numeric data.

    Args:
        df: DataFrame to validate
        numeric_columns: List of column names that should be numeric
        data_source_name: Human-readable name for error messages

    Raises:
        ValueError: If any columns are not numeric
    """
    for col in numeric_columns:
        if col not in df.columns:
            continue  # Skip if column doesn't exist

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' in {data_source_name} should be numeric, "
                f"but has dtype {df[col].dtype}"
            )


def _check_non_empty(
    df: pd.DataFrame,
    data_source_name: str
) -> None:
    """
    Check that DataFrame is non-empty.

    Args:
        df: DataFrame to validate
        data_source_name: Human-readable name for error messages

    Raises:
        ValueError: If DataFrame is empty
    """
    if len(df) == 0:
        raise ValueError(f"{data_source_name} DataFrame is empty (0 rows)")


def _check_optional_columns(
    df: pd.DataFrame,
    optional_columns: List[str],
    data_source_name: str
) -> None:
    """
    Check for optional columns and log warnings if missing.

    Does not raise exceptions - only logs warnings for missing columns.

    Args:
        df: DataFrame to validate
        optional_columns: List of optional column names
        data_source_name: Human-readable name for warning messages
    """
    import warnings

    missing_optional = set(optional_columns) - set(df.columns)

    if missing_optional:
        warnings.warn(
            f"Optional columns missing in {data_source_name}: {sorted(missing_optional)}. "
            f"Some features may not be available.",
            UserWarning
        )


# ============================================================================
# PUBLIC VALIDATION FUNCTIONS
# ============================================================================

def validate_nfelo_historical_df(df: pd.DataFrame) -> None:
    """
    Validate nfelo historical games DataFrame.

    Expected source: https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "nfelo historical games"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, NFELO_HISTORICAL_SCHEMA['required_columns'], data_source_name)
    _check_numeric_columns(df, NFELO_HISTORICAL_SCHEMA['numeric_columns'], data_source_name)


def validate_nfelo_power_ratings_df(df: pd.DataFrame) -> None:
    """
    Validate nfelo power ratings DataFrame (current season).

    Expected columns: team, nfelo, and optional QB Adj, Value, WoW, YTD

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "nfelo power ratings"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, NFELO_POWER_RATINGS_SCHEMA['required_columns'], data_source_name)
    _check_numeric_columns(df, NFELO_POWER_RATINGS_SCHEMA['numeric_columns'], data_source_name)


def validate_nfelo_epa_tiers_df(df: pd.DataFrame) -> None:
    """
    Validate nfelo EPA tiers DataFrame (current season).

    Expected columns: team, epa_off, epa_def, and optional epa_margin

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "nfelo EPA tiers"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, NFELO_EPA_TIERS_SCHEMA['required_columns'], data_source_name)
    _check_numeric_columns(df, NFELO_EPA_TIERS_SCHEMA['numeric_columns'], data_source_name)


def validate_nfelo_sos_df(df: pd.DataFrame) -> None:
    """
    Validate nfelo strength of schedule DataFrame (current season).

    Expected columns: team, and optional SOS metrics

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "nfelo strength of schedule"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, NFELO_SOS_SCHEMA['required_columns'], data_source_name)


def validate_substack_power_ratings_df(df: pd.DataFrame) -> None:
    """
    Validate Substack power ratings DataFrame (current season).

    Expected columns: team, Off., Def., Ovr.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "Substack power ratings"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, SUBSTACK_POWER_RATINGS_SCHEMA['required_columns'], data_source_name)
    _check_numeric_columns(df, SUBSTACK_POWER_RATINGS_SCHEMA['numeric_columns'], data_source_name)


def validate_substack_qb_epa_df(df: pd.DataFrame) -> None:
    """
    Validate Substack QB EPA DataFrame (current season).

    Expected columns: team, and optional EPA metrics

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "Substack QB EPA"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, SUBSTACK_QB_EPA_SCHEMA['required_columns'], data_source_name)


def validate_substack_weekly_proj_df(df: pd.DataFrame) -> None:
    """
    Validate Substack weekly projections DataFrame (current season).

    This is a flexible validator since weekly projections can be matchup-based
    or team-based, and column names vary.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "Substack weekly projections"

    _check_non_empty(df, data_source_name)
    # No strict column requirements for weekly projections
    # as format varies significantly


def validate_team_week_epa_df(df: pd.DataFrame) -> None:
    """
    Validate team-week EPA aggregated statistics DataFrame.

    Expected source: team_week_epa_2013_2024.csv (produced by aggregate_pbp_to_team_stats.py)

    Required columns: season, week, team
    Optional columns: EPA metrics (off_epa_per_play, def_epa_per_play, etc.)
                      and success rates (off_success_rate, def_success_rate)

    If optional columns are missing, a warning is logged but validation passes.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails with descriptive message
    """
    data_source_name = "team-week EPA"

    _check_non_empty(df, data_source_name)
    _check_required_columns(df, TEAM_WEEK_EPA_SCHEMA['required_columns'], data_source_name)
    _check_numeric_columns(df, TEAM_WEEK_EPA_SCHEMA['numeric_columns'], data_source_name)
    _check_optional_columns(df, TEAM_WEEK_EPA_SCHEMA['optional_columns'], data_source_name)
