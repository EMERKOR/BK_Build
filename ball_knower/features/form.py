"""
Team Form Feature Module (v1.3)

Computes leak-free rolling offensive and defensive efficiency features.

Team "form" features capture rolling offensive/defensive efficiency
trends beyond simple win/loss records.

IMPORTANT: This module is NOT used by v1.2.
Do not integrate these features into v1.2 pipelines.

Features include:
- Rolling 4-game offensive efficiency (EPA, success rate)
- Rolling 4-game defensive efficiency (EPA, success rate)
- Strictly leak-free: uses .shift(1) before rolling calculations

All rolling calculations exclude the current game to prevent data leakage.
"""

import pandas as pd
import warnings


def compute_offense_form(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Compute rolling offensive efficiency over N games.

    LEAK-FREE: Uses .shift(1) to exclude current game before rolling calculations.

    Args:
        df: DataFrame with team-level offensive stats.
            Required columns: team, season, week, off_epa_per_play, off_success_rate
        window: Number of games for rolling window (default: 4)

    Returns:
        DataFrame with columns:
            - team
            - season
            - week
            - offense_form_epa: Rolling N-game average of off_epa_per_play
            - offense_form_success: Rolling N-game average of off_success_rate

    Note:
        - Input DataFrame should be pre-sorted by team, season, week
        - Returns NaN for first N games where insufficient history exists
        - Strictly leak-free: current game is excluded from rolling calculation
    """
    # Ensure DataFrame is sorted correctly for rolling calculations
    df = df.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    # Create a copy with only needed columns
    result = df[['team', 'season', 'week']].copy()

    # Check for required columns
    if 'off_epa_per_play' not in df.columns or 'off_success_rate' not in df.columns:
        warnings.warn(
            "Missing offensive metrics (off_epa_per_play or off_success_rate). "
            "Offense form features will be NaN.",
            UserWarning
        )
        result['offense_form_epa'] = pd.NA
        result['offense_form_success'] = pd.NA
        return result

    # Compute rolling offensive EPA (LEAK-FREE: shift(1) before rolling)
    result['offense_form_epa'] = (
        df.groupby('team')['off_epa_per_play']
        .shift(1)  # Exclude current game
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Compute rolling offensive success rate (LEAK-FREE: shift(1) before rolling)
    result['offense_form_success'] = (
        df.groupby('team')['off_success_rate']
        .shift(1)  # Exclude current game
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return result


def compute_defense_form(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Compute rolling defensive efficiency over N games.

    LEAK-FREE: Uses .shift(1) to exclude current game before rolling calculations.

    Args:
        df: DataFrame with team-level defensive stats.
            Required columns: team, season, week, def_epa_per_play, def_success_rate
        window: Number of games for rolling window (default: 4)

    Returns:
        DataFrame with columns:
            - team
            - season
            - week
            - defense_form_epa: Rolling N-game average of def_epa_per_play
            - defense_form_success: Rolling N-game average of def_success_rate

    Note:
        - Input DataFrame should be pre-sorted by team, season, week
        - Returns NaN for first N games where insufficient history exists
        - Strictly leak-free: current game is excluded from rolling calculation
        - Lower defensive EPA is better (negative is good for defense)
    """
    # Ensure DataFrame is sorted correctly for rolling calculations
    df = df.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    # Create a copy with only needed columns
    result = df[['team', 'season', 'week']].copy()

    # Check for required columns
    if 'def_epa_per_play' not in df.columns or 'def_success_rate' not in df.columns:
        warnings.warn(
            "Missing defensive metrics (def_epa_per_play or def_success_rate). "
            "Defense form features will be NaN.",
            UserWarning
        )
        result['defense_form_epa'] = pd.NA
        result['defense_form_success'] = pd.NA
        return result

    # Compute rolling defensive EPA (LEAK-FREE: shift(1) before rolling)
    result['defense_form_epa'] = (
        df.groupby('team')['def_epa_per_play']
        .shift(1)  # Exclude current game
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Compute rolling defensive success rate (LEAK-FREE: shift(1) before rolling)
    result['defense_form_success'] = (
        df.groupby('team')['def_success_rate']
        .shift(1)  # Exclude current game
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return result


def compute_team_form(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Wrapper for computing all team form features.

    Combines offensive and defensive form metrics into a unified DataFrame
    suitable for model training.

    LEAK-FREE: All rolling calculations exclude the current game via .shift(1).

    Args:
        df: DataFrame with team-level game data.
            Required columns: team, season, week,
                             off_epa_per_play, off_success_rate,
                             def_epa_per_play, def_success_rate
        window: Number of games for rolling window (default: 4)

    Returns:
        DataFrame with columns:
            - team
            - season
            - week
            - offense_form_epa
            - offense_form_success
            - defense_form_epa
            - defense_form_success

    Note:
        - Input DataFrame should be pre-sorted by team, season, week
        - Returns NaN for first N games where insufficient history exists
        - Strictly leak-free: current game is excluded from all rolling calculations
    """
    # Compute offensive form
    offense_form = compute_offense_form(df, window=window)

    # Compute defensive form
    defense_form = compute_defense_form(df, window=window)

    # Merge on team, season, week
    team_form = offense_form.merge(
        defense_form[['team', 'season', 'week', 'defense_form_epa', 'defense_form_success']],
        on=['team', 'season', 'week'],
        how='left'
    )

    return team_form


# Guard against accidental imports in v1.2 pipelines
warnings.warn(
    "ball_knower.features.form is a v1.3-only module. "
    "These features should NOT be used in v1.2 pipelines.",
    FutureWarning,
    stacklevel=2
)
