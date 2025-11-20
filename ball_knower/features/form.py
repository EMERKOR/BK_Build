"""
Team Form Feature Module (v1.3 Placeholder)

This module is a placeholder for future v1.3 development.
Team "form" features will capture rolling offensive/defensive efficiency
trends beyond simple win/loss records.

IMPORTANT: This module is NOT used by v1.2.
Do not integrate these features into v1.2 pipelines.

Future v1.3 features may include:
- Rolling offensive efficiency (EPA, yards per play, success rate)
- Rolling defensive efficiency (opponent EPA, yards allowed, etc.)
- Momentum indicators (improving vs. declining performance)
- Context-aware form (vs. good/bad opponents)

Current Status: PLACEHOLDER ONLY
"""

import pandas as pd
import warnings


def compute_offense_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling offensive efficiency over N games.

    PLACEHOLDER: Not yet implemented.

    Args:
        df: DataFrame with team-level offensive stats

    Returns:
        DataFrame with offense_form features

    Raises:
        NotImplementedError: This is a placeholder for v1.3
    """
    raise NotImplementedError(
        "compute_offense_form is a placeholder for v1.3. "
        "Do not use in v1.2 pipelines."
    )


def compute_defense_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling defensive efficiency over N games.

    PLACEHOLDER: Not yet implemented.

    Args:
        df: DataFrame with team-level defensive stats

    Returns:
        DataFrame with defense_form features

    Raises:
        NotImplementedError: This is a placeholder for v1.3
    """
    raise NotImplementedError(
        "compute_defense_form is a placeholder for v1.3. "
        "Do not use in v1.2 pipelines."
    )


def compute_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for computing all team form features.

    PLACEHOLDER: Not yet implemented.

    This will eventually combine offensive and defensive form metrics
    into a unified DataFrame suitable for model training.

    Args:
        df: DataFrame with team-level game data

    Returns:
        DataFrame with team_form features (offense + defense)

    Raises:
        NotImplementedError: This is a placeholder for v1.3
    """
    raise NotImplementedError(
        "compute_team_form is a placeholder for v1.3. "
        "Do not use in v1.2 pipelines."
    )


# Guard against accidental imports
warnings.warn(
    "ball_knower.features.form is a placeholder module for v1.3. "
    "These features are NOT implemented and should NOT be used in v1.2.",
    FutureWarning,
    stacklevel=2
)
