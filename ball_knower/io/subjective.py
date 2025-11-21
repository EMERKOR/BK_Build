"""
Subjective Adjustment Layer

Allows manual adjustments to model predictions based on subjective analysis,
injury reports, weather conditions, or other factors not captured by the model.

File format: YAML
Location: data/current_season/subjective/subjective_{season}_week_{week}.yaml

Example YAML:
    KC:
      adjustment: 1.5
      reason: "Mahomes magic in primetime"
    GB:
      adjustment: -2.0
      reason: "Injury cluster on offensive line"
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
import pandas as pd

# Try to import YAML, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn(
        "PyYAML not installed. Subjective adjustments will not be available. "
        "Install with: pip install pyyaml",
        UserWarning
    )


# Default data directory
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "BALL_KNOWER_DATA_DIR",
        str(_DEFAULT_ROOT / "data" / "current_season"),
    )
)


def load_subjective_adjustments(
    season: int,
    week: int,
    data_dir: Optional[Path] = None
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Load subjective adjustments for a given season and week.

    Looks for: data/current_season/subjective/subjective_{season}_week_{week}.yaml

    Args:
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory (defaults to DEFAULT_DATA_DIR)

    Returns:
        Tuple of two dicts:
            - adjustments: {team_code: adjustment_value}
            - reasons: {team_code: reason_string}

    Example:
        >>> adjustments, reasons = load_subjective_adjustments(2023, 11)
        >>> print(adjustments)
        {'KC': 1.5, 'GB': -2.0}
        >>> print(reasons)
        {'KC': 'Mahomes magic', 'GB': 'Injury cluster'}
    """
    if not HAS_YAML:
        # Return empty dicts if YAML not available
        return {}, {}

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    subjective_dir = data_dir / "subjective"
    file_path = subjective_dir / f"subjective_{season}_week_{week}.yaml"

    # If file doesn't exist, return empty dicts
    if not file_path.exists():
        return {}, {}

    # Load and parse YAML
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            # Empty file
            return {}, {}

        # Parse into adjustments and reasons dicts
        adjustments = {}
        reasons = {}

        for team_code, team_data in data.items():
            if not isinstance(team_data, dict):
                warnings.warn(
                    f"Invalid format for team '{team_code}' in {file_path.name}. "
                    f"Expected dict with 'adjustment' and 'reason' keys. Skipping.",
                    UserWarning
                )
                continue

            # Extract adjustment value
            adjustment = team_data.get('adjustment')
            if adjustment is None:
                warnings.warn(
                    f"Missing 'adjustment' for team '{team_code}' in {file_path.name}. Skipping.",
                    UserWarning
                )
                continue

            # Validate that adjustment is numeric
            try:
                adjustment_float = float(adjustment)
            except (TypeError, ValueError):
                warnings.warn(
                    f"Invalid adjustment value for team '{team_code}' in {file_path.name}: {adjustment}. "
                    f"Expected numeric value. Skipping.",
                    UserWarning
                )
                continue

            # Extract reason (optional)
            reason = team_data.get('reason', '')

            # Store in dicts
            adjustments[team_code] = adjustment_float
            reasons[team_code] = str(reason)

        return adjustments, reasons

    except yaml.YAMLError as e:
        warnings.warn(
            f"Error parsing YAML file {file_path}: {e}. "
            f"Subjective adjustments will not be applied.",
            UserWarning
        )
        return {}, {}
    except Exception as e:
        warnings.warn(
            f"Error loading subjective adjustments from {file_path}: {e}",
            UserWarning
        )
        return {}, {}


def apply_subjective_adjustments(
    df: pd.DataFrame,
    adjustments: Dict[str, float],
    reasons: Dict[str, str]
) -> pd.DataFrame:
    """
    Apply subjective adjustments to prediction DataFrame.

    Args:
        df: DataFrame with columns: home_team, away_team, bk_line (or model_line)
        adjustments: Dict mapping team codes to adjustment values
        reasons: Dict mapping team codes to reason strings

    Returns:
        DataFrame with added columns:
            - subjective_home: Adjustment for home team
            - subjective_away: Adjustment for away team
            - subjective_reason_home: Reason for home adjustment
            - subjective_reason_away: Reason for away adjustment
            - final_bk_line: Adjusted line (model_line + subjective_home - subjective_away)

    Note:
        Adjustments are signed:
            - Positive = team is better than model thinks
            - Negative = team is worse than model thinks

        For spreads (negative = home favored):
            - Positive home adjustment makes home more favored (line more negative)
            - Positive away adjustment makes away more favored (line more positive)
    """
    df = df.copy()

    # Determine which column contains the base model line
    if 'model_line' in df.columns:
        base_line_col = 'model_line'
    elif 'bk_line' in df.columns:
        base_line_col = 'bk_line'
    else:
        raise ValueError(
            "DataFrame must contain either 'model_line' or 'bk_line' column"
        )

    # Determine team column names (support both team_home/team_away and home_team/away_team)
    if 'team_home' in df.columns and 'team_away' in df.columns:
        home_col = 'team_home'
        away_col = 'team_away'
    elif 'home_team' in df.columns and 'away_team' in df.columns:
        home_col = 'home_team'
        away_col = 'away_team'
    else:
        raise ValueError(
            "DataFrame must contain either 'team_home'/'team_away' or 'home_team'/'away_team' columns"
        )

    # Add subjective adjustment columns
    df['subjective_home'] = df[home_col].map(adjustments).fillna(0.0)
    df['subjective_away'] = df[away_col].map(adjustments).fillna(0.0)

    # Add reason columns
    df['subjective_reason_home'] = df[home_col].map(reasons).fillna('')
    df['subjective_reason_away'] = df[away_col].map(reasons).fillna('')

    # Compute final line
    # Spread convention: negative = home favored
    # If home team gets +1.5 adjustment (better), line becomes more negative (home more favored)
    # If away team gets +1.5 adjustment (better), line becomes more positive (away more favored)
    df['final_bk_line'] = (
        df[base_line_col] +
        (df['subjective_home'] - df['subjective_away'])
    )

    return df
