"""
v1.2 Feature Engineering

Defines the canonical feature set for v1.2 residual model.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


# Canonical feature columns for v1.2 model
# These match the features in the v1.2 training dataset
FEATURE_COLUMNS = [
    'diff_nfelo_power',    # Primary: nfelo rating differential
    'rest_advantage',      # Combined rest/bye week effect
    'div_game',            # Divisional game flag
    'surface_diff',        # Surface difference modifier
    'time_advantage',      # Time zone advantage
    'diff_qb_adj',         # QB quality differential
]


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract feature matrix from v1.2 training dataset.

    Args:
        df: DataFrame from v1_2 dataset builder (must contain FEATURE_COLUMNS)

    Returns:
        Tuple of (X, feature_names) where:
            X: Feature matrix (DataFrame)
            feature_names: List of feature column names

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> df = pd.read_parquet("data/v1_2_training_dataset.parquet")
        >>> X, features = get_feature_matrix(df)
        >>> print(X.shape)
        >>> print(features)
    """

    # Verify all required features exist
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Extract feature matrix
    X = df[FEATURE_COLUMNS].copy()

    # Verify no NaN values in features
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Features contain NaN values in columns: {nan_cols}")

    return X, FEATURE_COLUMNS


def get_target_vector(df: pd.DataFrame) -> pd.Series:
    """
    Extract target vector (residual) from v1.2 training dataset.

    The target is the residual between Vegas closing line and our base prediction:
        residual = vegas_closing_spread - bk_base_spread

    This allows the model to learn a correction factor on top of the base nfelo spread.

    Args:
        df: DataFrame from v1_2 dataset builder

    Returns:
        Series containing residual values

    Raises:
        ValueError: If required columns are missing or contain NaN

    Example:
        >>> df = pd.read_parquet("data/v1_2_training_dataset.parquet")
        >>> y = get_target_vector(df)
        >>> print(y.describe())
    """

    # Verify required columns exist
    required = ['vegas_closing_spread', 'bk_base_spread']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required target columns: {missing}")

    # Calculate residual
    # residual = vegas line - base prediction
    # If residual is positive, Vegas thinks home team is stronger than our base model
    # If residual is negative, Vegas thinks home team is weaker than our base model
    residual = df['vegas_closing_spread'] - df['bk_base_spread']

    # Verify no NaN values
    if residual.isna().any():
        raise ValueError("Target vector contains NaN values")

    return residual


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame has all required columns for v1.2 training.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails
    """

    # Check for required identifier columns
    identifiers = ['game_id', 'season', 'week', 'home_team', 'away_team']
    missing_ids = [col for col in identifiers if col not in df.columns]
    if missing_ids:
        raise ValueError(f"Missing identifier columns: {missing_ids}")

    # Check for required feature columns
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # Check for required target columns
    target_cols = ['vegas_closing_spread', 'bk_base_spread']
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    # Check for NaN values in critical columns
    critical_cols = FEATURE_COLUMNS + target_cols
    for col in critical_cols:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            raise ValueError(f"Column '{col}' contains {nan_count} NaN values")

    print(f"✓ Dataset validation passed ({len(df):,} rows)")
