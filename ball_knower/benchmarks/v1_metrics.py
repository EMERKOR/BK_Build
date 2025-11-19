"""
Ball Knower v1.0 Performance Metrics

Computes performance metrics for v1.0 spread predictions vs actual outcomes.

This module provides reusable functions for:
- Computing prediction errors (model vs actual, market vs actual)
- Calculating summary statistics (MAE, mean error, percentiles)
- Comparing model performance to Vegas/market baselines
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_v1_0_metrics(
    df: pd.DataFrame,
    model_col: str = 'model_spread',
    market_col: str = 'market_spread',
    actual_col: str = 'actual_margin'
) -> Dict[str, float]:
    """
    Compute performance metrics for v1.0 spread predictions.

    This function compares both model and market (Vegas) predictions to
    actual game outcomes, computing error statistics for both.

    Args:
        df: DataFrame with predictions and actual outcomes
        model_col: Column name for model spread predictions
        market_col: Column name for market (Vegas) spread
        actual_col: Column name for actual margin (home - away score)

    Returns:
        Dictionary with metrics:
            - n_games: Number of games analyzed
            - model_mae: Mean absolute error for model predictions
            - market_mae: Mean absolute error for market predictions
            - model_mean_error: Mean signed error for model (bias check)
            - market_mean_error: Mean signed error for market (bias check)
            - model_mae_improvement: Points better than market (negative = worse)
            - model_error_pct_50: 50th percentile of model absolute errors
            - model_error_pct_75: 75th percentile of model absolute errors
            - model_error_pct_90: 90th percentile of model absolute errors
            - market_error_pct_50: 50th percentile of market absolute errors
            - market_error_pct_75: 75th percentile of market absolute errors
            - market_error_pct_90: 90th percentile of market absolute errors

    Example:
        >>> df = pd.DataFrame({
        ...     'model_spread': [-3.5, -7.0, 2.5],
        ...     'market_spread': [-4.0, -6.5, 3.0],
        ...     'actual_margin': [-5, -10, 1]
        ... })
        >>> metrics = compute_v1_0_metrics(df)
        >>> print(f"Model MAE: {metrics['model_mae']:.2f}")
    """
    # Ensure no missing values
    df_clean = df[[model_col, market_col, actual_col]].dropna()

    if len(df_clean) == 0:
        raise ValueError("No valid rows with all required columns")

    # Compute errors
    model_error = df_clean[model_col] - df_clean[actual_col]
    market_error = df_clean[market_col] - df_clean[actual_col]

    # Absolute errors
    abs_model_error = model_error.abs()
    abs_market_error = market_error.abs()

    # Summary statistics
    metrics = {
        'n_games': len(df_clean),
        'model_mae': abs_model_error.mean(),
        'market_mae': abs_market_error.mean(),
        'model_mean_error': model_error.mean(),
        'market_mean_error': market_error.mean(),
    }

    # MAE improvement (negative = model is worse)
    metrics['model_mae_improvement'] = metrics['market_mae'] - metrics['model_mae']

    # Percentiles for error distribution analysis
    metrics['model_error_pct_50'] = abs_model_error.quantile(0.50)
    metrics['model_error_pct_75'] = abs_model_error.quantile(0.75)
    metrics['model_error_pct_90'] = abs_model_error.quantile(0.90)

    metrics['market_error_pct_50'] = abs_market_error.quantile(0.50)
    metrics['market_error_pct_75'] = abs_market_error.quantile(0.75)
    metrics['market_error_pct_90'] = abs_market_error.quantile(0.90)

    return metrics


def compute_v1_0_errors(
    df: pd.DataFrame,
    model_col: str = 'model_spread',
    market_col: str = 'market_spread',
    actual_col: str = 'actual_margin'
) -> pd.DataFrame:
    """
    Add error columns to a DataFrame for detailed analysis.

    Args:
        df: DataFrame with predictions and actual outcomes
        model_col: Column name for model spread predictions
        market_col: Column name for market (Vegas) spread
        actual_col: Column name for actual margin

    Returns:
        DataFrame with added columns:
            - model_error: model_spread - actual_margin
            - market_error: market_spread - actual_margin
            - abs_model_error: |model_error|
            - abs_market_error: |market_error|

    Example:
        >>> df = pd.DataFrame({
        ...     'game_id': ['2024_01_KC_BUF'],
        ...     'model_spread': [-3.5],
        ...     'market_spread': [-4.0],
        ...     'actual_margin': [-5]
        ... })
        >>> df_with_errors = compute_v1_0_errors(df)
        >>> print(df_with_errors[['model_error', 'market_error']])
    """
    df = df.copy()

    # Compute signed errors
    df['model_error'] = df[model_col] - df[actual_col]
    df['market_error'] = df[market_col] - df[actual_col]

    # Compute absolute errors
    df['abs_model_error'] = df['model_error'].abs()
    df['abs_market_error'] = df['market_error'].abs()

    return df
