"""
Ball Knower Evaluation Metrics

Centralized utilities for model evaluation and backtest reporting.
All functions are deterministic and pure (no side effects or logging).

Functions:
- compute_mae: Mean Absolute Error
- compute_rmse: Root Mean Squared Error
- compute_ats_record: Against The Spread win/loss/push record
- compute_edge_and_ev: Edge and expected value metrics
- summarize_backtest_results: Comprehensive backtest summary
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE as a float

    Example:
        >>> y_true = np.array([3.0, -7.0, 1.0])
        >>> y_pred = np.array([2.5, -6.0, 1.5])
        >>> compute_mae(y_true, y_pred)
        0.6666666666666666
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE as a float

    Example:
        >>> y_true = np.array([3.0, -7.0, 1.0])
        >>> y_pred = np.array([2.5, -6.0, 1.5])
        >>> compute_rmse(y_true, y_pred)
        0.7637626158259734
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        return 0.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_ats_record(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    closing_spread: np.ndarray,
    push_margin: float = 0.5
) -> Dict[str, Union[int, float]]:
    """
    Compute Against The Spread (ATS) betting record.

    This function determines how often a model's prediction would result in a
    winning bet against the closing spread.

    Logic:
    - A "bet" is placed on the side where the model disagrees with the line
    - Win: Model correctly predicted which side would cover
    - Loss: Model incorrectly predicted which side would cover
    - Push: Actual result lands exactly on the spread (within push_margin)

    Args:
        y_true: Actual margins (home_score - away_score)
        y_pred: Predicted margins or spreads
        closing_spread: Vegas closing spread (home team perspective)
        push_margin: Margin within which a result is considered a push (default: 0.5)

    Returns:
        Dictionary with:
            - wins: Number of winning bets
            - losses: Number of losing bets
            - pushes: Number of pushes
            - win_pct: Win percentage (excluding pushes)
            - cover_pct: Cover percentage (wins / total games)

    Example:
        >>> actual = np.array([7, -3, 3])  # Home won by 7, lost by 3, won by 3
        >>> predicted = np.array([5, -5, 2])
        >>> spread = np.array([-3, -2, 4])  # Home favored by 3, 2; underdog by 4
        >>> compute_ats_record(actual, predicted, spread)
        {'wins': 2, 'losses': 1, 'pushes': 0, 'win_pct': 0.6666..., 'cover_pct': 0.6666...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    closing_spread = np.asarray(closing_spread)

    if not (len(y_true) == len(y_pred) == len(closing_spread)):
        raise ValueError("All arrays must have the same length")

    if len(y_true) == 0:
        return {
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'win_pct': 0.0,
            'cover_pct': 0.0
        }

    # Calculate actual result vs spread (positive = home covered)
    actual_vs_spread = y_true - closing_spread

    # Calculate model prediction vs spread (positive = model thinks home will cover)
    pred_vs_spread = y_pred - closing_spread

    # Determine outcomes
    wins = 0
    losses = 0
    pushes = 0

    for actual_diff, pred_diff in zip(actual_vs_spread, pred_vs_spread):
        # Check for push (actual result very close to spread)
        if abs(actual_diff) < push_margin:
            pushes += 1
            continue

        # Did the model correctly predict which side would cover?
        # If model and actual both positive or both negative, it's a win
        if (actual_diff > 0 and pred_diff > 0) or (actual_diff < 0 and pred_diff < 0):
            wins += 1
        else:
            losses += 1

    # Calculate percentages
    decided_games = wins + losses
    win_pct = (wins / decided_games) if decided_games > 0 else 0.0
    cover_pct = (wins / len(y_true)) if len(y_true) > 0 else 0.0

    return {
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_pct': win_pct,
        'cover_pct': cover_pct
    }


def compute_edge_and_ev(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    closing_spread: np.ndarray,
    vig: float = 0.05
) -> Dict[str, float]:
    """
    Compute edge and expected value metrics for betting analysis.

    Edge = |model_prediction - vegas_line|
    Simple flat-bet EV calculation based on historical hit rate.

    Args:
        y_true: Actual margins
        y_pred: Predicted margins/spreads
        closing_spread: Vegas closing spread
        vig: Vigorish/juice (default: 0.05 for -110 odds)

    Returns:
        Dictionary with:
            - mean_edge: Average absolute edge
            - median_edge: Median absolute edge
            - max_edge: Maximum absolute edge
            - mean_signed_edge: Average signed edge (model - vegas)
            - flat_bet_ev: Simple flat-bet expected value estimate
            - roi: Return on investment percentage

    Example:
        >>> actual = np.array([7, -3, 3, -10])
        >>> predicted = np.array([5, -5, 2, -8])
        >>> spread = np.array([-3, -2, 4, -5])
        >>> ev = compute_edge_and_ev(actual, predicted, spread)
        >>> ev['mean_edge'] > 0
        True
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    closing_spread = np.asarray(closing_spread)

    if not (len(y_true) == len(y_pred) == len(closing_spread)):
        raise ValueError("All arrays must have the same length")

    if len(y_true) == 0:
        return {
            'mean_edge': 0.0,
            'median_edge': 0.0,
            'max_edge': 0.0,
            'mean_signed_edge': 0.0,
            'flat_bet_ev': 0.0,
            'roi': 0.0
        }

    # Calculate edges
    signed_edge = y_pred - closing_spread
    abs_edge = np.abs(signed_edge)

    # Calculate ATS record to determine hit rate
    ats = compute_ats_record(y_true, y_pred, closing_spread)
    hit_rate = ats['win_pct']

    # Simple flat-bet EV calculation
    # EV = (hit_rate * win_amount) - ((1 - hit_rate) * loss_amount)
    # For -110 odds: win $100 on $110 bet, lose $110
    # Normalized to $1 bet: win $0.909, lose $1
    win_multiplier = 1.0 / (1.0 + vig)  # ~0.909 for -110
    loss_amount = 1.0

    flat_bet_ev = (hit_rate * win_multiplier) - ((1 - hit_rate) * loss_amount)
    roi = flat_bet_ev * 100  # As percentage

    return {
        'mean_edge': float(np.mean(abs_edge)),
        'median_edge': float(np.median(abs_edge)),
        'max_edge': float(np.max(abs_edge)),
        'mean_signed_edge': float(np.mean(signed_edge)),
        'flat_bet_ev': float(flat_bet_ev),
        'roi': float(roi)
    }


def summarize_backtest_results(
    df: pd.DataFrame,
    actual_col: str = 'actual_margin',
    pred_col: str = 'model_line',
    spread_col: str = 'closing_spread',
    model_version: Optional[str] = None
) -> Dict[str, Union[int, float, str]]:
    """
    Generate comprehensive backtest summary from a DataFrame.

    Args:
        df: DataFrame containing backtest results
        actual_col: Column name for actual margins
        pred_col: Column name for model predictions
        spread_col: Column name for closing spreads
        model_version: Optional model version identifier

    Returns:
        Dictionary with comprehensive metrics:
            - model_version: Model identifier
            - n_games: Number of games evaluated
            - n_bets: Number of bets (if applicable)
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - ats_wins: ATS wins
            - ats_losses: ATS losses
            - ats_pushes: ATS pushes
            - ats_win_pct: ATS win percentage
            - mean_edge: Average edge vs Vegas
            - median_edge: Median edge vs Vegas
            - roi: Return on investment percentage

    Example:
        >>> df = pd.DataFrame({
        ...     'actual_margin': [7, -3, 3],
        ...     'model_line': [5, -5, 2],
        ...     'closing_spread': [-3, -2, 4]
        ... })
        >>> summary = summarize_backtest_results(df, model_version='v1.2')
        >>> summary['n_games']
        3
    """
    if df is None or len(df) == 0:
        return {
            'model_version': model_version or 'unknown',
            'n_games': 0,
            'n_bets': 0,
            'mae': 0.0,
            'rmse': 0.0,
            'ats_wins': 0,
            'ats_losses': 0,
            'ats_pushes': 0,
            'ats_win_pct': 0.0,
            'mean_edge': 0.0,
            'median_edge': 0.0,
            'roi': 0.0
        }

    # Extract arrays
    y_true = df[actual_col].values
    y_pred = df[pred_col].values
    closing_spread = df[spread_col].values

    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(closing_spread))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    closing_spread = closing_spread[mask]

    n_games = len(y_true)

    # If there's a 'bet' or 'edge_threshold' column, count bets
    n_bets = n_games
    if 'bet' in df.columns:
        n_bets = df['bet'].sum()
    elif 'abs_edge' in df.columns and 'edge_threshold' in df.columns:
        n_bets = (df['abs_edge'] >= df['edge_threshold']).sum()

    # Compute core metrics
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)

    # Compute ATS record
    ats = compute_ats_record(y_true, y_pred, closing_spread)

    # Compute edge and EV
    edge_ev = compute_edge_and_ev(y_true, y_pred, closing_spread)

    # Assemble summary
    summary = {
        'model_version': model_version or 'unknown',
        'n_games': int(n_games),
        'n_bets': int(n_bets),
        'mae': float(mae),
        'rmse': float(rmse),
        'ats_wins': int(ats['wins']),
        'ats_losses': int(ats['losses']),
        'ats_pushes': int(ats['pushes']),
        'ats_win_pct': float(ats['win_pct']),
        'mean_edge': float(edge_ev['mean_edge']),
        'median_edge': float(edge_ev['median_edge']),
        'roi': float(edge_ev['roi'])
    }

    return summary
