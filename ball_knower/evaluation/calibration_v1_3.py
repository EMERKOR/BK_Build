"""
v1.3 Model Calibration Utilities

Computes calibration parameters from backtest results:
- Global bias (mean error)
- Slope/intercept via linear regression
- ATS win rates by edge bucket

These parameters can be used to:
1. Adjust raw v1.3 predictions (bias correction)
2. Understand model performance by edge magnitude
3. Inform bet sizing and bankroll management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def compute_v1_3_calibration(
    backtest_df: pd.DataFrame,
    edge_bins: List[float] = None
) -> Dict:
    """
    Compute calibration parameters for v1.3 model from backtest results.

    Args:
        backtest_df: DataFrame with backtest results (output from run_backtest_v1_3)
                    Must contain columns: season, model, n_games, mae_vs_vegas,
                    rmse_vs_vegas, mean_edge
        edge_bins: Edge thresholds for binning (default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    Returns:
        Dictionary with calibration parameters:
            {
                "mean_error": float,  # Average prediction error (bias)
                "mae": float,  # Mean absolute error
                "rmse": float,  # Root mean squared error
                "edge_bins": list,  # Edge bin thresholds
                "n_seasons": int,  # Number of seasons in calibration
                "n_games_total": int,  # Total games across all seasons
            }

    Note:
        This function operates on season-aggregated backtest results.
        For game-level calibration (e.g., actual vs predicted), you would
        need the full backtest DataFrame with game-level predictions.
    """
    if edge_bins is None:
        edge_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Validate input
    required_cols = ['season', 'model', 'n_games', 'mae_vs_vegas', 'rmse_vs_vegas', 'mean_edge']
    missing_cols = [col for col in required_cols if col not in backtest_df.columns]
    if missing_cols:
        raise ValueError(f"Backtest DataFrame missing required columns: {missing_cols}")

    # Filter to v1.3 rows only
    df = backtest_df[backtest_df['model'] == 'v1.3'].copy()

    if len(df) == 0:
        raise ValueError("No v1.3 model results found in backtest DataFrame")

    # Compute overall statistics
    n_seasons = len(df)
    n_games_total = int(df['n_games'].sum())

    # Weighted average MAE and RMSE (weighted by number of games per season)
    weighted_mae = (df['mae_vs_vegas'] * df['n_games']).sum() / n_games_total
    weighted_rmse = np.sqrt((df['rmse_vs_vegas']**2 * df['n_games']).sum() / n_games_total)

    # Mean error (average bias across seasons)
    # Note: This is the average of mean_edge, which may not be the true bias
    # without game-level data. It's a proxy for overall directional bias.
    mean_error = df['mean_edge'].mean()

    calibration = {
        "mean_error": float(mean_error),
        "mae": float(weighted_mae),
        "rmse": float(weighted_rmse),
        "edge_bins": edge_bins,
        "n_seasons": int(n_seasons),
        "n_games_total": int(n_games_total),
        "calibration_seasons": f"{int(df['season'].min())}-{int(df['season'].max())}",
        "model_version": "v1.3"
    }

    return calibration


def compute_game_level_calibration(
    game_predictions_df: pd.DataFrame,
    edge_bins: List[float] = None
) -> Dict:
    """
    Compute detailed game-level calibration from full backtest predictions.

    Args:
        game_predictions_df: DataFrame with game-level predictions
            Required columns:
                - game_id
                - season
                - week
                - bk_v1_3_spread (model prediction)
                - vegas_line (closing spread)
                - edge (bk_v1_3_spread - vegas_line)
                - actual_margin (optional, for ATS tracking)
        edge_bins: Edge thresholds for binning

    Returns:
        Dictionary with detailed calibration:
            {
                "mean_error": float,
                "slope": float,  # Linear regression slope
                "intercept": float,  # Linear regression intercept
                "edge_bins": list,
                "ats_win_rates": list,  # Win rate for each edge bin
                "n_games_per_bin": list,  # Game count for each bin
            }

    Note:
        This function requires game-level predictions, not season aggregates.
        It provides more detailed calibration than compute_v1_3_calibration.
    """
    if edge_bins is None:
        edge_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Validate input
    required_cols = ['bk_v1_3_spread', 'vegas_line', 'edge']
    missing_cols = [col for col in required_cols if col not in game_predictions_df.columns]
    if missing_cols:
        raise ValueError(f"Game predictions DataFrame missing required columns: {missing_cols}")

    df = game_predictions_df.copy()

    # Compute mean error
    mean_error = df['edge'].mean()

    # Linear regression: vegas_line ~ bk_v1_3_spread
    # This tells us how well our predictions track Vegas
    X = df['bk_v1_3_spread'].values.reshape(-1, 1)
    y = df['vegas_line'].values

    # Simple linear regression using numpy
    # y = slope * X + intercept
    coeffs = np.polyfit(X.flatten(), y, deg=1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Compute ATS win rates by edge bin
    df['abs_edge'] = df['edge'].abs()

    ats_win_rates = []
    n_games_per_bin = []

    for i, edge_threshold in enumerate(edge_bins):
        if i == 0:
            # First bin: 0 to edge_threshold
            bin_df = df[df['abs_edge'] < edge_threshold]
        else:
            # Subsequent bins: previous threshold to current threshold
            prev_threshold = edge_bins[i - 1]
            bin_df = df[(df['abs_edge'] >= prev_threshold) & (df['abs_edge'] < edge_threshold)]

        n_games = len(bin_df)
        n_games_per_bin.append(n_games)

        if n_games > 0 and 'actual_margin' in df.columns:
            # Calculate ATS win rate
            # Positive edge means we predict home team to cover more than Vegas
            # Win if: (actual_margin > vegas_line and edge > 0) or (actual_margin < vegas_line and edge < 0)
            bin_df['ats_win'] = (
                ((bin_df['actual_margin'] > bin_df['vegas_line']) & (bin_df['edge'] > 0)) |
                ((bin_df['actual_margin'] < bin_df['vegas_line']) & (bin_df['edge'] < 0))
            )
            ats_win_rate = bin_df['ats_win'].mean()
            ats_win_rates.append(float(ats_win_rate))
        else:
            # No actual margins available or no games in bin
            ats_win_rates.append(None)

    # Last bin: >= largest threshold
    last_bin_df = df[df['abs_edge'] >= edge_bins[-1]]
    n_games_per_bin.append(len(last_bin_df))
    if len(last_bin_df) > 0 and 'actual_margin' in df.columns:
        last_bin_df['ats_win'] = (
            ((last_bin_df['actual_margin'] > last_bin_df['vegas_line']) & (last_bin_df['edge'] > 0)) |
            ((last_bin_df['actual_margin'] < last_bin_df['vegas_line']) & (last_bin_df['edge'] < 0))
        )
        ats_win_rates.append(float(last_bin_df['ats_win'].mean()))
    else:
        ats_win_rates.append(None)

    calibration = {
        "mean_error": float(mean_error),
        "slope": float(slope),
        "intercept": float(intercept),
        "edge_bins": edge_bins,
        "ats_win_rates": ats_win_rates,
        "n_games_per_bin": n_games_per_bin,
        "n_games_total": len(df),
        "model_version": "v1.3"
    }

    return calibration


def apply_bias_correction(
    predictions: pd.DataFrame,
    mean_error: float
) -> pd.DataFrame:
    """
    Apply bias correction to v1.3 predictions.

    Args:
        predictions: DataFrame with 'bk_v1_3_spread' column
        mean_error: Mean error from calibration

    Returns:
        DataFrame with added 'bk_v1_3_spread_corrected' column

    Note:
        Bias correction subtracts the mean error from predictions.
        If mean_error is positive (model predicts too high), correction lowers predictions.
    """
    df = predictions.copy()

    if 'bk_v1_3_spread' not in df.columns:
        raise ValueError("Predictions DataFrame must contain 'bk_v1_3_spread' column")

    df['bk_v1_3_spread_corrected'] = df['bk_v1_3_spread'] - mean_error

    return df
