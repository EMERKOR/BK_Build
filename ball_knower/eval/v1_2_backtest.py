"""
Ball Knower v1.2 Backtest Module

Evaluates v1.2 model against actual game outcomes and tests edge betting strategies.

This module provides:
- Dataset loading with proper feature engineering
- Actual game outcome merging
- Model loading and prediction
- Replication metrics (how well we match Vegas)
- Edge betting strategy evaluation (betting when |edge| >= threshold)
"""

from pathlib import Path
from typing import Optional, Union
import warnings

import pandas as pd
import numpy as np
import json


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_v1_2_dataset(
    start_season: Optional[int] = None,
    end_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the canonical v1.2 dataset with all training features.

    This replicates the data loading and feature engineering from the
    training script (ball_knower_v1_2.py).

    Args:
        start_season: Optional filter for minimum season (inclusive)
        end_season: Optional filter for maximum season (inclusive)

    Returns:
        DataFrame with columns:
            - game_id, season, week, home_team, away_team
            - vegas_line (home_line_close)
            - Feature columns: nfelo_diff, rest_advantage, div_game,
              surface_mod, time_advantage, qb_diff
            - Raw nfelo columns for outcome merging later
    """
    print("Loading nfelo historical data...")

    # Load nfelo historical games
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(nfelo_url)

    # Extract season/week/teams from game_id
    df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(
        r'(\d{4})_(\d+)_(\w+)_(\w+)'
    )
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter by season if specified
    if start_season is not None:
        df = df[df['season'] >= start_season]
    if end_season is not None:
        df = df[df['season'] <= end_season]

    # Filter to games with complete data
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()

    print(f"  Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

    # ========================================================================
    # FEATURE ENGINEERING (matches training script exactly)
    # ========================================================================

    print("Engineering features...")

    # Primary feature: ELO differential
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Situational adjustments from nfelo
    df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
    df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
    df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

    # QB adjustments
    df['qb_diff'] = (
        df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0)
    )

    # Target: Vegas closing line
    df['vegas_line'] = df['home_line_close']

    # Feature set (must match training!)
    feature_cols = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff'
    ]

    # Remove rows with NaN in features or target
    mask = df[feature_cols + ['vegas_line']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)

    print(f"  Engineered {len(feature_cols)} features for {len(df):,} games")

    return df


# ============================================================================
# GAME OUTCOMES
# ============================================================================

def merge_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge actual game outcomes into the dataset.

    Loads actual game scores from schedules.parquet and merges with
    the nfelo feature dataset.

    Args:
        df: DataFrame from load_v1_2_dataset()

    Returns:
        DataFrame with additional columns:
            - home_points: Home team final score
            - away_points: Away team final score
            - home_margin: home_points - away_points (positive = home win)
            - home_covers_closing: 1 if home covered closing spread, 0 if not, 0.5 if push
            - away_covers_closing: 1 if away covered closing spread, 0 if not, 0.5 if push

    Raises:
        FileNotFoundError: If schedules.parquet is not found
    """
    print("Merging game outcomes...")

    # Load actual game results from local schedules.parquet
    project_root = Path(__file__).resolve().parents[2]
    schedules_path = project_root / 'schedules.parquet'

    if not schedules_path.exists():
        raise FileNotFoundError(
            f"schedules.parquet not found at {schedules_path}\n"
            f"This file is required to get actual game outcomes."
        )

    schedules = pd.read_parquet(schedules_path)

    # Filter to regular season games only
    schedules = schedules[schedules['game_type'] == 'REG'].copy()

    # Select relevant columns
    schedules = schedules[['season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']]

    # Merge with nfelo data
    df = df.copy()
    df = df.merge(
        schedules,
        on=['season', 'week', 'home_team', 'away_team'],
        how='left'
    )

    # Standardize column names
    df['home_points'] = df['home_score']
    df['away_points'] = df['away_score']

    # Calculate margin (positive = home team won)
    df['home_margin'] = df['home_points'] - df['away_points']

    # Did home team cover the closing spread?
    # Spread convention: negative = home favored
    # Example: spread = -3.5, home wins by 7, margin = 7, home covers (7 > 3.5)
    # Example: spread = -3.5, home wins by 3, margin = 3, home doesn't cover (3 < 3.5)
    # home_covers = (margin + spread) > 0

    df['spread_result'] = df['home_margin'] + df['vegas_line']

    # Handle pushes (spread_result == 0)
    df['home_covers_closing'] = 0.0
    df.loc[df['spread_result'] > 0, 'home_covers_closing'] = 1.0
    df.loc[df['spread_result'] == 0, 'home_covers_closing'] = 0.5  # Push

    df['away_covers_closing'] = 0.0
    df.loc[df['spread_result'] < 0, 'away_covers_closing'] = 1.0
    df.loc[df['spread_result'] == 0, 'away_covers_closing'] = 0.5  # Push

    # Count games with outcomes
    games_with_outcomes = df[df['home_points'].notna()].shape[0]
    print(f"  Merged outcomes for {games_with_outcomes:,} games")

    return df


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_v1_2_model(
    model_path: Optional[Union[Path, str]] = None
) -> dict:
    """
    Load the trained v1.2 model parameters.

    Args:
        model_path: Optional path to model JSON file.
                   Defaults to output/ball_knower_v1_2_model.json

    Returns:
        Dictionary with keys:
            - intercept: Model intercept
            - coefficients: Dict of feature -> coefficient
            - alpha: Ridge regularization parameter
            - train_mae, test_mae, train_r2, test_r2: Performance metrics

    Raises:
        FileNotFoundError: If model file doesn't exist with helpful message
    """
    if model_path is None:
        # Default path matches the training script
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / 'output' / 'ball_knower_v1_2_model.json'
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n\n"
            f"Please train the model first by running:\n"
            f"  python ball_knower_v1_2.py\n\n"
            f"This will create the model file at {model_path}"
        )

    with open(model_path, 'r') as f:
        model_params = json.load(f)

    print(f"Loaded model from {model_path}")
    print(f"  Train MAE: {model_params.get('train_mae', 'N/A'):.2f}")
    print(f"  Test MAE:  {model_params.get('test_mae', 'N/A'):.2f}")

    return model_params


# ============================================================================
# PREDICTIONS
# ============================================================================

def add_v1_2_predictions(
    df: pd.DataFrame,
    model: dict
) -> pd.DataFrame:
    """
    Add v1.2 model predictions to the dataset.

    Args:
        df: DataFrame with v1.2 features (from load_v1_2_dataset)
        model: Model parameters from load_trained_v1_2_model()

    Returns:
        DataFrame with additional columns:
            - bk_line: Ball Knower predicted home spread
            - bk_residual_vs_vegas: bk_line - vegas_line (our edge)
    """
    print("Generating v1.2 predictions...")

    df = df.copy()

    # Extract model parameters
    intercept = model['intercept']
    coefs = model['coefficients']

    # Generate predictions (must match training feature order!)
    df['bk_line'] = (
        intercept +
        (df['nfelo_diff'] * coefs['nfelo_diff']) +
        (df['rest_advantage'] * coefs['rest_advantage']) +
        (df['div_game'] * coefs['div_game']) +
        (df['surface_mod'] * coefs['surface_mod']) +
        (df['time_advantage'] * coefs['time_advantage']) +
        (df['qb_diff'] * coefs['qb_diff'])
    )

    # Calculate residual vs Vegas
    df['bk_residual_vs_vegas'] = df['bk_line'] - df['vegas_line']

    print(f"  Generated predictions for {len(df):,} games")

    return df


# ============================================================================
# REPLICATION METRICS
# ============================================================================

def compute_replication_metrics(df: pd.DataFrame) -> dict:
    """
    Compute how well Ball Knower replicates Vegas spreads.

    Args:
        df: DataFrame with 'bk_line', 'vegas_line', 'bk_residual_vs_vegas'

    Returns:
        Dictionary with:
            - mae: Mean absolute error vs Vegas
            - rmse: Root mean squared error vs Vegas
            - mean_residual: Mean of bk_residual_vs_vegas (should be ~0 if calibrated)
            - std_residual: Std of residuals
            - median_abs_residual: Median absolute residual
            - pct_within_1pt: Percentage within 1 point of Vegas
            - pct_within_2pt: Percentage within 2 points of Vegas
            - pct_within_3pt: Percentage within 3 points of Vegas
    """
    residuals = df['bk_residual_vs_vegas']
    abs_residuals = residuals.abs()

    metrics = {
        'mae': float(abs_residuals.mean()),
        'rmse': float(np.sqrt((residuals ** 2).mean())),
        'mean_residual': float(residuals.mean()),
        'std_residual': float(residuals.std()),
        'median_abs_residual': float(abs_residuals.median()),
        'pct_within_1pt': float((abs_residuals <= 1.0).mean() * 100),
        'pct_within_2pt': float((abs_residuals <= 2.0).mean() * 100),
        'pct_within_3pt': float((abs_residuals <= 3.0).mean() * 100),
    }

    return metrics


# ============================================================================
# EDGE BETTING STRATEGY
# ============================================================================

def compute_edge_betting_metrics(
    df: pd.DataFrame,
    edge_thresholds: list[float]
) -> pd.DataFrame:
    """
    Evaluate naive edge betting strategy at different thresholds.

    Betting rules:
        - home_edge = bk_line - vegas_line
        - If home_edge >= threshold: bet HOME
        - If home_edge <= -threshold: bet AWAY
        - Otherwise: no bet

    For each threshold, compute:
        - Number of bets placed
        - Win rate vs the closing spread
        - Push rate
        - Units won (assuming -110 odds, 1 unit per bet)
        - ROI percentage

    Args:
        df: DataFrame with outcomes and predictions
        edge_thresholds: List of edge thresholds to test (e.g., [1.0, 2.0, 3.0])

    Returns:
        DataFrame with one row per threshold, columns:
            - threshold
            - num_bets
            - win_rate (excluding pushes)
            - push_rate
            - units_won (at -110 odds)
            - roi_pct
    """
    results = []

    # Filter to games with actual outcomes
    df_complete = df[df['home_covers_closing'].notna()].copy()

    # Edge convention (documented clearly):
    # home_edge = bk_line - vegas_line
    # Positive home_edge means we think home team is undervalued (Vegas line is too high)
    # Example: vegas_line = -3, bk_line = -5 -> home_edge = -2 (we like AWAY)
    # Example: vegas_line = -3, bk_line = -1 -> home_edge = +2 (we like HOME)
    df_complete['home_edge'] = df_complete['bk_residual_vs_vegas']

    for threshold in edge_thresholds:
        # Identify bets
        bet_home_mask = df_complete['home_edge'] >= threshold
        bet_away_mask = df_complete['home_edge'] <= -threshold

        # Get bet outcomes
        home_bets = df_complete[bet_home_mask]
        away_bets = df_complete[bet_away_mask]

        # Count wins/pushes for home bets
        home_wins = home_bets['home_covers_closing'].sum()
        home_pushes = (home_bets['home_covers_closing'] == 0.5).sum()
        home_losses = len(home_bets) - home_wins - home_pushes

        # Count wins/pushes for away bets
        away_wins = away_bets['away_covers_closing'].sum()
        away_pushes = (away_bets['away_covers_closing'] == 0.5).sum()
        away_losses = len(away_bets) - away_wins - away_pushes

        # Total stats
        total_bets = len(home_bets) + len(away_bets)
        total_wins = home_wins + away_wins
        total_pushes = home_pushes + away_pushes
        total_losses = home_losses + away_losses

        if total_bets == 0:
            results.append({
                'threshold': threshold,
                'num_bets': 0,
                'win_rate': 0.0,
                'push_rate': 0.0,
                'units_won': 0.0,
                'roi_pct': 0.0,
            })
            continue

        # Win rate (excluding pushes)
        decided_bets = total_bets - total_pushes
        win_rate = total_wins / decided_bets if decided_bets > 0 else 0.0
        push_rate = total_pushes / total_bets

        # Units won at -110 odds
        # Win: +0.909 units (risk 1.1 to win 1)
        # Loss: -1.0 units
        # Push: 0.0 units
        units_won = (total_wins * (10/11)) - total_losses
        roi_pct = (units_won / total_bets * 100) if total_bets > 0 else 0.0

        results.append({
            'threshold': threshold,
            'num_bets': total_bets,
            'win_rate': win_rate,
            'push_rate': push_rate,
            'units_won': units_won,
            'roi_pct': roi_pct,
        })

    return pd.DataFrame(results)
