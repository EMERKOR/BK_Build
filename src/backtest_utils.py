"""
Backtest Utilities - Shared logic for v1.0 and v1.2 backtesting

This module provides reusable functions for:
- Loading and preprocessing historical nfelo data
- Generating predictions for different model versions
- Computing season-level performance metrics
- Calculating ATS records and ROI
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


# ============================================================================
# DATA LOADING
# ============================================================================

def load_nfelo_historical_data(min_season: Optional[int] = None,
                                max_season: Optional[int] = None) -> pd.DataFrame:
    """
    Load and preprocess historical nfelo games data.

    Args:
        min_season: Minimum season to include (inclusive)
        max_season: Maximum season to include (inclusive)

    Returns:
        DataFrame with parsed game_id fields and complete data only
    """
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(nfelo_url)

    # Extract season/week/teams from game_id
    df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(
        r'(\d{4})_(\d+)_(\w+)_(\w+)'
    )
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to complete data (has closing line and ELO ratings)
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()

    # Filter by season range if specified
    if min_season is not None:
        df = df[df['season'] >= min_season].copy()
    if max_season is not None:
        df = df[df['season'] <= max_season].copy()

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_v1_2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features needed for v1.2 model.

    Args:
        df: DataFrame with nfelo data

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Primary feature: ELO differential
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Situational adjustments
    df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
    df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
    df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

    # QB adjustments
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

    # Vegas line
    df['vegas_line'] = df['home_line_close']

    # Remove rows with missing features
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                   'surface_mod', 'time_advantage', 'qb_diff']
    mask = df[feature_cols + ['vegas_line']].notna().all(axis=1)
    df = df[mask].copy()

    return df


def engineer_v1_0_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for v1.0 model (just nfelo_diff).

    Args:
        df: DataFrame with nfelo data

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Only feature: ELO differential
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    df['vegas_line'] = df['home_line_close']

    # Remove rows with missing data
    mask = df[['nfelo_diff', 'vegas_line']].notna().all(axis=1)
    df = df[mask].copy()

    return df


# ============================================================================
# PREDICTIONS
# ============================================================================

def generate_v1_0_predictions(df: pd.DataFrame,
                              intercept: float = 2.67,
                              nfelo_coef: float = 0.0447) -> pd.DataFrame:
    """
    Generate v1.0 predictions (simple nfelo-based model).

    Args:
        df: DataFrame with engineered features
        intercept: Model intercept
        nfelo_coef: Coefficient for nfelo_diff

    Returns:
        DataFrame with predictions and edge calculations
    """
    df = df.copy()

    # Generate predictions
    df['bk_prediction'] = intercept + (df['nfelo_diff'] * nfelo_coef)

    # Calculate edge
    df['edge'] = df['bk_prediction'] - df['vegas_line']
    df['abs_edge'] = df['edge'].abs()

    return df


def generate_v1_2_predictions(df: pd.DataFrame, model_params: Dict) -> pd.DataFrame:
    """
    Generate v1.2 predictions using calibrated model weights.

    Args:
        df: DataFrame with engineered features
        model_params: Dictionary with 'intercept' and 'coefficients'

    Returns:
        DataFrame with predictions and edge calculations
    """
    df = df.copy()

    intercept = model_params['intercept']
    coefs = model_params['coefficients']

    # Generate predictions
    df['bk_prediction'] = intercept + \
        (df['nfelo_diff'] * coefs['nfelo_diff']) + \
        (df['rest_advantage'] * coefs['rest_advantage']) + \
        (df['div_game'] * coefs['div_game']) + \
        (df['surface_mod'] * coefs['surface_mod']) + \
        (df['time_advantage'] * coefs['time_advantage']) + \
        (df['qb_diff'] * coefs['qb_diff'])

    # Calculate edge
    df['edge'] = df['bk_prediction'] - df['vegas_line']
    df['abs_edge'] = df['edge'].abs()

    return df


# ============================================================================
# ATS CALCULATIONS
# ============================================================================

def calculate_ats_result(row: pd.Series) -> str:
    """
    Calculate ATS result for a single game.

    Assumes we bet on the side with the edge.
    - If edge is negative (model favors home more than Vegas), we bet home.
    - If edge is positive (model favors away more than Vegas), we bet away.

    Args:
        row: Game row with 'edge', 'home_score', 'away_score', 'vegas_line'

    Returns:
        'win', 'loss', or 'push'
    """
    # Check if we have actual scores
    if pd.isna(row.get('home_score')) or pd.isna(row.get('away_score')):
        return 'unknown'

    home_score = row['home_score']
    away_score = row['away_score']
    vegas_line = row['vegas_line']
    edge = row['edge']

    # Actual margin (home perspective)
    actual_margin = home_score - away_score

    # Determine which side we bet based on edge
    if edge < 0:
        # We bet home (model favors home more)
        # Home covers if: actual_margin > vegas_line (more conservative: >=)
        ats_margin = actual_margin - vegas_line
    else:
        # We bet away (model favors away more)
        # Away covers if: actual_margin < vegas_line (more conservative: <=)
        ats_margin = vegas_line - actual_margin

    # Check result
    if abs(ats_margin) < 0.5:  # Push (within 0.5 points)
        return 'push'
    elif ats_margin > 0:
        return 'win'
    else:
        return 'loss'


def calculate_flat_roi(ats_wins: int, ats_losses: int, ats_pushes: int,
                       juice: float = -110) -> float:
    """
    Calculate flat-stake ROI assuming standard juice.

    Args:
        ats_wins: Number of ATS wins
        ats_losses: Number of ATS losses
        ats_pushes: Number of pushes (returned)
        juice: American odds (default -110)

    Returns:
        ROI as decimal (e.g., 0.05 = 5% ROI)
    """
    # Total wagers (pushes are returned, don't count)
    total_wagers = ats_wins + ats_losses

    if total_wagers == 0:
        return 0.0

    # At -110: risk $110 to win $100
    # Win: +$100, Loss: -$110
    if juice == -110:
        profit = (ats_wins * 100) - (ats_losses * 110)
        total_risk = total_wagers * 110
    else:
        # General case (not needed for now, but keeping for extensibility)
        from src.betting_utils import american_to_implied_prob

        if juice < 0:
            win_amount = 100 / (-juice / 100)
            risk_amount = 100
        else:
            win_amount = juice
            risk_amount = 100

        profit = (ats_wins * win_amount) - (ats_losses * risk_amount)
        total_risk = total_wagers * risk_amount

    return profit / total_risk if total_risk > 0 else 0.0


# ============================================================================
# SEASON METRICS
# ============================================================================

def compute_season_metrics(season_df: pd.DataFrame,
                          model_name: str,
                          edge_threshold: float = 0.0) -> Dict:
    """
    Compute performance metrics for a single season.

    Args:
        season_df: DataFrame for games in this season
        model_name: Name of model (e.g., 'v1.0', 'v1.2')
        edge_threshold: Minimum absolute edge to consider a "bet"

    Returns:
        Dictionary with season metrics
    """
    # Filter to bets meeting threshold
    bets_df = season_df[season_df['abs_edge'] >= edge_threshold].copy()

    # Basic counts
    n_games = len(season_df)
    n_bets = len(bets_df)

    # MAE vs Vegas
    mae_vs_vegas = season_df['abs_edge'].mean()

    # ATS results (only for bets)
    if n_bets > 0 and 'ats_result' in bets_df.columns:
        ats_wins = (bets_df['ats_result'] == 'win').sum()
        ats_losses = (bets_df['ats_result'] == 'loss').sum()
        ats_pushes = (bets_df['ats_result'] == 'push').sum()

        # Calculate flat ROI
        flat_roi = calculate_flat_roi(ats_wins, ats_losses, ats_pushes)
    else:
        ats_wins = 0
        ats_losses = 0
        ats_pushes = 0
        flat_roi = 0.0

    return {
        'n_games': n_games,
        'n_bets': n_bets,
        'mae_vs_vegas': mae_vs_vegas,
        'ats_wins': ats_wins,
        'ats_losses': ats_losses,
        'ats_pushes': ats_pushes,
        'flat_roi': flat_roi,
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_v1_2_model(model_file: Path) -> Dict:
    """
    Load v1.2 model parameters from JSON file.

    Args:
        model_file: Path to model JSON file

    Returns:
        Dictionary with model parameters
    """
    with open(model_file, 'r') as f:
        return json.load(f)
