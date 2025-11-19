"""
Ball Knower v1.x Model Comparison Harness

Compares v1.0 and v1.2 models on:
1. Accuracy metrics (MAE, hit rates)
2. ATS (Against The Spread) betting performance
3. PnL simulation with configurable edge thresholds

Usage:
    from ball_knower.benchmarks.v1_comparison import compare_v1_models

    results = compare_v1_models(
        test_seasons=[2020, 2021, 2022],
        edge_threshold=1.5
    )
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src import config


# ============================================================================
# ATS & PNL SIMULATION
# ============================================================================

def simulate_ats_pnl(
    df: pd.DataFrame,
    model_spread_col: str,
    market_spread_col: str = "vegas_closing_spread",
    actual_margin_col: str = "actual_margin",
    edge_threshold: float = 1.0,
    odds: float = -110.0,
) -> Dict[str, Any]:
    """
    Simulate fixed-stake ATS betting for a model vs the market.

    Betting Logic:
    --------------
    - Edge = model_spread - market_spread (home team perspective)
    - If model_spread < market_spread: model thinks home is STRONGER
      → Bet on home team to cover (home - market_spread)
    - If model_spread > market_spread: model thinks home is WEAKER
      → Bet on away team to cover (away + market_spread)

    Only bet when |edge| >= edge_threshold.

    Odds Convention:
    ---------------
    Standard -110 both sides means:
    - Win: +1.0 unit (risk 1.1 to win 1.0)
    - Loss: -1.1 units
    - Push: 0 units

    Args:
        df: DataFrame with predictions and outcomes
        model_spread_col: Column name for model's spread prediction
        market_spread_col: Column name for Vegas closing spread
        actual_margin_col: Column name for actual game margin
        edge_threshold: Minimum |edge| required to place bet (in points)
        odds: American odds (default -110)

    Returns:
        Dictionary with:
            - n_games: Total games in dataset
            - n_bets: Number of bets placed
            - win_rate: Percentage of bets won (excluding pushes)
            - units_won: Net profit in units
            - roi: Return on investment (units_won / total_risked)
            - avg_edge: Average edge for bets placed
            - edge_threshold: The threshold used
            - wins: Number of winning bets
            - losses: Number of losing bets
            - pushes: Number of push bets
    """
    # Handle empty DataFrame
    if len(df) == 0:
        return {
            'n_games': 0,
            'n_bets': 0,
            'win_rate': None,
            'units_won': 0.0,
            'roi': None,
            'avg_edge': None,
            'edge_threshold': edge_threshold,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
        }

    # Filter to rows with complete data
    required_cols = [model_spread_col, market_spread_col, actual_margin_col]
    mask = df[required_cols].notna().all(axis=1)
    clean_df = df[mask].copy()

    if len(clean_df) == 0:
        return {
            'n_games': 0,
            'n_bets': 0,
            'win_rate': None,
            'units_won': 0.0,
            'roi': None,
            'avg_edge': None,
            'edge_threshold': edge_threshold,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
        }

    # Calculate edge
    clean_df['edge'] = clean_df[model_spread_col] - clean_df[market_spread_col]
    clean_df['abs_edge'] = clean_df['edge'].abs()

    # Filter to bets (where |edge| >= threshold)
    bets_df = clean_df[clean_df['abs_edge'] >= edge_threshold].copy()

    if len(bets_df) == 0:
        return {
            'n_games': len(clean_df),
            'n_bets': 0,
            'win_rate': None,
            'units_won': 0.0,
            'roi': None,
            'avg_edge': None,
            'edge_threshold': edge_threshold,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
        }

    # Determine bet outcome for each bet
    # Betting convention:
    # - If edge < 0: model likes home more than market → bet home -market_spread
    # - If edge > 0: model likes away more than market → bet away +market_spread

    def evaluate_bet(row):
        """
        Evaluate if a bet wins, loses, or pushes.

        Betting Logic:
        - If edge < 0: model_spread < market_spread
          → Model thinks home will win by MORE than market expects
          → Bet on home to cover market_spread
          → Win if actual_margin < market_spread (more negative = bigger home win)

        - If edge > 0: model_spread > market_spread
          → Model thinks home will win by LESS than market expects
          → Bet on away to cover (+market_spread points)
          → Win if actual_margin > market_spread (less negative/more positive = smaller home win or away win)

        Returns: 'win', 'loss', or 'push'
        """
        edge = row['edge']
        market_spread = row[market_spread_col]
        actual_margin = row[actual_margin_col]

        if edge < 0:
            # Bet on home to cover market_spread
            # For negative spreads: home covers if actual < spread
            # Example: market=-5, actual=-7 → home covered (won by 7 vs needed 5)
            if actual_margin < market_spread:
                return 'win'
            elif actual_margin == market_spread:
                return 'push'
            else:
                return 'loss'
        else:
            # Bet on away to cover market_spread
            # Away gets +market_spread points, covers if actual > spread
            # Example: market=-5 (home fav), actual=-3 → away covered (lost by 3, got 5 pts)
            if actual_margin > market_spread:
                return 'win'
            elif actual_margin == market_spread:
                return 'push'
            else:
                return 'loss'

    bets_df['outcome'] = bets_df.apply(evaluate_bet, axis=1)

    # Count outcomes
    wins = (bets_df['outcome'] == 'win').sum()
    losses = (bets_df['outcome'] == 'loss').sum()
    pushes = (bets_df['outcome'] == 'push').sum()

    # Calculate PnL
    # Win: +1.0 unit (risk 1.1 to win 1.0 at -110)
    # Loss: -1.1 units
    # Push: 0 units
    if odds == -110.0:
        win_payout = 1.0
        loss_cost = 1.1
    else:
        # General case for other odds
        if odds < 0:
            win_payout = 100.0 / (-odds)
            loss_cost = 1.0
        else:
            win_payout = odds / 100.0
            loss_cost = 1.0

    units_won = (wins * win_payout) - (losses * loss_cost)

    # Calculate total risk (for ROI)
    total_risked = (wins + losses) * loss_cost

    # Calculate win rate (excluding pushes)
    decided_bets = wins + losses
    if decided_bets > 0:
        win_rate = wins / decided_bets
    else:
        win_rate = None

    # Calculate ROI
    if total_risked > 0:
        roi = units_won / total_risked
    else:
        roi = None

    return {
        'n_games': len(clean_df),
        'n_bets': len(bets_df),
        'win_rate': win_rate,
        'units_won': units_won,
        'roi': roi,
        'avg_edge': bets_df['abs_edge'].mean(),
        'edge_threshold': edge_threshold,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
    }


# ============================================================================
# V1.0 BACKTEST
# ============================================================================

def run_v1_0_backtest_on_frame(
    df: pd.DataFrame,
    edge_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Run v1.0 model backtest on a prepared DataFrame.

    v1.0 Model:
    -----------
    Simple nfelo-based spread prediction:
        predicted_spread = INTERCEPT + (nfelo_diff * COEF)

    Args:
        df: DataFrame from nfelo with required columns
        edge_threshold: Edge threshold for ATS betting

    Returns:
        Dictionary with accuracy and betting metrics
    """
    # v1.0 model parameters (calibrated)
    NFELO_COEF = 0.0447
    INTERCEPT = 2.67

    # Ensure required columns exist
    required_cols = [
        'starting_nfelo_home', 'starting_nfelo_away',
        'home_line_close', 'home_score', 'away_score'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to complete data
    mask = df[required_cols].notna().all(axis=1)
    eval_df = df[mask].copy()

    if len(eval_df) == 0:
        return {
            'model_name': 'v1.0',
            'n_games': 0,
            'mae_spread': None,
            'mae_total': None,
            'hit_rate_spread_within_3': None,
            'hit_rate_spread_within_7': None,
            'ats': simulate_ats_pnl(
                pd.DataFrame(),
                model_spread_col='pred_spread_v1_0',
                edge_threshold=edge_threshold
            ),
        }

    # Calculate predictions
    eval_df['nfelo_diff'] = eval_df['starting_nfelo_home'] - eval_df['starting_nfelo_away']
    eval_df['pred_spread_v1_0'] = INTERCEPT + (eval_df['nfelo_diff'] * NFELO_COEF)

    # Calculate actuals
    eval_df['actual_margin'] = eval_df['home_score'] - eval_df['away_score']
    eval_df['vegas_closing_spread'] = eval_df['home_line_close']

    # Accuracy metrics
    eval_df['error_spread'] = (eval_df['pred_spread_v1_0'] - eval_df['actual_margin']).abs()
    eval_df['error_total'] = (eval_df['pred_spread_v1_0'] - eval_df['vegas_closing_spread']).abs()

    mae_spread = eval_df['error_spread'].mean()
    mae_total = eval_df['error_total'].mean()

    # Hit rates
    hit_within_3 = (eval_df['error_spread'] <= 3.0).mean()
    hit_within_7 = (eval_df['error_spread'] <= 7.0).mean()

    # ATS/PnL simulation
    ats_results = simulate_ats_pnl(
        df=eval_df,
        model_spread_col='pred_spread_v1_0',
        market_spread_col='vegas_closing_spread',
        actual_margin_col='actual_margin',
        edge_threshold=edge_threshold,
    )

    return {
        'model_name': 'v1.0',
        'n_games': len(eval_df),
        'mae_spread': mae_spread,
        'mae_total': mae_total,
        'hit_rate_spread_within_3': hit_within_3,
        'hit_rate_spread_within_7': hit_within_7,
        'ats': ats_results,
    }


# ============================================================================
# V1.2 BACKTEST
# ============================================================================

def run_v1_2_backtest_on_frame(
    df: pd.DataFrame,
    edge_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Run v1.2 model backtest on a prepared DataFrame.

    v1.2 Model:
    -----------
    Enhanced spread prediction with situational adjustments:
        predicted_spread = intercept + nfelo_coef*nfelo_diff + ...

    Args:
        df: DataFrame from nfelo with required columns
        edge_threshold: Edge threshold for ATS betting

    Returns:
        Dictionary with accuracy and betting metrics
    """
    # Load v1.2 model parameters
    model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'

    if not model_file.exists():
        raise FileNotFoundError(
            f"v1.2 model file not found at {model_file}. "
            "Run the v1.2 training script first."
        )

    with open(model_file, 'r') as f:
        model_params = json.load(f)

    # Feature engineering
    df = df.copy()
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    df['rest_advantage'] = df['home_bye_mod'].fillna(0) + df['away_bye_mod'].fillna(0)
    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

    # Filter to complete data
    feature_cols = [
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff',
        'home_line_close', 'home_score', 'away_score'
    ]
    mask = df[feature_cols].notna().all(axis=1)
    eval_df = df[mask].copy()

    if len(eval_df) == 0:
        return {
            'model_name': 'v1.2',
            'n_games': 0,
            'mae_spread': None,
            'mae_total': None,
            'hit_rate_spread_within_3': None,
            'hit_rate_spread_within_7': None,
            'ats': simulate_ats_pnl(
                pd.DataFrame(),
                model_spread_col='pred_spread_v1_2',
                edge_threshold=edge_threshold
            ),
        }

    # Generate predictions
    intercept = model_params['intercept']
    coefs = model_params['coefficients']

    eval_df['pred_spread_v1_2'] = (
        intercept +
        (eval_df['nfelo_diff'] * coefs['nfelo_diff']) +
        (eval_df['rest_advantage'] * coefs['rest_advantage']) +
        (eval_df['div_game'] * coefs['div_game']) +
        (eval_df['surface_mod'] * coefs['surface_mod']) +
        (eval_df['time_advantage'] * coefs['time_advantage']) +
        (eval_df['qb_diff'] * coefs['qb_diff'])
    )

    # Calculate actuals
    eval_df['actual_margin'] = eval_df['home_score'] - eval_df['away_score']
    eval_df['vegas_closing_spread'] = eval_df['home_line_close']

    # Accuracy metrics
    eval_df['error_spread'] = (eval_df['pred_spread_v1_2'] - eval_df['actual_margin']).abs()
    eval_df['error_total'] = (eval_df['pred_spread_v1_2'] - eval_df['vegas_closing_spread']).abs()

    mae_spread = eval_df['error_spread'].mean()
    mae_total = eval_df['error_total'].mean()

    # Hit rates
    hit_within_3 = (eval_df['error_spread'] <= 3.0).mean()
    hit_within_7 = (eval_df['error_spread'] <= 7.0).mean()

    # ATS/PnL simulation
    ats_results = simulate_ats_pnl(
        df=eval_df,
        model_spread_col='pred_spread_v1_2',
        market_spread_col='vegas_closing_spread',
        actual_margin_col='actual_margin',
        edge_threshold=edge_threshold,
    )

    return {
        'model_name': 'v1.2',
        'n_games': len(eval_df),
        'mae_spread': mae_spread,
        'mae_total': mae_total,
        'hit_rate_spread_within_3': hit_within_3,
        'hit_rate_spread_within_7': hit_within_7,
        'ats': ats_results,
    }


# ============================================================================
# COMPARISON DRIVER
# ============================================================================

def compare_v1_models(
    test_seasons: Optional[List[int]] = None,
    edge_threshold: float = 1.0,
    nfelo_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Compare v1.0 and v1.2 models on accuracy and betting performance.

    Args:
        test_seasons: List of seasons to test on (default: [2020, 2021, 2022, 2023])
        edge_threshold: Minimum edge to place ATS bet (in points)
        nfelo_url: Optional custom nfelo data URL

    Returns:
        List of dictionaries, one per model, containing:
            - model_name: 'v1.0' or 'v1.2'
            - n_games: Number of games evaluated
            - mae_spread: Mean absolute error vs actual margin
            - mae_total: Mean absolute error vs Vegas line
            - hit_rate_spread_within_3: % of predictions within 3 points
            - hit_rate_spread_within_7: % of predictions within 7 points
            - ats: Dictionary with ATS betting metrics
                - n_bets: Number of bets placed
                - win_rate: Win percentage (excluding pushes)
                - units_won: Net profit in units
                - roi: Return on investment
                - avg_edge: Average edge per bet
    """
    if test_seasons is None:
        test_seasons = [2020, 2021, 2022, 2023]

    if nfelo_url is None:
        nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'

    # Load nfelo data
    print(f"Loading nfelo data from {nfelo_url}...")
    df = pd.read_csv(nfelo_url)

    # Extract season/week/teams
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Load actual game scores from schedules.parquet
    print("Loading actual scores from schedules.parquet...")
    schedules_path = project_root / 'schedules.parquet'
    if schedules_path.exists():
        schedules = pd.read_parquet(schedules_path)
        # Merge scores based on game_id
        df = df.merge(
            schedules[['game_id', 'home_score', 'away_score']],
            on='game_id',
            how='left'
        )
    else:
        print(f"Warning: {schedules_path} not found, cannot evaluate betting performance")
        df['home_score'] = np.nan
        df['away_score'] = np.nan

    # Filter to test seasons
    test_df = df[df['season'].isin(test_seasons)].copy()

    print(f"Evaluating on {len(test_df)} games from seasons {test_seasons}")
    print(f"Edge threshold: {edge_threshold} points\n")

    # Run backtests
    results = []

    print("Running v1.0 backtest...")
    v1_0_results = run_v1_0_backtest_on_frame(test_df, edge_threshold=edge_threshold)
    results.append(v1_0_results)

    print("Running v1.2 backtest...")
    v1_2_results = run_v1_2_backtest_on_frame(test_df, edge_threshold=edge_threshold)
    results.append(v1_2_results)

    print("Comparison complete!\n")

    return results
