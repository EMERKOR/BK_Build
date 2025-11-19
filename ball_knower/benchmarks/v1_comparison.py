"""
Ball Knower v1.x Model Comparison Harness

This module provides utilities for comparing v1.0, v1.2, and v1.3 models
on a shared test period to evaluate their relative performance.

Model Descriptions
------------------
v1.0: Simple linear model using only nfelo_diff
      Predicts: spread only
      Model: spread = 2.67 + (nfelo_diff × 0.0447)

v1.2: Ridge regression with multiple features (nfelo_diff, rest, div_game, etc.)
      Predicts: spread only
      Features: nfelo_diff, rest_advantage, div_game, surface_mod, time_advantage, qb_diff

v1.3: Ridge regression predicting actual scores
      Predicts: home_score, away_score (derives spread and total)
      Features: Same as v1.2

Usage
-----
>>> from ball_knower.benchmarks.v1_comparison import compare_v1_models
>>> results = compare_v1_models(test_seasons=[2022, 2023])
>>> print(results['models']['v1.3']['mae_spread'])
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error


# ============================================================================
# V1.0 INTEGRATION
# ============================================================================
# How v1.0 works today (from archive/backtest_v1_0.py):
# - Uses simple formula: spread = 2.67 + (nfelo_diff × 0.0447)
# - nfelo_diff = starting_nfelo_home - starting_nfelo_away
# - Only predicts spreads, no totals

# Model parameters from v1.0 (calibrated from Week 11 2025)
V1_0_NFELO_COEF = 0.0447
V1_0_INTERCEPT = 2.67


def run_v1_0_backtest_on_frame(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run v1.0 backtest on a shared test frame.

    v1.0 uses a simple linear model: spread = 2.67 + (nfelo_diff × 0.0447)

    Parameters
    ----------
    df : pd.DataFrame
        Test frame with columns:
        - nfelo_diff: ELO differential (home - away)
        - actual_margin: Actual game margin (home_score - away_score)
        - home_score, away_score: Actual scores

    Returns
    -------
    dict
        {
            "model_name": "v1.0",
            "mae_spread": float,
            "mae_total": None (v1.0 doesn't predict totals),
            "rmse_spread": float,
            "hit_rate_spread_within_3": float,
            "hit_rate_spread_within_7": float,
            "n_games": int,
            "status": "ok" or error message
        }
    """
    try:
        # Validate required columns
        required_cols = ['nfelo_diff', 'actual_margin']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {
                "model_name": "v1.0",
                "status": f"missing columns: {missing}",
                "mae_spread": None,
                "mae_total": None,
                "n_games": 0
            }

        # Generate v1.0 predictions
        df = df.copy()
        df['bk_v1_0_spread'] = V1_0_INTERCEPT + (df['nfelo_diff'] * V1_0_NFELO_COEF)

        # Calculate errors
        df['spread_error'] = np.abs(df['bk_v1_0_spread'] - df['actual_margin'])
        mae_spread = df['spread_error'].mean()
        rmse_spread = np.sqrt((df['spread_error'] ** 2).mean())

        # Hit rates within thresholds
        hit_3 = (df['spread_error'] <= 3).mean() * 100
        hit_7 = (df['spread_error'] <= 7).mean() * 100

        return {
            "model_name": "v1.0",
            "mae_spread": float(mae_spread),
            "mae_total": None,  # v1.0 doesn't predict totals
            "rmse_spread": float(rmse_spread),
            "hit_rate_spread_within_3": float(hit_3),
            "hit_rate_spread_within_7": float(hit_7),
            "n_games": len(df),
            "status": "ok"
        }

    except Exception as e:
        return {
            "model_name": "v1.0",
            "status": f"error: {str(e)}",
            "mae_spread": None,
            "mae_total": None,
            "n_games": 0
        }


# ============================================================================
# V1.2 INTEGRATION
# ============================================================================
# How v1.2 works today (from archive/backtest_v1_2.py):
# - Ridge regression trained on 2009-2024
# - Features: nfelo_diff, rest_advantage, div_game, surface_mod, time_advantage, qb_diff
# - Loads model coefficients from JSON file
# - Only predicts spreads, no totals

def run_v1_2_backtest_on_frame(
    df: pd.DataFrame,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run v1.2 backtest on a shared test frame.

    v1.2 uses Ridge regression with multiple features to predict spreads.

    Parameters
    ----------
    df : pd.DataFrame
        Test frame with v1.2 feature columns:
        - nfelo_diff, rest_advantage, div_game, surface_mod, time_advantage, qb_diff
        - actual_margin: Actual game margin
        - home_score, away_score: Actual scores
    model_path : str, optional
        Path to v1.2 model JSON file. If None, uses default location.

    Returns
    -------
    dict
        {
            "model_name": "v1.2",
            "mae_spread": float,
            "mae_total": None (v1.2 doesn't predict totals),
            "rmse_spread": float,
            "hit_rate_spread_within_3": float,
            "hit_rate_spread_within_7": float,
            "n_games": int,
            "status": "ok" or error message
        }
    """
    try:
        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / 'output' / 'ball_knower_v1_2_model.json'

        # Load v1.2 model
        if not Path(model_path).exists():
            return {
                "model_name": "v1.2",
                "status": f"model file not found: {model_path}",
                "mae_spread": None,
                "mae_total": None,
                "n_games": 0
            }

        with open(model_path, 'r') as f:
            model_params = json.load(f)

        # Validate required columns
        feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                       'surface_mod', 'time_advantage', 'qb_diff']
        missing = [col for col in feature_cols + ['actual_margin'] if col not in df.columns]
        if missing:
            return {
                "model_name": "v1.2",
                "status": f"missing columns: {missing}",
                "mae_spread": None,
                "mae_total": None,
                "n_games": 0
            }

        # Generate v1.2 predictions
        df = df.copy()
        intercept = model_params['intercept']
        coefs = model_params['coefficients']

        df['bk_v1_2_spread'] = (
            intercept +
            (df['nfelo_diff'] * coefs['nfelo_diff']) +
            (df['rest_advantage'] * coefs['rest_advantage']) +
            (df['div_game'] * coefs['div_game']) +
            (df['surface_mod'] * coefs['surface_mod']) +
            (df['time_advantage'] * coefs['time_advantage']) +
            (df['qb_diff'] * coefs['qb_diff'])
        )

        # Calculate errors
        df['spread_error'] = np.abs(df['bk_v1_2_spread'] - df['actual_margin'])
        mae_spread = df['spread_error'].mean()
        rmse_spread = np.sqrt((df['spread_error'] ** 2).mean())

        # Hit rates
        hit_3 = (df['spread_error'] <= 3).mean() * 100
        hit_7 = (df['spread_error'] <= 7).mean() * 100

        return {
            "model_name": "v1.2",
            "mae_spread": float(mae_spread),
            "mae_total": None,  # v1.2 doesn't predict totals
            "rmse_spread": float(rmse_spread),
            "hit_rate_spread_within_3": float(hit_3),
            "hit_rate_spread_within_7": float(hit_7),
            "n_games": len(df),
            "status": "ok"
        }

    except Exception as e:
        return {
            "model_name": "v1.2",
            "status": f"error: {str(e)}",
            "mae_spread": None,
            "mae_total": None,
            "n_games": 0
        }


# ============================================================================
# V1.3 INTEGRATION
# ============================================================================
# How v1.3 works today (from ball_knower/modeling/v1_3):
# - Predicts actual scores (home_score, away_score)
# - Derives spread and total from score predictions
# - Uses same features as v1.2
# - Training done via training_template.py, backtesting via backtest_template.py

def run_v1_3_backtest_on_frame(
    df: pd.DataFrame,
    train_seasons: Optional[List[int]] = None,
    val_seasons: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Run v1.3 backtest on a shared test frame.

    v1.3 predicts actual scores (home_score, away_score) and derives spread/total.

    Parameters
    ----------
    df : pd.DataFrame
        Test frame with v1.2 feature columns plus:
        - home_score, away_score: Actual scores
        - actual_margin: Actual game margin
    train_seasons : list, optional
        Seasons to use for training. If None, uses all seasons before test period.
    val_seasons : list, optional
        Seasons to use for validation. If None, uses default [2019, 2020, 2021].

    Returns
    -------
    dict
        {
            "model_name": "v1.3",
            "mae_spread": float,
            "mae_total": float,
            "mae_home_score": float,
            "mae_away_score": float,
            "rmse_spread": float,
            "rmse_total": float,
            "hit_rate_spread_within_3": float,
            "hit_rate_spread_within_7": float,
            "hit_rate_total_within_3": float,
            "hit_rate_total_within_7": float,
            "n_games": int,
            "status": "ok" or error message
        }
    """
    try:
        from ball_knower.modeling.v1_3.training_template import train_v1_3
        from ball_knower.modeling.v1_3.score_model_template import ScorePredictionModelV13

        # Determine test seasons from the input df
        if 'season' not in df.columns:
            return {
                "model_name": "v1.3",
                "status": "missing 'season' column",
                "mae_spread": None,
                "mae_total": None,
                "n_games": 0
            }

        test_seasons = sorted(df['season'].unique().tolist())

        # Build training data using the same approach as the test frame
        # Get all seasons before test period
        min_test_season = min(test_seasons)
        all_train_seasons = list(range(2009, min_test_season))

        if not all_train_seasons:
            # If test starts at 2009, we have no historical data
            return {
                "model_name": "v1.3",
                "status": "no training data available (test starts at earliest season)",
                "mae_spread": None,
                "mae_total": None,
                "n_games": 0
            }

        # Default validation seasons
        if val_seasons is None:
            val_seasons = [2019, 2020, 2021]
            # Make sure val seasons don't overlap with test
            val_seasons = [s for s in val_seasons if s < min_test_season]

        # Build full training + validation frame
        train_val_seasons = [s for s in all_train_seasons if s not in val_seasons]
        full_df = build_common_test_frame(test_seasons=train_val_seasons + val_seasons)

        # Split into train and val
        train_df = full_df[full_df['season'].isin(train_val_seasons)].copy()
        val_df = full_df[full_df['season'].isin(val_seasons)].copy()

        if len(train_df) == 0:
            return {
                "model_name": "v1.3",
                "status": "no training data available",
                "mae_spread": None,
                "mae_total": None,
                "n_games": 0
            }

        # Train v1.3 model
        results = train_v1_3(
            train_df=train_df,
            val_df=val_df if len(val_df) > 0 else None,
            model_type='ridge',
            hyperparams={'alpha': 1.0}
        )

        # Create model instance
        model = ScorePredictionModelV13(
            home_model=results['home_model'],
            away_model=results['away_model'],
            feature_names=results['feature_names'],
            metadata=results['training_metadata']
        )

        # Generate predictions on test data (the input df)
        predictions_df = model.predict(df)

        # Compute metrics
        actual_home = df['home_score'].values
        actual_away = df['away_score'].values
        pred_home = predictions_df['pred_home_score'].values
        pred_away = predictions_df['pred_away_score'].values

        # Score metrics
        mae_home = mean_absolute_error(actual_home, pred_home)
        mae_away = mean_absolute_error(actual_away, pred_away)

        # Derived metrics
        actual_spread = actual_home - actual_away
        pred_spread = pred_home - pred_away
        actual_total = actual_home + actual_away
        pred_total = pred_home + pred_away

        mae_spread = mean_absolute_error(actual_spread, pred_spread)
        mae_total = mean_absolute_error(actual_total, pred_total)

        rmse_spread = np.sqrt(np.mean((actual_spread - pred_spread) ** 2))
        rmse_total = np.sqrt(np.mean((actual_total - pred_total) ** 2))

        # Hit rates
        spread_errors = np.abs(actual_spread - pred_spread)
        total_errors = np.abs(actual_total - pred_total)

        hit_spread_3 = (spread_errors <= 3).mean() * 100
        hit_spread_7 = (spread_errors <= 7).mean() * 100
        hit_total_3 = (total_errors <= 3).mean() * 100
        hit_total_7 = (total_errors <= 7).mean() * 100

        return {
            "model_name": "v1.3",
            "mae_spread": float(mae_spread),
            "mae_total": float(mae_total),
            "mae_home_score": float(mae_home),
            "mae_away_score": float(mae_away),
            "rmse_spread": float(rmse_spread),
            "rmse_total": float(rmse_total),
            "hit_rate_spread_within_3": float(hit_spread_3),
            "hit_rate_spread_within_7": float(hit_spread_7),
            "hit_rate_total_within_3": float(hit_total_3),
            "hit_rate_total_within_7": float(hit_total_7),
            "n_games": len(df),
            "status": "ok"
        }

    except Exception as e:
        import traceback
        return {
            "model_name": "v1.3",
            "status": f"error: {str(e)}\n{traceback.format_exc()}",
            "mae_spread": None,
            "mae_total": None,
            "n_games": 0
        }


# ============================================================================
# COMMON TEST FRAME BUILDER
# ============================================================================

def build_common_test_frame(
    test_seasons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Build a common test frame that all three models can use.

    This frame includes:
    - All features needed by v1.0, v1.2, and v1.3
    - Actual game outcomes (scores, margins)
    - Game identifiers (game_id, season, week, teams)

    Parameters
    ----------
    test_seasons : list of int, optional
        List of seasons to include in test frame.
        If None, defaults to [2022, 2023, 2024].

    Returns
    -------
    pd.DataFrame
        Common test frame with all features and outcomes
    """
    from pathlib import Path

    # Default test seasons
    if test_seasons is None:
        test_seasons = [2022, 2023, 2024]

    # Load actual game scores from local schedules data
    schedules_path = Path(__file__).parent.parent.parent / 'schedules.parquet'
    schedules = pd.read_parquet(schedules_path)

    # Filter to regular season and specified seasons
    schedules = schedules[schedules['game_type'] == 'REG'].copy()
    schedules = schedules[schedules['season'].isin(test_seasons)].copy()

    # Filter to completed games (have scores)
    schedules = schedules[schedules['home_score'].notna()].copy()
    schedules = schedules[schedules['away_score'].notna()].copy()

    # Load nfelo features
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    nfelo = pd.read_csv(nfelo_url)

    # Join schedules with nfelo features on game_id
    df = schedules.merge(nfelo, on='game_id', how='left', suffixes=('', '_nfelo'))

    # Use season/week from schedules (more reliable)
    if 'season_nfelo' in df.columns:
        df = df.drop(columns=['season_nfelo', 'week_nfelo'], errors='ignore')

    # Calculate nfelo_diff (required for all models)
    if 'nfelo_diff' not in df.columns:
        df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

    # Add v1.2 features
    df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
    df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
    df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']
    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

    # Add derived columns needed for evaluation
    df['actual_margin'] = df['home_score'] - df['away_score']
    df['actual_total'] = df['home_score'] + df['away_score']
    df['vegas_closing_spread'] = df['home_line_close']

    # Filter to games with complete nfelo features
    required_features = ['nfelo_diff', 'rest_advantage', 'div_game',
                        'surface_mod', 'time_advantage', 'qb_diff']
    for col in required_features:
        if col not in df.columns:
            df[col] = 0  # Default to 0 if feature not available

    # Remove games with missing critical data
    df = df[df['nfelo_diff'].notna()].copy()

    # Keep only numeric and essential columns to avoid issues with v1.3 training
    # Essential columns for identification and targets
    essential_cols = ['game_id', 'season', 'week', 'home_team', 'away_team',
                      'home_score', 'away_score', 'actual_margin', 'actual_total']

    # Feature columns for models
    feature_cols = required_features + ['vegas_closing_spread']

    # Select only essential and feature columns (all numeric except identifiers)
    keep_cols = essential_cols + feature_cols
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].copy()

    return df


# ============================================================================
# COMPARISON ORCHESTRATOR
# ============================================================================

def compare_v1_models(
    test_seasons: Optional[List[int]] = None,
    v1_2_model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run v1.0, v1.2, and v1.3 on a shared test period and return consolidated metrics.

    This is the main entry point for comparing all three v1.x models.

    Parameters
    ----------
    test_seasons : list of int, optional
        List of seasons to use for testing.
        If None, defaults to [2022, 2023, 2024] (consistent with v1.3 defaults).
    v1_2_model_path : str, optional
        Path to v1.2 model JSON file. If None, uses default location.

    Returns
    -------
    dict
        {
            "test_seasons": list of int,
            "n_games": int,
            "models": {
                "v1.0": {dict of metrics},
                "v1.2": {dict of metrics},
                "v1.3": {dict of metrics}
            }
        }

    Examples
    --------
    >>> results = compare_v1_models(test_seasons=[2022, 2023])
    >>> print(f"v1.0 Spread MAE: {results['models']['v1.0']['mae_spread']:.2f}")
    >>> print(f"v1.2 Spread MAE: {results['models']['v1.2']['mae_spread']:.2f}")
    >>> print(f"v1.3 Spread MAE: {results['models']['v1.3']['mae_spread']:.2f}")
    >>> print(f"v1.3 Total MAE: {results['models']['v1.3']['mae_total']:.2f}")
    """
    # Default test seasons
    if test_seasons is None:
        test_seasons = [2022, 2023, 2024]

    print(f"\n{'='*80}")
    print(f"Ball Knower v1.x Model Comparison")
    print(f"{'='*80}")
    print(f"Test seasons: {test_seasons}")

    # Build common test frame
    print(f"\nBuilding common test frame...")
    test_df = build_common_test_frame(test_seasons=test_seasons)
    print(f"  Loaded {len(test_df)} games")

    # Run each model
    print(f"\nRunning v1.0 backtest...")
    v1_0_results = run_v1_0_backtest_on_frame(test_df)

    print(f"Running v1.2 backtest...")
    v1_2_results = run_v1_2_backtest_on_frame(test_df, model_path=v1_2_model_path)

    print(f"Running v1.3 backtest...")
    v1_3_results = run_v1_3_backtest_on_frame(test_df)

    # Assemble results
    return {
        "test_seasons": test_seasons,
        "n_games": len(test_df),
        "models": {
            "v1.0": v1_0_results,
            "v1.2": v1_2_results,
            "v1.3": v1_3_results
        }
    }
