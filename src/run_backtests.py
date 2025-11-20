#!/usr/bin/env python3
"""
Unified Backtest Driver for Ball Knower Models

Runs backtests for v1.0 or v1.2 models across specified season ranges
and edge thresholds. Outputs summary statistics to CSV.

Usage:
    python src/run_backtests.py --start-season 2019 --end-season 2019 \
        --model v1.2 --edge-threshold 0.5 --output output/backtest_results.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src import config
from ball_knower.features import engineering as features
from ball_knower.evaluation import metrics as eval_metrics


# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def run_backtest_v1_0(
    start_season: int,
    end_season: int,
    edge_threshold: float = 0.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run backtest for v1.0 model across specified seasons.

    Args:
        start_season: Start season year
        end_season: End season year
        edge_threshold: Minimum edge threshold for "betting"
        verbose: Print summary statistics

    Returns:
        DataFrame with one row per game containing all predictions and actuals
    """
    # Load nfelo data
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(nfelo_url)

    # Extract season/week/teams
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to season range
    df = df[(df['season'] >= start_season) & (df['season'] <= end_season)].copy()

    # Filter to complete data
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()
    df = df[df['home_score'].notna()].copy()
    df = df[df['away_score'].notna()].copy()

    # v1.0 model parameters (calibrated)
    NFELO_COEF = 0.0447
    INTERCEPT = 2.67

    # Calculate predictions
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    df['model_line'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)
    df['bk_line'] = df['model_line']  # For compatibility

    # Actual outcomes
    df['actual_margin'] = df['home_score'] - df['away_score']
    df['closing_spread'] = df['home_line_close']

    # Calculate edge vs Vegas
    df['edge'] = df['model_line'] - df['closing_spread']
    df['abs_edge'] = df['edge'].abs()

    # Mark bets (games with edge >= threshold)
    df['bet'] = (df['abs_edge'] >= edge_threshold).astype(int)

    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL: v1.0 (Baseline nfelo)")
        print(f"SEASONS: {start_season}-{end_season}")
        print(f"{'='*70}")

        # Compute overall metrics using centralized functions
        mae = eval_metrics.compute_mae(df['actual_margin'], df['model_line'])
        rmse = eval_metrics.compute_rmse(df['actual_margin'], df['model_line'])
        ats = eval_metrics.compute_ats_record(
            df['actual_margin'].values,
            df['model_line'].values,
            df['closing_spread'].values
        )
        edge_ev = eval_metrics.compute_edge_and_ev(
            df['actual_margin'].values,
            df['model_line'].values,
            df['closing_spread'].values
        )

        n_bets = df['bet'].sum()

        print(f"\nGames analyzed: {len(df)}")
        print(f"Games with edge >= {edge_threshold}: {n_bets}")
        print(f"\nPredictive Accuracy:")
        print(f"  MAE vs Actual:      {mae:.2f} points")
        print(f"  RMSE vs Actual:     {rmse:.2f} points")
        print(f"\nAgainst The Spread:")
        print(f"  Wins:               {ats['wins']}")
        print(f"  Losses:             {ats['losses']}")
        print(f"  Pushes:             {ats['pushes']}")
        print(f"  Win %:              {ats['win_pct']:.1%}")
        print(f"\nEdge Analysis:")
        print(f"  Mean edge:          {edge_ev['mean_edge']:.2f} points")
        print(f"  Median edge:        {edge_ev['median_edge']:.2f} points")
        print(f"  Max edge:           {edge_ev['max_edge']:.2f} points")
        print(f"  Flat-bet ROI:       {edge_ev['roi']:.2%}")
        print(f"{'='*70}\n")

    return df


def run_backtest_v1_2(
    start_season: int,
    end_season: int,
    edge_threshold: float = 0.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run backtest for v1.2 model across specified seasons.

    Args:
        start_season: Start season year
        end_season: End season year
        edge_threshold: Minimum edge threshold for "betting"
        verbose: Print summary statistics

    Returns:
        DataFrame with one row per game containing all predictions and actuals
    """
    # Load trained v1.2 model parameters
    model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'

    if not model_file.exists():
        raise FileNotFoundError(
            f"v1.2 model file not found at {model_file}. "
            "Run ball_knower_v1_2.py to train the model first."
        )

    with open(model_file, 'r') as f:
        model_params = json.load(f)

    # Load nfelo data
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(nfelo_url)

    # Extract season/week/teams
    df[['season', 'week', 'away_team', 'home_team']] = \
        df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to season range
    df = df[(df['season'] >= start_season) & (df['season'] <= end_season)].copy()

    # Filter to complete data
    df = df[df['home_line_close'].notna()].copy()
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()
    df = df[df['home_score'].notna()].copy()
    df = df[df['away_score'].notna()].copy()

    # Engineer features
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    # Use canonical rest advantage calculation from ball_knower.features.engineering
    df['rest_advantage'] = features.compute_rest_advantage_from_nfelo(df)
    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

    # Actual outcomes
    df['actual_margin'] = df['home_score'] - df['away_score']
    df['closing_spread'] = df['home_line_close']

    # Filter rows with complete features
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff', 'closing_spread']
    mask = df[feature_cols].notna().all(axis=1)
    df = df[mask].copy()

    # Generate v1.2 predictions
    intercept = model_params['intercept']
    coefs = model_params['coefficients']

    df['model_line'] = intercept + \
        (df['nfelo_diff'] * coefs['nfelo_diff']) + \
        (df['rest_advantage'] * coefs['rest_advantage']) + \
        (df['div_game'] * coefs['div_game']) + \
        (df['surface_mod'] * coefs['surface_mod']) + \
        (df['time_advantage'] * coefs['time_advantage']) + \
        (df['qb_diff'] * coefs['qb_diff'])

    df['bk_line'] = df['model_line']  # For compatibility

    # Calculate edge vs Vegas
    df['edge'] = df['model_line'] - df['closing_spread']
    df['abs_edge'] = df['edge'].abs()

    # Mark bets (games with edge >= threshold)
    df['bet'] = (df['abs_edge'] >= edge_threshold).astype(int)

    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL: v1.2 (Enhanced Ridge Regression)")
        print(f"SEASONS: {start_season}-{end_season}")
        print(f"{'='*70}")

        # Compute overall metrics using centralized functions
        mae = eval_metrics.compute_mae(df['actual_margin'], df['model_line'])
        rmse = eval_metrics.compute_rmse(df['actual_margin'], df['model_line'])
        ats = eval_metrics.compute_ats_record(
            df['actual_margin'].values,
            df['model_line'].values,
            df['closing_spread'].values
        )
        edge_ev = eval_metrics.compute_edge_and_ev(
            df['actual_margin'].values,
            df['model_line'].values,
            df['closing_spread'].values
        )

        n_bets = df['bet'].sum()

        print(f"\nGames analyzed: {len(df)}")
        print(f"Games with edge >= {edge_threshold}: {n_bets}")
        print(f"\nPredictive Accuracy:")
        print(f"  MAE vs Actual:      {mae:.2f} points")
        print(f"  RMSE vs Actual:     {rmse:.2f} points")
        print(f"\nAgainst The Spread:")
        print(f"  Wins:               {ats['wins']}")
        print(f"  Losses:             {ats['losses']}")
        print(f"  Pushes:             {ats['pushes']}")
        print(f"  Win %:              {ats['win_pct']:.1%}")
        print(f"\nEdge Analysis:")
        print(f"  Mean edge:          {edge_ev['mean_edge']:.2f} points")
        print(f"  Median edge:        {edge_ev['median_edge']:.2f} points")
        print(f"  Max edge:           {edge_ev['max_edge']:.2f} points")
        print(f"  Flat-bet ROI:       {edge_ev['roi']:.2%}")
        print(f"{'='*70}\n")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified backtest driver for Ball Knower models'
    )

    parser.add_argument(
        '--start-season',
        type=int,
        required=True,
        help='Start season year (e.g., 2019)'
    )

    parser.add_argument(
        '--end-season',
        type=int,
        required=True,
        help='End season year (e.g., 2024)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['v1.0', 'v1.2'],
        required=True,
        help='Model version to backtest'
    )

    parser.add_argument(
        '--edge-threshold',
        type=float,
        default=0.0,
        help='Minimum edge threshold for betting (default: 0.0)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: output/backtest_{model}_{start}-{end}.csv)'
    )

    args = parser.parse_args()

    # Validate season range
    if args.start_season > args.end_season:
        print(f"Error: start-season ({args.start_season}) cannot be greater than "
              f"end-season ({args.end_season})", file=sys.stderr)
        return 1

    # Run backtest (verbose mode prints metrics automatically)
    if args.model == 'v1.0':
        results_df = run_backtest_v1_0(
            args.start_season,
            args.end_season,
            args.edge_threshold,
            verbose=True
        )
    else:  # v1.2
        results_df = run_backtest_v1_2(
            args.start_season,
            args.end_season,
            args.edge_threshold,
            verbose=True
        )

    # Determine output path using standardized convention
    if args.output is None:
        # Standard convention: output/backtests/{model_version}/backtest_{model_version}_{start}_{end}.csv
        model_dir = config.OUTPUT_DIR / 'backtests' / args.model
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / f"backtest_{args.model}_{args.start_season}_{args.end_season}.csv"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure consistent schema across all model versions
    # Standardized column order for backtest CSVs
    standard_cols = [
        'game_id', 'season', 'week',
        'home_team', 'away_team',
        'actual_margin', 'closing_spread',
        'bk_line', 'model_line',
        'edge', 'abs_edge', 'bet'
    ]

    # Select columns that exist in the DataFrame
    output_cols = [col for col in standard_cols if col in results_df.columns]

    # Add any additional columns not in the standard list (preserve extra data)
    extra_cols = [col for col in results_df.columns if col not in standard_cols]
    output_cols.extend(extra_cols)

    # Save game-level results with consistent schema
    results_df[output_cols].to_csv(output_path, index=False)

    print(f"✓ Game-level results saved to: {output_path}")
    print(f"  {len(results_df)} games written")
    print(f"  Standard schema: {', '.join(standard_cols[:5])}...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
