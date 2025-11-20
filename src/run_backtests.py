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

from ball_knower import config
from ball_knower.features import engineering as features


# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def run_backtest_v1_0(
    start_season: int,
    end_season: int,
    edge_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Run backtest for v1.0 model across specified seasons.

    Args:
        start_season: Start season year
        end_season: End season year
        edge_threshold: Minimum edge threshold for "betting"

    Returns:
        DataFrame with one row per season containing:
            - season
            - model (v1.0)
            - edge_threshold
            - n_games
            - n_bets (games with edge >= threshold)
            - mae_vs_vegas
            - rmse_vs_vegas
            - mean_edge
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

    # v1.0 model parameters (calibrated)
    NFELO_COEF = 0.0447
    INTERCEPT = 2.67

    # Calculate predictions
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    df['bk_v1_0_spread'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)

    # Calculate edge vs Vegas
    df['edge'] = df['bk_v1_0_spread'] - df['home_line_close']
    df['abs_edge'] = df['edge'].abs()

    # Group by season and calculate metrics
    results = []
    for season in range(start_season, end_season + 1):
        season_df = df[df['season'] == season]

        if len(season_df) == 0:
            continue

        # Bets are games with edge >= threshold
        bets_df = season_df[season_df['abs_edge'] >= edge_threshold]

        results.append({
            'season': season,
            'model': 'v1.0',
            'edge_threshold': edge_threshold,
            'n_games': len(season_df),
            'n_bets': len(bets_df),
            'mae_vs_vegas': season_df['abs_edge'].mean(),
            'rmse_vs_vegas': np.sqrt((season_df['edge'] ** 2).mean()),
            'mean_edge': season_df['edge'].mean(),
        })

    return pd.DataFrame(results)


def run_backtest_v1_2(
    start_season: int,
    end_season: int,
    edge_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Run backtest for v1.2 model across specified seasons.

    Args:
        start_season: Start season year
        end_season: End season year
        edge_threshold: Minimum edge threshold for "betting"

    Returns:
        DataFrame with one row per season containing metrics
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

    # Engineer features
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    # Use canonical rest advantage calculation from ball_knower.features.engineering
    df['rest_advantage'] = features.compute_rest_advantage_from_nfelo(df)
    df['div_game'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
    df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

    # Target
    df['vegas_line'] = df['home_line_close']

    # Filter rows with complete features
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff', 'vegas_line']
    mask = df[feature_cols].notna().all(axis=1)
    df = df[mask].copy()

    # Generate v1.2 predictions
    intercept = model_params['intercept']
    coefs = model_params['coefficients']

    df['bk_v1_2_spread'] = intercept + \
        (df['nfelo_diff'] * coefs['nfelo_diff']) + \
        (df['rest_advantage'] * coefs['rest_advantage']) + \
        (df['div_game'] * coefs['div_game']) + \
        (df['surface_mod'] * coefs['surface_mod']) + \
        (df['time_advantage'] * coefs['time_advantage']) + \
        (df['qb_diff'] * coefs['qb_diff'])

    # Calculate edge
    df['edge'] = df['bk_v1_2_spread'] - df['vegas_line']
    df['abs_edge'] = df['edge'].abs()

    # Group by season and calculate metrics
    results = []
    for season in range(start_season, end_season + 1):
        season_df = df[df['season'] == season]

        if len(season_df) == 0:
            continue

        # Bets are games with edge >= threshold
        bets_df = season_df[season_df['abs_edge'] >= edge_threshold]

        results.append({
            'season': season,
            'model': 'v1.2',
            'edge_threshold': edge_threshold,
            'n_games': len(season_df),
            'n_bets': len(bets_df),
            'mae_vs_vegas': season_df['abs_edge'].mean(),
            'rmse_vs_vegas': np.sqrt((season_df['edge'] ** 2).mean()),
            'mean_edge': season_df['edge'].mean(),
        })

    return pd.DataFrame(results)


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

    # Run backtest
    print(f"\nRunning backtest for {args.model} model...")
    print(f"  Seasons: {args.start_season}-{args.end_season}")
    print(f"  Edge threshold: {args.edge_threshold}")

    if args.model == 'v1.0':
        results = run_backtest_v1_0(
            args.start_season,
            args.end_season,
            args.edge_threshold
        )
    else:  # v1.2
        results = run_backtest_v1_2(
            args.start_season,
            args.end_season,
            args.edge_threshold
        )

    # Determine output path
    if args.output is None:
        output_path = (
            config.OUTPUT_DIR /
            f"backtest_{args.model.replace('.', '_')}_{args.start_season}_{args.end_season}.csv"
        )
    else:
        output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results.to_csv(output_path, index=False)

    print(f"\n✓ Backtest complete!")
    print(f"  Results saved to: {output_path}")
    print(f"\nSummary:")
    print(results.to_string(index=False))

    return 0


if __name__ == '__main__':
    sys.exit(main())
