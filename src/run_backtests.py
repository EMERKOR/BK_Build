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
from ball_knower.utils import paths, version


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
    model_file = paths.get_model_artifact_path("v1.2", "ball_knower_v1_2_model.json")

    if not model_file.exists():
        raise FileNotFoundError(
            f"v1.2 model file not found at {model_file}. "
            "Train the v1.2 model first."
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


def run_backtest_v1_3(
    start_season: int,
    end_season: int,
    edge_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Run backtest for v1.3 model across specified seasons.

    Args:
        start_season: Start season year (must be >= 2013 for form features)
        end_season: End season year
        edge_threshold: Minimum edge threshold for "betting"

    Returns:
        DataFrame with one row per season containing metrics
    """
    import joblib
    from ball_knower.datasets import v1_3

    # Validate start year (v1.3 requires EPA data from 2013+)
    if start_season < 2013:
        raise ValueError(
            f"v1.3 backtest requires start_season >= 2013 (got {start_season}). "
            "Team-week EPA data needed for form features starts in 2013."
        )

    # Load trained v1.3 model
    model_dir = paths.get_models_dir("v1.3")
    model_file = model_dir / "model.pkl"
    features_file = model_dir / "features.json"

    if not model_file.exists():
        raise FileNotFoundError(
            f"v1.3 model file not found at {model_file}. "
            "Train the v1.3 model first using: python src/bk_build.py train-v1-3"
        )

    if not features_file.exists():
        raise FileNotFoundError(
            f"v1.3 features file not found at {features_file}. "
            "Train the v1.3 model first."
        )

    # Load model and feature names
    model = joblib.load(model_file)
    with open(features_file, 'r') as f:
        features_metadata = json.load(f)
        feature_names = features_metadata['features']

    print(f"  ✓ Loaded v1.3 model from {model_file}")
    print(f"  ✓ Using {len(feature_names)} features: {', '.join(feature_names)}")

    # Build v1.3 dataset for backtest period
    print(f"  ✓ Building v1.3 dataset ({start_season}-{end_season})...")
    df = v1_3.build_training_frame(
        start_year=start_season,
        end_year=end_season
    )
    print(f"  ✓ Loaded {len(df)} games")

    # Filter to rows with complete features and vegas line
    required_cols = feature_names + ['vegas_closing_spread']
    mask = df[required_cols].notna().all(axis=1)
    df = df[mask].copy()

    print(f"  ✓ {len(df)} games with complete features")

    # Generate v1.3 predictions
    X = df[feature_names].values
    df['bk_v1_3_spread'] = model.predict(X)

    # Calculate edge vs Vegas
    df['vegas_line'] = df['vegas_closing_spread']
    df['edge'] = df['bk_v1_3_spread'] - df['vegas_line']
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
            'model': 'v1.3',
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
        choices=['v1.0', 'v1.2', 'v1.3'],
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

    # Print version banner
    version.print_version_banner("run_backtests", model_version=args.model)

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
    elif args.model == 'v1.2':
        results = run_backtest_v1_2(
            args.start_season,
            args.end_season,
            args.edge_threshold
        )
    elif args.model == 'v1.3':
        results = run_backtest_v1_3(
            args.start_season,
            args.end_season,
            args.edge_threshold
        )
    else:
        print(f"Error: Unknown model '{args.model}'", file=sys.stderr)
        return 1

    # Determine output path
    if args.output is None:
        output_path = paths.get_backtest_results_path(
            args.model,
            args.start_season,
            args.end_season
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
