#!/usr/bin/env python3
"""
Unified Backtest Driver for Ball Knower Models

Runs backtests for v1.0 or v1.2 models across specified season ranges
and edge thresholds. Outputs summary statistics including PnL and CLV metrics to CSV.

Features:
- ATS PnL tracking with flat-bet strategy (-110 pricing)
- Closing Line Value (CLV) analysis
- Win rate, ROI, and units won/lost metrics

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


# ============================================================================
# PNL AND CLV HELPER FUNCTIONS
# ============================================================================

def compute_ats_pnl(df: pd.DataFrame) -> dict:
    """
    Compute flat-bet ATS PnL given a DataFrame with:
        - bet_flag: 1 if we bet, 0 if no bet
        - bet_result: 1 for win, -1 for loss, 0 for push

    Returns a dict with:
        - n_bets: Total number of bets placed
        - n_wins: Number of winning bets
        - n_losses: Number of losing bets
        - n_pushes: Number of pushes
        - win_rate: Proportion of wins (excluding pushes)
        - units_won: Total units won/lost (win: +0.9091, loss: -1, push: 0)
        - roi: Return on investment (units_won / n_bets)
    """
    bets = df[df['bet_flag'] == 1].copy()

    if len(bets) == 0:
        return {
            'n_bets': 0,
            'n_wins': 0,
            'n_losses': 0,
            'n_pushes': 0,
            'win_rate': 0.0,
            'units_won': 0.0,
            'roi': 0.0
        }

    n_wins = (bets['bet_result'] == 1).sum()
    n_losses = (bets['bet_result'] == -1).sum()
    n_pushes = (bets['bet_result'] == 0).sum()

    # Standard -110 pricing: win = +0.9091 units, loss = -1 unit
    units_won = (n_wins * 0.9091) + (n_losses * -1.0)

    # Win rate excludes pushes
    n_decided = n_wins + n_losses
    win_rate = n_wins / n_decided if n_decided > 0 else 0.0

    # ROI is total units won per bet placed
    roi = units_won / len(bets)

    return {
        'n_bets': len(bets),
        'n_wins': int(n_wins),
        'n_losses': int(n_losses),
        'n_pushes': int(n_pushes),
        'win_rate': win_rate,
        'units_won': units_won,
        'roi': roi
    }


def compute_clv_metrics(df: pd.DataFrame) -> dict:
    """
    Compute simple CLV (Closing Line Value) metrics for rows where bet_flag == 1.

    CLV measures whether we're betting at better lines than the closing line.
    Positive CLV means we got a better line than close.

    Note: This implementation uses a simple proxy since we don't have opening lines.
    We compare our model's line against the closing line to estimate CLV.

    Returns:
        - n_bets: Number of bets
        - mean_clv_diff: Average CLV (positive = beating closing line)
        - pct_beating_closing_line: Percentage of bets with positive CLV
    """
    bets = df[df['bet_flag'] == 1].copy()

    if len(bets) == 0 or 'clv_diff' not in bets.columns:
        return {
            'n_bets': 0,
            'mean_clv_diff': 0.0,
            'pct_beating_closing_line': 0.0
        }

    # Filter out any NaN values
    valid_clv = bets['clv_diff'].notna()
    clv_values = bets.loc[valid_clv, 'clv_diff']

    if len(clv_values) == 0:
        return {
            'n_bets': len(bets),
            'mean_clv_diff': 0.0,
            'pct_beating_closing_line': 0.0
        }

    mean_clv = clv_values.mean()
    pct_positive = (clv_values > 0).sum() / len(clv_values)

    return {
        'n_bets': len(bets),
        'mean_clv_diff': mean_clv,
        'pct_beating_closing_line': pct_positive
    }


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
            - n_wins, n_losses, n_pushes (bet outcomes)
            - win_rate (wins / decided bets)
            - units_won (total units won/lost)
            - roi (return on investment)
            - mean_clv (average closing line value)
            - pct_beat_close (% of bets beating closing line)
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
    df = df[df['home_result_spread'].notna()].copy()  # Need actual results for PnL

    # v1.0 model parameters (calibrated)
    NFELO_COEF = 0.0447
    INTERCEPT = 2.67

    # Calculate predictions
    df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
    df['bk_v1_0_spread'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)

    # Calculate edge vs Vegas
    df['edge'] = df['bk_v1_0_spread'] - df['home_line_close']
    df['abs_edge'] = df['edge'].abs()

    # Determine which side to bet (negative edge = bet home, positive = bet away)
    # and compute bet outcomes
    df['bet_flag'] = (df['abs_edge'] >= edge_threshold).astype(int)
    df['bet_side'] = np.where(df['edge'] < 0, 'home', 'away')

    # Compute ATS result for each side
    # For home bets: win if home_result_spread > 0, lose if < 0, push if == 0
    # For away bets: win if home_result_spread < 0, lose if > 0, push if == 0
    def compute_bet_result(row):
        if row['bet_flag'] == 0:
            return 0  # No bet

        result = row['home_result_spread']

        # Check for push (exact tie against spread)
        if abs(result) < 0.01:  # Float comparison tolerance
            return 0

        # Determine win/loss based on bet side
        if row['bet_side'] == 'home':
            return 1 if result > 0 else -1
        else:  # away
            return 1 if result < 0 else -1

    df['bet_result'] = df.apply(compute_bet_result, axis=1)

    # CLV proxy: Use our edge as a simple CLV indicator
    # Positive edge on our bet side = beating closing line
    # Note: This is a simplified proxy. True CLV requires opening line data.
    df['clv_diff'] = np.where(df['bet_flag'] == 1, df['abs_edge'], np.nan)

    # Group by season and calculate metrics
    results = []
    for season in range(start_season, end_season + 1):
        season_df = df[df['season'] == season]

        if len(season_df) == 0:
            continue

        # Bets are games with edge >= threshold
        bets_df = season_df[season_df['abs_edge'] >= edge_threshold]

        # Compute PnL metrics
        pnl_metrics = compute_ats_pnl(season_df)

        # Compute CLV metrics
        clv_metrics = compute_clv_metrics(season_df)

        results.append({
            'season': season,
            'model': 'v1.0',
            'edge_threshold': edge_threshold,
            'n_games': len(season_df),
            'n_bets': len(bets_df),
            'mae_vs_vegas': season_df['abs_edge'].mean(),
            'rmse_vs_vegas': np.sqrt((season_df['edge'] ** 2).mean()),
            'mean_edge': season_df['edge'].mean(),
            # PnL metrics
            'n_wins': pnl_metrics['n_wins'],
            'n_losses': pnl_metrics['n_losses'],
            'n_pushes': pnl_metrics['n_pushes'],
            'win_rate': pnl_metrics['win_rate'],
            'units_won': pnl_metrics['units_won'],
            'roi': pnl_metrics['roi'],
            # CLV metrics
            'mean_clv': clv_metrics['mean_clv_diff'],
            'pct_beat_close': clv_metrics['pct_beating_closing_line'],
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
        DataFrame with one row per season containing:
            - season, model, edge_threshold
            - n_games, n_bets
            - mae_vs_vegas, rmse_vs_vegas, mean_edge
            - n_wins, n_losses, n_pushes, win_rate
            - units_won, roi
            - mean_clv, pct_beat_close
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
    df = df[df['home_result_spread'].notna()].copy()  # Need actual results for PnL

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

    # Determine which side to bet and compute bet outcomes
    df['bet_flag'] = (df['abs_edge'] >= edge_threshold).astype(int)
    df['bet_side'] = np.where(df['edge'] < 0, 'home', 'away')

    # Compute ATS result for each side
    def compute_bet_result(row):
        if row['bet_flag'] == 0:
            return 0  # No bet

        result = row['home_result_spread']

        # Check for push (exact tie against spread)
        if abs(result) < 0.01:  # Float comparison tolerance
            return 0

        # Determine win/loss based on bet side
        if row['bet_side'] == 'home':
            return 1 if result > 0 else -1
        else:  # away
            return 1 if result < 0 else -1

    df['bet_result'] = df.apply(compute_bet_result, axis=1)

    # CLV proxy: Use our edge as a simple CLV indicator
    df['clv_diff'] = np.where(df['bet_flag'] == 1, df['abs_edge'], np.nan)

    # Group by season and calculate metrics
    results = []
    for season in range(start_season, end_season + 1):
        season_df = df[df['season'] == season]

        if len(season_df) == 0:
            continue

        # Bets are games with edge >= threshold
        bets_df = season_df[season_df['abs_edge'] >= edge_threshold]

        # Compute PnL metrics
        pnl_metrics = compute_ats_pnl(season_df)

        # Compute CLV metrics
        clv_metrics = compute_clv_metrics(season_df)

        results.append({
            'season': season,
            'model': 'v1.2',
            'edge_threshold': edge_threshold,
            'n_games': len(season_df),
            'n_bets': len(bets_df),
            'mae_vs_vegas': season_df['abs_edge'].mean(),
            'rmse_vs_vegas': np.sqrt((season_df['edge'] ** 2).mean()),
            'mean_edge': season_df['edge'].mean(),
            # PnL metrics
            'n_wins': pnl_metrics['n_wins'],
            'n_losses': pnl_metrics['n_losses'],
            'n_pushes': pnl_metrics['n_pushes'],
            'win_rate': pnl_metrics['win_rate'],
            'units_won': pnl_metrics['units_won'],
            'roi': pnl_metrics['roi'],
            # CLV metrics
            'mean_clv': clv_metrics['mean_clv_diff'],
            'pct_beat_close': clv_metrics['pct_beating_closing_line'],
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

    # Print aggregated PnL and CLV metrics
    if len(results) > 0 and 'units_won' in results.columns:
        total_units = results['units_won'].sum()
        total_bets = results['n_bets'].sum()
        total_wins = results['n_wins'].sum()
        total_losses = results['n_losses'].sum()
        total_pushes = results['n_pushes'].sum()

        print(f"\n" + "="*60)
        print("PnL SUMMARY (Flat-bet strategy, -110 pricing)")
        print("="*60)
        print(f"  Total Bets:    {total_bets}")
        print(f"  Wins:          {total_wins}")
        print(f"  Losses:        {total_losses}")
        print(f"  Pushes:        {total_pushes}")
        if total_bets > 0:
            avg_win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
            avg_roi = total_units / total_bets
            print(f"  Win Rate:      {avg_win_rate:.1%} (excl. pushes)")
            print(f"  Total Units:   {total_units:+.2f}")
            print(f"  ROI:           {avg_roi:+.2%}")

        if 'mean_clv' in results.columns:
            avg_clv = results.loc[results['n_bets'] > 0, 'mean_clv'].mean()
            avg_pct_beat = results.loc[results['n_bets'] > 0, 'pct_beat_close'].mean()
            print(f"\n" + "="*60)
            print("CLV SUMMARY (Closing Line Value)")
            print("="*60)
            print(f"  Avg CLV:       {avg_clv:.2f} points")
            print(f"  % Beat Close:  {avg_pct_beat:.1%}")
            print(f"\nNote: CLV uses model edge as proxy (no opening line data)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
