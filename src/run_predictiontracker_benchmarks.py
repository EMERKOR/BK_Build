#!/usr/bin/env python3
"""
PredictionTracker Benchmark Script

Compares Ball Knower predictions against The Prediction Tracker's weekly CSV data.
Evaluates prediction accuracy vs actual margins and Vegas spreads.

Usage:
    python src/run_predictiontracker_benchmarks.py --season 2025 --week 11
    python src/run_predictiontracker_benchmarks.py --csv data/predictiontracker/nflpredictions.csv
    python src/run_predictiontracker_benchmarks.py --season 2025 --week 11 --output results/pt_benchmark.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from team_mapping import normalize_team_name
from betting_utils import american_to_implied_prob


# ============================================================================
# PREDICTIONTRACKER TEAM NAME MAPPINGS
# ============================================================================

# PredictionTracker uses full team names, sometimes abbreviated
PT_TEAM_MAPPING = {
    # Standard full names
    'Arizona': 'ARI', 'Cardinals': 'ARI', 'Arizona Cardinals': 'ARI',
    'Atlanta': 'ATL', 'Falcons': 'ATL', 'Atlanta Falcons': 'ATL',
    'Baltimore': 'BAL', 'Ravens': 'BAL', 'Baltimore Ravens': 'BAL',
    'Buffalo': 'BUF', 'Bills': 'BUF', 'Buffalo Bills': 'BUF',
    'Carolina': 'CAR', 'Panthers': 'CAR', 'Carolina Panthers': 'CAR',
    'Chicago': 'CHI', 'Bears': 'CHI', 'Chicago Bears': 'CHI',
    'Cincinnati': 'CIN', 'Bengals': 'CIN', 'Cincinnati Bengals': 'CIN',
    'Cleveland': 'CLE', 'Browns': 'CLE', 'Cleveland Browns': 'CLE',
    'Dallas': 'DAL', 'Cowboys': 'DAL', 'Dallas Cowboys': 'DAL',
    'Denver': 'DEN', 'Broncos': 'DEN', 'Denver Broncos': 'DEN',
    'Detroit': 'DET', 'Lions': 'DET', 'Detroit Lions': 'DET',
    'Green Bay': 'GB', 'Packers': 'GB', 'Green Bay Packers': 'GB',
    'Houston': 'HOU', 'Texans': 'HOU', 'Houston Texans': 'HOU',
    'Indianapolis': 'IND', 'Colts': 'IND', 'Indianapolis Colts': 'IND',
    'Jacksonville': 'JAX', 'Jaguars': 'JAX', 'Jacksonville Jaguars': 'JAX',
    'Kansas City': 'KC', 'Chiefs': 'KC', 'Kansas City Chiefs': 'KC',
    'LA Chargers': 'LAC', 'L.A. Chargers': 'LAC', 'Los Angeles Chargers': 'LAC', 'Chargers': 'LAC',
    'LA Rams': 'LAR', 'L.A. Rams': 'LAR', 'Los Angeles Rams': 'LAR', 'Rams': 'LAR',
    'Las Vegas': 'LV', 'Raiders': 'LV', 'Las Vegas Raiders': 'LV',
    'Miami': 'MIA', 'Dolphins': 'MIA', 'Miami Dolphins': 'MIA',
    'Minnesota': 'MIN', 'Vikings': 'MIN', 'Minnesota Vikings': 'MIN',
    'New England': 'NE', 'Patriots': 'NE', 'New England Patriots': 'NE',
    'New Orleans': 'NO', 'Saints': 'NO', 'New Orleans Saints': 'NO',
    'NY Giants': 'NYG', 'N.Y. Giants': 'NYG', 'New York Giants': 'NYG', 'Giants': 'NYG',
    'NY Jets': 'NYJ', 'N.Y. Jets': 'NYJ', 'New York Jets': 'NYJ', 'Jets': 'NYJ',
    'Philadelphia': 'PHI', 'Eagles': 'PHI', 'Philadelphia Eagles': 'PHI',
    'Pittsburgh': 'PIT', 'Steelers': 'PIT', 'Pittsburgh Steelers': 'PIT',
    'San Francisco': 'SF', '49ers': 'SF', 'San Francisco 49ers': 'SF',
    'Seattle': 'SEA', 'Seahawks': 'SEA', 'Seattle Seahawks': 'SEA',
    'Tampa Bay': 'TB', 'Buccaneers': 'TB', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee': 'TEN', 'Titans': 'TEN', 'Tennessee Titans': 'TEN',
    'Washington': 'WAS', 'Commanders': 'WAS', 'Washington Commanders': 'WAS',
}


def normalize_pt_team_name(name):
    """Normalize PredictionTracker team names to standard abbreviations."""
    if pd.isna(name):
        return None

    name = str(name).strip()

    # Direct lookup in PT mapping
    if name in PT_TEAM_MAPPING:
        return PT_TEAM_MAPPING[name]

    # Fall back to general normalize function
    return normalize_team_name(name)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_predictiontracker_csv(csv_path, season=None, week=None):
    """
    Load and parse PredictionTracker CSV.

    Args:
        csv_path: Path to nflpredictions.csv
        season: Filter to specific season (optional)
        week: Filter to specific week (optional)

    Returns:
        DataFrame with normalized team names and predictor columns
    """
    df = pd.read_csv(csv_path)

    # Print columns for debugging
    print(f"PredictionTracker CSV columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")

    # Filter by season/week if specified
    if season is not None and 'Season' in df.columns:
        df = df[df['Season'] == season].copy()
    if week is not None and 'Week' in df.columns:
        df = df[df['Week'] == week].copy()

    # Normalize team names - handle multiple column name formats
    if 'HomeTeam' in df.columns:
        df['home_team'] = df['HomeTeam'].apply(normalize_pt_team_name)
    elif 'home' in df.columns:
        df['home_team'] = df['home'].apply(normalize_pt_team_name)

    if 'AwayTeam' in df.columns:
        df['away_team'] = df['AwayTeam'].apply(normalize_pt_team_name)
    elif 'road' in df.columns:
        df['away_team'] = df['road'].apply(normalize_pt_team_name)

    print(f"Loaded {len(df)} games from PredictionTracker")

    return df


def load_ball_knower_predictions(season, week):
    """
    Load Ball Knower predictions for specified season/week.

    Args:
        season: NFL season year
        week: Week number

    Returns:
        DataFrame with BK predictions
    """
    # Try multiple potential locations
    potential_paths = [
        f'output/week_{week}_value_bets_v1_2.csv',
        f'output/week{week}_predictions.csv',
        f'predictions/week{week}_predictions_{season}.csv',
        f'predictions/week_{week}_{season}.csv',
        f'predictions/{season}_week{week}.csv',
        f'output/predictions_week{week}_{season}.csv',
    ]

    for path in potential_paths:
        if Path(path).exists():
            print(f"Loading Ball Knower predictions from: {path}")
            df = pd.read_csv(path)

            # Normalize team names if needed
            if 'home_team' in df.columns:
                df['home_team'] = df['home_team'].apply(normalize_team_name)
            if 'away_team' in df.columns:
                df['away_team'] = df['away_team'].apply(normalize_team_name)

            return df

    print(f"WARNING: No Ball Knower predictions found for {season} Week {week}")
    print(f"Searched: {potential_paths}")
    return None


# ============================================================================
# BENCHMARKING METRICS
# ============================================================================

def calculate_mae(predictions, actuals):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predictions - actuals))


def calculate_rmse(predictions, actuals):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - actuals) ** 2))


def calculate_accuracy(predictions, actuals):
    """Calculate prediction accuracy (correct winner percentage)."""
    pred_winners = predictions < 0  # Negative spread = home team favored
    actual_winners = actuals < 0    # Negative margin = home team won
    return np.mean(pred_winners == actual_winners)


def calculate_ats_accuracy(predictions, vegas_spreads, actuals):
    """Calculate Against-The-Spread accuracy."""
    # Prediction beats spread if closer to actual than Vegas
    pred_errors = np.abs(predictions - actuals)
    vegas_errors = np.abs(vegas_spreads - actuals)
    return np.mean(pred_errors < vegas_errors)


# ============================================================================
# BENCHMARKING PIPELINE
# ============================================================================

def run_benchmark(pt_data, bk_predictions=None, output_path=None):
    """
    Run comprehensive benchmark analysis.

    Args:
        pt_data: PredictionTracker DataFrame
        bk_predictions: Ball Knower predictions DataFrame (optional)
        output_path: Where to save detailed results CSV

    Returns:
        Dictionary of benchmark metrics
    """
    results = {
        'total_games': len(pt_data),
        'matched_games': 0,
        'unmatched_games': 0,
        'bk_mae_vs_actual': None,
        'bk_mae_vs_vegas': None,
        'bk_accuracy': None,
        'bk_ats_accuracy': None,
    }

    # Check required columns
    required_cols = ['home_team', 'away_team']
    missing_cols = [col for col in required_cols if col not in pt_data.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {pt_data.columns.tolist()}")
        return results

    # Identify actual scores and Vegas lines in PT data
    # Common column names: HomeScore, AwayScore, Spread, Line, etc.
    actual_margin_col = None
    vegas_spread_col = None

    for col in pt_data.columns:
        if 'homescore' in col.lower() and 'away' not in col.lower():
            home_score_col = col
        if 'awayscore' in col.lower():
            away_score_col = col
        if 'spread' in col.lower() or col == 'line':
            if vegas_spread_col is None:  # Use first found
                vegas_spread_col = col

    # IMPORTANT: PredictionTracker 'line' column is from ROAD/AWAY perspective
    # (positive = home favored), but Ball Knower uses HOME perspective
    # (negative = home favored). Convert PT to match BK convention.
    if vegas_spread_col == 'line':
        pt_data['vegas_spread_home'] = -pt_data['line']  # Flip sign
        vegas_spread_col = 'vegas_spread_home'

    # Also convert PT average to home perspective for fair comparison
    if 'lineavg' in pt_data.columns:
        pt_data['pt_avg_home'] = -pt_data['lineavg']

    # Calculate actual margin if scores available
    if 'home_score_col' in locals() and 'away_score_col' in locals():
        pt_data['actual_margin'] = pt_data[home_score_col] - pt_data[away_score_col]
        actual_margin_col = 'actual_margin'
        print(f"Calculated actual margins from {home_score_col} - {away_score_col}")

    # If Ball Knower predictions provided, merge and compare
    if bk_predictions is not None:
        # Determine which BK column names are available
        bk_spread_col = 'bk_v1_2_spread' if 'bk_v1_2_spread' in bk_predictions.columns else 'predicted_spread'

        merge_cols = ['home_team', 'away_team', bk_spread_col]
        if 'predicted_margin' in bk_predictions.columns:
            merge_cols.append('predicted_margin')

        merged = pt_data.merge(
            bk_predictions[merge_cols],
            on=['home_team', 'away_team'],
            how='left',
            suffixes=('_pt', '_bk')
        )

        # Rename to standard column name for consistency
        if bk_spread_col in merged.columns:
            merged['predicted_spread'] = merged[bk_spread_col]

        results['matched_games'] = merged['predicted_spread'].notna().sum()
        results['unmatched_games'] = merged['predicted_spread'].isna().sum()

        print(f"\n{'='*60}")
        print(f"MATCH RESULTS:")
        print(f"  Matched: {results['matched_games']}")
        print(f"  Unmatched: {results['unmatched_games']}")
        print(f"{'='*60}\n")

        # Show unmatched games
        if results['unmatched_games'] > 0:
            unmatched = merged[merged['predicted_spread'].isna()][['home_team', 'away_team']]
            print("UNMATCHED GAMES:")
            for _, row in unmatched.iterrows():
                print(f"  {row['away_team']} @ {row['home_team']}")
            print()

        # Calculate metrics for matched games
        matched_data = merged[merged['predicted_spread'].notna()].copy()

        if len(matched_data) > 0:
            # MAE vs actual margin (if we have both predicted margin and actual scores)
            if actual_margin_col and actual_margin_col in matched_data.columns:
                # Use predicted_margin if available, otherwise use predicted_spread
                if 'predicted_margin' in matched_data.columns:
                    bk_preds = matched_data['predicted_margin'].values
                else:
                    bk_preds = -matched_data['predicted_spread'].values  # Negative because spread is home team perspective

                actuals = matched_data[actual_margin_col].values
                results['bk_mae_vs_actual'] = calculate_mae(bk_preds, actuals)
                results['bk_rmse_vs_actual'] = calculate_rmse(bk_preds, actuals)
                results['bk_accuracy'] = calculate_accuracy(bk_preds, actuals) * 100

            # MAE vs Vegas
            if vegas_spread_col and vegas_spread_col in matched_data.columns:
                bk_preds = matched_data['predicted_spread'].values
                vegas_lines = matched_data[vegas_spread_col].values
                results['bk_mae_vs_vegas'] = calculate_mae(bk_preds, vegas_lines)

                # ATS accuracy if we have actuals
                if actual_margin_col:
                    actuals = matched_data[actual_margin_col].values
                    results['bk_ats_accuracy'] = calculate_ats_accuracy(
                        bk_preds, vegas_lines, actuals
                    ) * 100

            # MAE vs PredictionTracker consensus
            if 'pt_avg_home' in matched_data.columns:
                bk_preds = matched_data['predicted_spread'].values
                pt_consensus = matched_data['pt_avg_home'].values
                results['bk_mae_vs_pt_consensus'] = calculate_mae(bk_preds, pt_consensus)

        # Save detailed results
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(output_file, index=False)
            print(f"Detailed results saved to: {output_file}")

    return results


def print_summary(results):
    """Print formatted benchmark summary."""
    print(f"\n{'='*60}")
    print(f"PREDICTION TRACKER BENCHMARK RESULTS")
    print(f"{'='*60}\n")

    print(f"Total Games: {results['total_games']}")
    print(f"Matched Games: {results['matched_games']}")
    print(f"Unmatched Games: {results['unmatched_games']}")
    print()

    if results['bk_mae_vs_actual'] is not None:
        print(f"Ball Knower Performance:")
        print(f"  MAE vs Actual Margin: {results['bk_mae_vs_actual']:.2f} points")
        if results.get('bk_rmse_vs_actual'):
            print(f"  RMSE vs Actual: {results['bk_rmse_vs_actual']:.2f} points")
        if results.get('bk_accuracy'):
            print(f"  Winner Accuracy: {results['bk_accuracy']:.1f}%")
        print()

    if results['bk_mae_vs_vegas'] is not None:
        print(f"Ball Knower vs Vegas:")
        print(f"  MAE vs Vegas Spread: {results['bk_mae_vs_vegas']:.2f} points")
        if results.get('bk_ats_accuracy'):
            print(f"  ATS Accuracy: {results['bk_ats_accuracy']:.1f}%")
        print()

    if results.get('bk_mae_vs_pt_consensus') is not None:
        print(f"Ball Knower vs PredictionTracker Consensus:")
        print(f"  MAE vs PT Average: {results['bk_mae_vs_pt_consensus']:.2f} points")
        print()

    print(f"{'='*60}\n")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Ball Knower predictions against PredictionTracker data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark 2025 Week 11 with auto-detected file
  python src/run_predictiontracker_benchmarks.py --season 2025 --week 11

  # Specify custom CSV path
  python src/run_predictiontracker_benchmarks.py --csv data/predictiontracker/custom.csv --season 2025 --week 11

  # Save detailed output
  python src/run_predictiontracker_benchmarks.py --season 2025 --week 11 --output results/pt_benchmark_w11.csv
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        default='data/predictiontracker/nflpredictions.csv',
        help='Path to PredictionTracker CSV file (default: data/predictiontracker/nflpredictions.csv)'
    )

    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help='NFL season year (default: 2025)'
    )

    parser.add_argument(
        '--week',
        type=int,
        default=None,
        help='Week number to filter (optional, analyzes all weeks if not specified)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for detailed results CSV (optional)'
    )

    parser.add_argument(
        '--no-bk',
        action='store_true',
        help='Skip loading Ball Knower predictions (just analyze PredictionTracker data)'
    )

    args = parser.parse_args()

    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: PredictionTracker CSV not found: {csv_path}")
        print(f"\nPlease download the CSV manually from:")
        print(f"  https://www.thepredictiontracker.com/nflpredictions.csv")
        print(f"\nAnd save it to: {csv_path}")
        return 1

    # Load PredictionTracker data
    print(f"Loading PredictionTracker data from: {csv_path}")
    pt_data = load_predictiontracker_csv(csv_path, season=args.season, week=args.week)

    if len(pt_data) == 0:
        print(f"ERROR: No data found for {args.season} Week {args.week}")
        return 1

    # Load Ball Knower predictions
    bk_predictions = None
    if not args.no_bk:
        bk_predictions = load_ball_knower_predictions(args.season, args.week)

    # Run benchmark
    results = run_benchmark(pt_data, bk_predictions, output_path=args.output)

    # Print summary
    print_summary(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
