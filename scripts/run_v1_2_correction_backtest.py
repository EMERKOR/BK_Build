#!/usr/bin/env python3
"""
Ball Knower v1.2 Correction Model - Backtest Script

Trains the v1.2 spread correction model on a range of training weeks and
evaluates performance on test weeks.

Usage:
    python scripts/run_v1_2_correction_backtest.py --season 2024 --train-weeks 1-10 --test-weeks 11-12
    python scripts/run_v1_2_correction_backtest.py --season 2024 --train-weeks 1-10 --test-weeks 11-12 --edge-thresholds 0.5,1.0,2.0
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Ball Knower modules
from ball_knower.io import loaders, feature_maps
from ball_knower.models.v1_2_correction import SpreadCorrectionModel
from src.models import DeterministicSpreadModel
from src.nflverse_data import nflverse
from src import config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def parse_week_range(week_range_str: str) -> List[int]:
    """
    Parse week range string like '1-10' or '11,12,13' into list of week numbers.

    Args:
        week_range_str: String like '1-10' or '11,12,13'

    Returns:
        List of week numbers
    """
    if '-' in week_range_str:
        start, end = week_range_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(w.strip()) for w in week_range_str.split(',')]


def load_week_matchups(season: int, week: int) -> pd.DataFrame:
    """
    Load matchup data for a specific week with canonical features.

    Args:
        season: NFL season year
        week: Week number

    Returns:
        DataFrame with matchup features and Vegas lines
    """
    # Load game data from nflverse (contains spreads)
    games = nflverse.games(season=season, week=week)

    # Filter to games with spread lines
    games = games[games['spread_line'].notna()].copy()

    if len(games) == 0:
        return pd.DataFrame()

    # Load ratings data via unified loader
    all_data = loaders.load_all_sources(season=season, week=week)

    # Get canonical feature view
    canonical_ratings = feature_maps.get_canonical_features(all_data['merged_ratings'])

    # Compute feature differentials for matchups
    matchups = feature_maps.get_feature_differential(
        canonical_ratings,
        games['home_team'],
        games['away_team'],
        features=['overall_rating', 'epa_margin', 'offensive_rating', 'defensive_rating', 'qb_adjustment']
    )

    # Add game metadata
    matchups['season'] = season
    matchups['week'] = week
    matchups['game_id'] = games['game_id'].values
    matchups['vegas_line'] = games['spread_line'].values

    # Add actual margins if available (for ATS analysis)
    if 'home_score' in games.columns and 'away_score' in games.columns:
        matchups['actual_margin'] = (games['home_score'] - games['away_score']).values

    return matchups


def train_correction_model(
    season: int,
    train_weeks: List[int],
    alpha: float = 10.0,
    verbose: bool = True
) -> Tuple[SpreadCorrectionModel, pd.DataFrame]:
    """
    Train v1.2 correction model on specified training weeks.

    Args:
        season: NFL season year
        train_weeks: List of week numbers to train on
        alpha: Ridge regression regularization parameter
        verbose: Whether to print training info

    Returns:
        Tuple of (trained model, training data DataFrame)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING v1.2 CORRECTION MODEL")
        print(f"{'='*80}")
        print(f"Season: {season}")
        print(f"Training weeks: {min(train_weeks)}-{max(train_weeks)}")
        print(f"Regularization (alpha): {alpha}")

    # Load all training weeks
    train_data_list = []
    for week in train_weeks:
        if verbose:
            print(f"  Loading week {week}...", end=' ')

        week_matchups = load_week_matchups(season, week)

        if len(week_matchups) > 0:
            train_data_list.append(week_matchups)
            if verbose:
                print(f"✓ {len(week_matchups)} games")
        else:
            if verbose:
                print("✗ No data available")

    # Combine all training data
    train_data = pd.concat(train_data_list, ignore_index=True)

    if verbose:
        print(f"\nTotal training samples: {len(train_data)}")

    # Initialize base model (deterministic v1.0)
    base_model = DeterministicSpreadModel(hfa=config.HOME_FIELD_ADVANTAGE)

    # Initialize correction model
    correction_model = SpreadCorrectionModel(
        base_model=base_model,
        alpha=alpha,
        fit_intercept=True,
        normalize_features=True
    )

    # Extract Vegas lines for training
    vegas_lines = train_data['vegas_line'].values

    # Train correction model
    correction_model.fit(train_data, vegas_lines, verbose=verbose)

    # Display feature importance
    if verbose:
        print(f"\nFEATURE IMPORTANCE (Ridge Coefficients):")
        print(f"{'-'*80}")
        importance = correction_model.get_feature_importance()
        for feature, coef in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature:30s}: {coef:+.4f}")
        print(f"{'-'*80}")

    return correction_model, train_data


def evaluate_correction_model(
    model: SpreadCorrectionModel,
    season: int,
    test_weeks: List[int],
    edge_thresholds: List[float] = [0.5, 1.0, 2.0, 3.0],
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate correction model on test weeks.

    Args:
        model: Trained SpreadCorrectionModel
        season: NFL season year
        test_weeks: List of week numbers to test on
        edge_thresholds: List of edge thresholds for ATS analysis
        verbose: Whether to print evaluation info

    Returns:
        Tuple of (predictions DataFrame, ATS summary DataFrame)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"EVALUATING v1.2 CORRECTION MODEL")
        print(f"{'='*80}")
        print(f"Season: {season}")
        print(f"Test weeks: {min(test_weeks)}-{max(test_weeks)}")

    # Load all test weeks
    test_data_list = []
    for week in test_weeks:
        if verbose:
            print(f"  Loading week {week}...", end=' ')

        week_matchups = load_week_matchups(season, week)

        if len(week_matchups) > 0:
            test_data_list.append(week_matchups)
            if verbose:
                print(f"✓ {len(week_matchups)} games")
        else:
            if verbose:
                print("✗ No data available")

    # Combine all test data
    test_data = pd.concat(test_data_list, ignore_index=True)

    if verbose:
        print(f"\nTotal test samples: {len(test_data)}")

    # Generate base predictions
    base_predictions = model.predict_base(test_data)

    # Generate corrected predictions
    corrected_predictions = model.predict(test_data)

    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        'season': test_data['season'],
        'week': test_data['week'],
        'game_id': test_data['game_id'],
        'away_team': test_data['away_team'],
        'home_team': test_data['home_team'],
        'vegas_line': test_data['vegas_line'],
        'bk_v1_base': base_predictions,
        'bk_v1_2_corrected': corrected_predictions,
        'correction': corrected_predictions - base_predictions,
        'edge': corrected_predictions - test_data['vegas_line'].values
    })

    # Add actual margins if available
    if 'actual_margin' in test_data.columns:
        predictions_df['actual_margin'] = test_data['actual_margin'].values

    # Calculate overall metrics
    if verbose:
        print(f"\n{'='*80}")
        print(f"OVERALL METRICS")
        print(f"{'='*80}")

        base_mae = np.mean(np.abs(base_predictions - test_data['vegas_line'].values))
        corrected_mae = np.mean(np.abs(corrected_predictions - test_data['vegas_line'].values))

        base_rmse = np.sqrt(np.mean((base_predictions - test_data['vegas_line'].values) ** 2))
        corrected_rmse = np.sqrt(np.mean((corrected_predictions - test_data['vegas_line'].values) ** 2))

        print(f"\nBase Model (v1.0):")
        print(f"  MAE:  {base_mae:.3f} points")
        print(f"  RMSE: {base_rmse:.3f} points")

        print(f"\nCorrected Model (v1.2):")
        print(f"  MAE:  {corrected_mae:.3f} points")
        print(f"  RMSE: {corrected_rmse:.3f} points")

        improvement = base_mae - corrected_mae
        print(f"\nImprovement: {improvement:.3f} points ({improvement/base_mae*100:.1f}%)")

    # ATS analysis (if actual margins available)
    ats_summary = None
    if 'actual_margin' in predictions_df.columns:
        ats_results = []

        for threshold in edge_thresholds:
            # Find games where model has edge >= threshold
            games_with_edge = predictions_df[predictions_df['edge'].abs() >= threshold].copy()

            if len(games_with_edge) > 0:
                # Model picks home when edge < 0 (negative = home favored)
                games_with_edge['model_picks_home'] = games_with_edge['edge'] < 0

                # Home covers when actual + vegas > 0
                games_with_edge['home_covered'] = (games_with_edge['actual_margin'] + games_with_edge['vegas_line']) > 0

                # Calculate ATS accuracy
                correct = (games_with_edge['model_picks_home'] == games_with_edge['home_covered']).sum()
                total = len(games_with_edge)
                accuracy = correct / total

                ats_results.append({
                    'edge_threshold': threshold,
                    'games': total,
                    'correct': int(correct),
                    'accuracy': accuracy,
                    'roi_estimate': (accuracy - 0.524) * 100  # Assuming -110 odds
                })

        ats_summary = pd.DataFrame(ats_results)

        if verbose and len(ats_summary) > 0:
            print(f"\n{'='*80}")
            print(f"ATS PERFORMANCE BY EDGE THRESHOLD")
            print(f"{'='*80}\n")
            print(ats_summary.to_string(index=False))

    return predictions_df, ats_summary


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate Ball Knower v1.2 spread correction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on weeks 1-10, test on weeks 11-12 of 2024 season
  python scripts/run_v1_2_correction_backtest.py --season 2024 --train-weeks 1-10 --test-weeks 11-12

  # Custom edge thresholds for ATS analysis
  python scripts/run_v1_2_correction_backtest.py --season 2024 --train-weeks 1-10 --test-weeks 11-12 --edge-thresholds 0.5,1.0,2.0,3.0

  # Train on multiple non-consecutive weeks
  python scripts/run_v1_2_correction_backtest.py --season 2024 --train-weeks 1,2,3,5,6,7 --test-weeks 11,12
        """
    )

    parser.add_argument(
        '--season',
        type=int,
        required=True,
        help='NFL season year (e.g., 2024)'
    )

    parser.add_argument(
        '--train-weeks',
        type=str,
        required=True,
        help='Training weeks range or list (e.g., "1-10" or "1,2,3,5")'
    )

    parser.add_argument(
        '--test-weeks',
        type=str,
        required=True,
        help='Test weeks range or list (e.g., "11-12" or "11,12,13")'
    )

    parser.add_argument(
        '--edge-thresholds',
        type=str,
        default='0.5,1.0,2.0,3.0',
        help='Edge thresholds for ATS analysis (comma-separated, e.g., "0.5,1.0,2.0")'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=10.0,
        help='Ridge regression regularization strength (default: 10.0)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: output/v1_2_predictions_{season}_weeks_{test_weeks}.csv)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Parse week ranges
    train_weeks = parse_week_range(args.train_weeks)
    test_weeks = parse_week_range(args.test_weeks)
    edge_thresholds = [float(t.strip()) for t in args.edge_thresholds.split(',')]

    verbose = not args.quiet

    # Print header
    if verbose:
        print("\n" + "="*80)
        print("BALL KNOWER v1.2 CORRECTION MODEL - BACKTEST")
        print("="*80)

    # Train model
    model, train_data = train_correction_model(
        season=args.season,
        train_weeks=train_weeks,
        alpha=args.alpha,
        verbose=verbose
    )

    # Evaluate model
    predictions_df, ats_summary = evaluate_correction_model(
        model=model,
        season=args.season,
        test_weeks=test_weeks,
        edge_thresholds=edge_thresholds,
        verbose=verbose
    )

    # Save predictions to CSV
    if args.output:
        output_path = Path(args.output)
    else:
        test_weeks_str = f"{min(test_weeks)}-{max(test_weeks)}" if len(test_weeks) > 1 else str(test_weeks[0])
        output_path = config.get_output_path(f'v1_2_predictions_{args.season}_weeks_{test_weeks_str}.csv')

    predictions_df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED")
        print(f"{'='*80}")
        print(f"\nPredictions saved to: {output_path}")

        if ats_summary is not None and len(ats_summary) > 0:
            ats_output_path = output_path.parent / output_path.name.replace('.csv', '_ats_summary.csv')
            ats_summary.to_csv(ats_output_path, index=False)
            print(f"ATS summary saved to: {ats_output_path}")

        print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
