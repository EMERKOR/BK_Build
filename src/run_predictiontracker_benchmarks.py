#!/usr/bin/env python3
"""
PredictionTracker Benchmarks

Compare various prediction sources (models, experts, aggregators) against actual outcomes.
Calculates performance metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- ATS Accuracy (Against the Spread win rate)
- ROI (Return on Investment)
- Calibration metrics

Expected CSV format:
- game_id: Unique game identifier
- season: Year
- week: Week number
- gameday: Date
- home_team: Home team abbreviation
- away_team: Away team abbreviation
- home_score: Actual home score (if known)
- away_score: Actual away score (if known)
- spread_line: Vegas spread (from home perspective)
- predictor_spread_X: Predicted spread from predictor X
- [Additional columns for other predictors]

Usage:
    python src/run_predictiontracker_benchmarks.py \\
        --pt_csv data/external/predictiontracker_nfl_2024_sample.csv \\
        --output_dir data/benchmarks \\
        --outlier_threshold 4.0
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


def load_predictiontracker_data(csv_path):
    """
    Load PredictionTracker CSV data.

    Args:
        csv_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"✓ Loaded {len(df)} games")
    print(f"  Columns: {', '.join(df.columns)}")

    return df


def calculate_actual_margin(df):
    """
    Calculate actual game margin (home - away) from home perspective.

    Args:
        df (pd.DataFrame): Games dataframe

    Returns:
        pd.DataFrame: DataFrame with actual_margin column
    """
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['actual_margin'] = df['home_score'] - df['away_score']
        print(f"✓ Calculated actual margins for {df['actual_margin'].notna().sum()} games")
    else:
        print("⚠ Warning: No home_score/away_score columns found")
        df['actual_margin'] = np.nan

    return df


def identify_predictor_columns(df):
    """
    Identify all predictor columns in the dataframe.

    Args:
        df (pd.DataFrame): Games dataframe

    Returns:
        list: List of predictor column names
    """
    # Look for columns that end with common patterns or contain prediction-related keywords
    predictor_patterns = [
        'predictor_spread_',
        '_prediction',
        '_spread',
        '_line',
        'model_',
        'expert_'
    ]

    # Exclude known non-predictor columns
    exclude_patterns = ['spread_line', 'home_', 'away_', 'actual_', 'game_', 'season', 'week']

    predictors = []
    for col in df.columns:
        # Check if column matches predictor patterns
        is_predictor = any(pattern in col.lower() for pattern in predictor_patterns)

        # Check if it's NOT an excluded column
        is_excluded = any(pattern in col.lower() for pattern in exclude_patterns)

        if is_predictor and not is_excluded:
            predictors.append(col)

    # If no predictors found with patterns, look for numeric columns
    if not predictors:
        print("⚠ No predictor columns found with standard patterns")
        print("  Looking for numeric columns that might be predictions...")

        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                is_excluded = any(pattern in col.lower() for pattern in exclude_patterns)
                if not is_excluded and col not in ['season', 'week', 'home_score', 'away_score']:
                    predictors.append(col)

    print(f"\n✓ Identified {len(predictors)} predictor(s):")
    for pred in predictors:
        print(f"  - {pred}")

    return predictors


def calculate_prediction_metrics(actual, predicted, predictor_name="Model", vegas_line=None):
    """
    Calculate comprehensive prediction metrics.

    Args:
        actual (np.array): Actual game margins
        predicted (np.array): Predicted spreads
        predictor_name (str): Name of predictor
        vegas_line (np.array): Vegas spread lines (optional)

    Returns:
        dict: Dictionary of metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {
            'predictor': predictor_name,
            'n_games': 0,
            'mae': np.nan,
            'rmse': np.nan,
            'mean_error': np.nan,
            'median_error': np.nan
        }

    # Basic error metrics
    errors = predicted - actual
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    metrics = {
        'predictor': predictor_name,
        'n_games': len(actual),
        'mae': mae,
        'rmse': rmse,
        'mean_error': mean_error,  # Bias (positive = tends to over-predict home)
        'median_error': median_error,
        'std_error': np.std(errors)
    }

    # ATS metrics (Against the Spread)
    # Home covers if: actual_margin + spread_line > 0
    # We predict home covers if: predicted > 0 (home wins by more than predicted line)
    # For simplicity, use 0 as threshold (pick winner)

    actual_home_covered = actual > 0  # Home team won
    predicted_home_covered = predicted > 0  # Prediction favors home

    ats_correct = (actual_home_covered == predicted_home_covered)
    ats_accuracy = np.mean(ats_correct)

    metrics['ats_accuracy'] = ats_accuracy
    metrics['ats_correct'] = ats_correct.sum()

    # Vegas comparison (if available)
    if vegas_line is not None:
        vegas_line = vegas_line[mask]

        # Calculate edge vs Vegas
        edge = predicted - vegas_line
        metrics['mean_edge'] = np.mean(np.abs(edge))
        metrics['max_edge'] = np.max(np.abs(edge))

        # ROI analysis (simplified: did we beat Vegas?)
        vegas_errors = np.abs(vegas_line - actual)
        model_errors = abs_errors

        beat_vegas = model_errors < vegas_errors
        metrics['beat_vegas_pct'] = np.mean(beat_vegas)
        metrics['beat_vegas_games'] = beat_vegas.sum()

    # Calibration: Are large predictions accurate?
    # Bin by prediction magnitude
    pred_abs = np.abs(predicted)
    bins = [0, 3, 7, 14, 100]  # Small, medium, large, huge spreads

    for i in range(len(bins) - 1):
        bin_mask = (pred_abs >= bins[i]) & (pred_abs < bins[i+1])
        if bin_mask.sum() > 0:
            bin_mae = np.mean(abs_errors[bin_mask])
            metrics[f'mae_spread_{bins[i]}_{bins[i+1]}'] = bin_mae

    return metrics


def detect_and_remove_outliers(df, predictor_cols, threshold=4.0):
    """
    Detect and remove outlier predictions using IQR method.

    Args:
        df (pd.DataFrame): Games dataframe
        predictor_cols (list): List of predictor column names
        threshold (float): IQR multiplier for outlier detection

    Returns:
        pd.DataFrame: DataFrame with outliers removed
        pd.DataFrame: DataFrame of outlier games
    """
    outlier_mask = pd.Series(False, index=df.index)

    for col in predictor_cols:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outliers

        if col_outliers.sum() > 0:
            print(f"  {col}: {col_outliers.sum()} outliers detected")

    outliers_df = df[outlier_mask].copy()
    clean_df = df[~outlier_mask].copy()

    print(f"\n✓ Removed {len(outliers_df)} outlier games (threshold={threshold})")
    print(f"  Remaining: {len(clean_df)} games")

    return clean_df, outliers_df


def benchmark_all_predictors(df, predictor_cols, vegas_col='spread_line'):
    """
    Benchmark all predictors in the dataset.

    Args:
        df (pd.DataFrame): Games dataframe with predictions
        predictor_cols (list): List of predictor column names
        vegas_col (str): Name of Vegas line column

    Returns:
        pd.DataFrame: Benchmark results for all predictors
    """
    print("\n" + "="*80)
    print("BENCHMARKING PREDICTORS")
    print("="*80 + "\n")

    results = []

    # Get actual margins and Vegas line
    actual = df['actual_margin'].values
    vegas_line = df[vegas_col].values if vegas_col in df.columns else None

    # Benchmark Vegas first (if available)
    if vegas_line is not None and not np.all(np.isnan(vegas_line)):
        print("Benchmarking Vegas spread_line...")
        vegas_metrics = calculate_prediction_metrics(
            actual, vegas_line,
            predictor_name="Vegas",
            vegas_line=None  # Don't compare Vegas to itself
        )
        results.append(vegas_metrics)
        print(f"  MAE: {vegas_metrics['mae']:.3f}, RMSE: {vegas_metrics['rmse']:.3f}, "
              f"ATS: {vegas_metrics['ats_accuracy']:.3f}")

    # Benchmark each predictor
    for pred_col in predictor_cols:
        if pred_col not in df.columns:
            print(f"⚠ Warning: {pred_col} not found in dataframe")
            continue

        print(f"\nBenchmarking {pred_col}...")
        predicted = df[pred_col].values

        metrics = calculate_prediction_metrics(
            actual, predicted,
            predictor_name=pred_col,
            vegas_line=vegas_line
        )
        results.append(metrics)

        print(f"  MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, "
              f"ATS: {metrics['ats_accuracy']:.3f}")

        if 'beat_vegas_pct' in metrics:
            print(f"  Beat Vegas: {metrics['beat_vegas_pct']:.1%} "
                  f"({metrics['beat_vegas_games']}/{metrics['n_games']} games)")

    results_df = pd.DataFrame(results)

    # Sort by MAE (best first)
    results_df = results_df.sort_values('mae')

    print("\n" + "="*80)
    print("✓ BENCHMARKING COMPLETE")
    print("="*80)

    return results_df


def save_results(results_df, output_dir, timestamp=None):
    """
    Save benchmark results to CSV and JSON.

    Args:
        results_df (pd.DataFrame): Benchmark results
        output_dir (str): Output directory path
        timestamp (str): Timestamp string (optional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_path = output_path / f"predictiontracker_benchmarks_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to: {csv_path}")

    # Save JSON (more detailed)
    json_path = output_path / f"predictiontracker_benchmarks_{timestamp}.json"
    results_dict = {
        'timestamp': timestamp,
        'n_predictors': len(results_df),
        'results': results_df.to_dict('records')
    }

    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"✓ Saved detailed results to: {json_path}")

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by MAE)")
    print("="*80)

    # Select key columns for display
    display_cols = ['predictor', 'n_games', 'mae', 'rmse', 'ats_accuracy', 'mean_error']
    if 'beat_vegas_pct' in results_df.columns:
        display_cols.append('beat_vegas_pct')

    print(results_df[display_cols].to_string(index=False))
    print("="*80)


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description='Run PredictionTracker benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--pt_csv',
        type=str,
        required=True,
        help='Path to PredictionTracker CSV file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/benchmarks',
        help='Output directory for benchmark results (default: data/benchmarks)'
    )

    parser.add_argument(
        '--outlier_threshold',
        type=float,
        default=4.0,
        help='IQR multiplier for outlier detection (default: 4.0, use 0 to disable)'
    )

    parser.add_argument(
        '--vegas_col',
        type=str,
        default='spread_line',
        help='Name of Vegas spread column (default: spread_line)'
    )

    parser.add_argument(
        '--no_outlier_removal',
        action='store_true',
        help='Disable outlier removal'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*80)
    print("PREDICTIONTRACKER BENCHMARKS")
    print("="*80)
    print(f"\nInput CSV: {args.pt_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Outlier threshold: {args.outlier_threshold if not args.no_outlier_removal else 'Disabled'}")

    # Load data
    df = load_predictiontracker_data(args.pt_csv)

    # Calculate actual margins
    df = calculate_actual_margin(df)

    # Identify predictor columns
    predictor_cols = identify_predictor_columns(df)

    if len(predictor_cols) == 0:
        print("\n❌ ERROR: No predictor columns found in CSV")
        print("   Expected columns like: predictor_spread_X, model_X, expert_X, etc.")
        print(f"   Available columns: {', '.join(df.columns)}")
        return 1

    # Remove outliers (optional)
    if not args.no_outlier_removal and args.outlier_threshold > 0:
        print(f"\nDetecting outliers (threshold={args.outlier_threshold})...")
        df, outliers_df = detect_and_remove_outliers(df, predictor_cols, args.outlier_threshold)

        # Save outliers for review
        if len(outliers_df) > 0:
            outliers_path = Path(args.output_dir) / 'outliers.csv'
            outliers_path.parent.mkdir(parents=True, exist_ok=True)
            outliers_df.to_csv(outliers_path, index=False)
            print(f"✓ Saved outliers to: {outliers_path}")

    # Run benchmarks
    results_df = benchmark_all_predictors(df, predictor_cols, vegas_col=args.vegas_col)

    # Save results
    save_results(results_df, args.output_dir)

    print("\n✓ Benchmark complete!\n")
    return 0


if __name__ == '__main__':
    exit(main())
