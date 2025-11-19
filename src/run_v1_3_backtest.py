#!/usr/bin/env python3
"""
CLI script to train and backtest the v1.3 score prediction model.

This script:
1. Builds training data from v1.2 features
2. Splits into train/val/test sets (temporal)
3. Trains Ridge models for home_score and away_score
4. Evaluates on test set
5. Prints comprehensive metrics
6. Optionally saves the trained model

Usage:
    python src/run_v1_3_backtest.py
    python src/run_v1_3_backtest.py --save-model output/v1_3/model
"""

import argparse
import json
from pathlib import Path

from ball_knower.modeling.v1_3.training_template import (
    build_training_frame,
    split_train_val_test,
    train_v1_3
)
from ball_knower.modeling.v1_3.score_model_template import ScorePredictionModelV13
from ball_knower.modeling.v1_3.backtest_template import backtest_v1_3


def main():
    """Run v1.3 training and backtesting pipeline."""
    parser = argparse.ArgumentParser(description='Train and backtest v1.3 score prediction model')
    parser.add_argument(
        '--save-model',
        type=str,
        default=None,
        help='Path to save trained model artifacts (default: None, no save)'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save backtest results JSON (default: None, no save)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='ridge',
        choices=['ridge', 'linear'],
        help='Type of model to train (default: ridge)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Regularization parameter for Ridge (default: 1.0)'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BK v1.3 Score Prediction Model - Training and Backtest")
    print("=" * 80)

    # Build training data
    print("\n[1/5] Building training data from v1.2 features...")
    df = build_training_frame()
    print(f"  Loaded {len(df)} games from {df['season'].min()}-{df['season'].max()}")

    # Split into train/val/test
    print("\n[2/5] Splitting data (temporal)...")
    train_df, val_df, test_df = split_train_val_test(df)
    print(f"  Train: {len(train_df)} games (seasons {sorted(train_df['season'].unique())})")
    print(f"  Val:   {len(val_df)} games (seasons {sorted(val_df['season'].unique())})")
    print(f"  Test:  {len(test_df)} games (seasons {sorted(test_df['season'].unique())})")

    # Train models
    print(f"\n[3/5] Training {args.model_type} models...")
    hyperparams = {'alpha': args.alpha} if args.model_type == 'ridge' else {}
    results = train_v1_3(
        train_df=train_df,
        val_df=val_df,
        model_type=args.model_type,
        hyperparams=hyperparams,
        save_path=args.save_model
    )

    # Create model instance
    model = ScorePredictionModelV13(
        home_model=results['home_model'],
        away_model=results['away_model'],
        feature_names=results['feature_names'],
        metadata=results['training_metadata']
    )

    # Backtest on test set
    print("\n[4/5] Running backtest on test set...")
    backtest_results = backtest_v1_3(
        model=model,
        test_df=test_df,
        compute_derived_metrics=True,
        stratify_by=['season']
    )

    # Print results
    print("\n[5/5] Backtest Results Summary")
    print("=" * 80)
    print("\nScore Prediction Metrics:")
    print(f"  Home Score MAE:  {backtest_results['score_metrics']['mae_home_score']:.2f}")
    print(f"  Home Score RMSE: {backtest_results['score_metrics']['rmse_home_score']:.2f}")
    print(f"  Home Score R²:   {backtest_results['score_metrics']['r2_home_score']:.3f}")
    print(f"  Away Score MAE:  {backtest_results['score_metrics']['mae_away_score']:.2f}")
    print(f"  Away Score RMSE: {backtest_results['score_metrics']['rmse_away_score']:.2f}")
    print(f"  Away Score R²:   {backtest_results['score_metrics']['r2_away_score']:.3f}")

    print("\nDerived Metrics (Spread and Total):")
    dm = backtest_results['derived_metrics']
    print(f"  Spread MAE:  {dm['mae_spread']:.2f}")
    print(f"  Spread RMSE: {dm['rmse_spread']:.2f}")
    print(f"  Total MAE:   {dm['mae_total']:.2f}")
    print(f"  Total RMSE:  {dm['rmse_total']:.2f}")

    print("\nAccuracy Within Thresholds:")
    print(f"  Spread ±3 pts:  {dm['spread_accuracy_3pt']:.1f}%")
    print(f"  Spread ±7 pts:  {dm['spread_accuracy_7pt']:.1f}%")
    print(f"  Total ±3 pts:   {dm['total_accuracy_3pt']:.1f}%")
    print(f"  Total ±7 pts:   {dm['total_accuracy_7pt']:.1f}%")

    # Per-season breakdown
    if backtest_results.get('stratified_results'):
        print("\nPer-Season Breakdown:")
        for season, metrics in sorted(backtest_results['stratified_results']['season'].items()):
            print(f"  {season}: Home MAE={metrics['mae_home_score']:.2f}, "
                  f"Away MAE={metrics['mae_away_score']:.2f}")

    # Save results if requested
    if args.save_results:
        print(f"\nSaving results to {args.save_results}...")
        results_path = Path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare results for JSON (exclude DataFrame)
        save_data = {
            'score_metrics': backtest_results['score_metrics'],
            'derived_metrics': backtest_results['derived_metrics'],
            'stratified_results': backtest_results['stratified_results'],
            'summary': backtest_results['summary'],
            'training_metadata': results['training_metadata']
        }

        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"  Results saved successfully.")

    print("\n" + "=" * 80)
    print("Backtest complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
