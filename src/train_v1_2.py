#!/usr/bin/env python3
"""
Ball Knower v1.2 Training Script

Trains the v1.2 two-stage model:
1. v1.1 base model (EnhancedSpreadModel)
2. v1.2 ML correction layer (MLCorrectionModel)

The v1.2 model learns to predict Vegas spreads from NFElo features,
then uses an ML model to correct residual errors.

Usage:
    python src/train_v1_2.py --start-year 2009 --end-year 2024
    python src/train_v1_2.py --start-year 2015 --end-year 2023 --output custom_model_dir/
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.datasets import v1_2
from ball_knower.modeling import models
from src import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Ball Knower v1.2 model on historical data'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=2009,
        help='Start year for training data (default: 2009)'
    )

    parser.add_argument(
        '--end-year',
        type=int,
        default=2024,
        help='End year for training data (default: 2024)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for model artifacts (default: output/models/v1_2/)'
    )

    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    print("\n" + "="*80)
    print("BALL KNOWER v1.2 TRAINING")
    print("="*80)
    print(f"\nTraining years: {args.start_year}-{args.end_year}")
    print(f"Test split: {args.test_split:.0%}")

    # =========================================================================
    # STEP 1: LOAD TRAINING DATA
    # =========================================================================

    print("\n[1/5] Loading training data from v1.2 dataset builder...")
    training_frame = v1_2.build_training_frame(
        start_year=args.start_year,
        end_year=args.end_year
    )

    print(f"  Loaded {len(training_frame):,} games")
    print(f"  Features: {list(training_frame.columns)}")

    # Split into train/test by time (last N% for testing)
    split_idx = int(len(training_frame) * (1 - args.test_split))
    train_df = training_frame.iloc[:split_idx].copy()
    test_df = training_frame.iloc[split_idx:].copy()

    print(f"\n  Train set: {len(train_df):,} games ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"  Test set: {len(test_df):,} games ({test_df['season'].min()}-{test_df['season'].max()})")

    # =========================================================================
    # STEP 2: INSTANTIATE MODELS
    # =========================================================================

    print("\n[2/5] Instantiating models...")

    # Base model: v1.1 EnhancedSpreadModel
    base_model = models.EnhancedSpreadModel(use_calibrated=False)
    print("  ✓ Created v1.1 base model (EnhancedSpreadModel)")

    # Correction model: v1.2 MLCorrectionModel
    correction_model = models.MLCorrectionModel(base_model=base_model)
    print("  ✓ Created v1.2 correction model (MLCorrectionModel)")

    # =========================================================================
    # STEP 3: TRAIN BASE MODEL
    # =========================================================================

    print("\n[3/5] Training base model (v1.1)...")

    # NOTE: v1.1 is deterministic and doesn't need training in the traditional sense.
    # However, we could calibrate its weights here on the training data.
    # For now, we use the default weights.

    # Generate base model predictions on training data
    print("  Generating base model predictions...")

    # The base model expects feature dictionaries, but we have a DataFrame
    # We need to construct features for each game
    train_features = []
    for idx, game in train_df.iterrows():
        train_features.append({
            'nfelo_diff': game['nfelo_diff'],
            'rest_advantage': game['rest_advantage'],
            'div_game': game['div_game'],
            'surface_mod': game['surface_mod'],
            'time_advantage': game['time_advantage'],
            'qb_diff': game['qb_diff'],
        })

    # For the ML correction model, we need the DataFrame format
    # The base model can work with DataFrames if we use predict_games()
    # But predict_games expects home/away features, not differentials

    # Instead, we'll compute base predictions directly using the v1.2 formula
    # (which is essentially what v1.1 would predict if given these features)

    print("  Computing base predictions using v1.1 logic...")

    # v1.1 base prediction (simplified - just uses nfelo and rest)
    # This is a placeholder - ideally we'd have a proper v1.1 predict method
    # that takes the v1.2 features

    # For now, use a simple linear combination (matches v1.1 logic roughly)
    base_predictions_train = (
        train_df['nfelo_diff'] * 0.04 +  # nfelo coefficient
        train_df['rest_advantage'] * 0.3 +  # rest coefficient
        train_df['qb_diff'] * 0.1  # qb coefficient
    )

    base_predictions_test = (
        test_df['nfelo_diff'] * 0.04 +
        test_df['rest_advantage'] * 0.3 +
        test_df['qb_diff'] * 0.1
    )

    # =========================================================================
    # STEP 4: TRAIN CORRECTION MODEL
    # =========================================================================

    print("\n[4/5] Training ML correction layer (v1.2)...")

    # Target: Vegas closing spread
    y_train = train_df['vegas_closing_spread'].values
    y_test = test_df['vegas_closing_spread'].values

    # Compute residuals (what the base model got wrong)
    residuals_train = y_train - base_predictions_train.values

    print(f"  Base model MAE: {np.abs(y_train - base_predictions_train.values).mean():.2f} pts")
    print(f"  Training correction model on residuals...")

    # Define feature columns for ML model
    ml_feature_cols = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff'
    ]

    # Train correction model
    from sklearn.linear_model import Ridge
    ml_model = Ridge(alpha=10.0)
    ml_model.fit(train_df[ml_feature_cols], residuals_train)

    print("  ✓ Correction model trained")

    # =========================================================================
    # STEP 5: EVALUATE AND SAVE
    # =========================================================================

    print("\n[5/5] Evaluating and saving models...")

    # Generate corrected predictions on test set
    residual_corrections_test = ml_model.predict(test_df[ml_feature_cols])
    corrected_predictions_test = base_predictions_test.values + residual_corrections_test

    # Evaluate
    base_mae = np.abs(y_test - base_predictions_test.values).mean()
    corrected_mae = np.abs(y_test - corrected_predictions_test).mean()

    print(f"\n  Test set evaluation:")
    print(f"    Base model MAE: {base_mae:.2f} pts")
    print(f"    Corrected model MAE: {corrected_mae:.2f} pts")
    print(f"    Improvement: {base_mae - corrected_mae:.2f} pts")

    # Determine output directory
    if args.output is None:
        output_dir = config.OUTPUT_DIR / 'models' / 'v1_2'
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    print(f"\n  Saving model artifacts to {output_dir}...")

    # Save base model weights
    base_model_config = {
        'hfa': base_model.hfa,
        'weights': base_model.weights,
        'model_type': 'EnhancedSpreadModel',
        'version': 'v1.1'
    }

    with open(output_dir / 'base_model_config.json', 'w') as f:
        json.dump(base_model_config, f, indent=2)

    # Save correction model (Ridge)
    with open(output_dir / 'correction_model.pkl', 'wb') as f:
        pickle.dump(ml_model, f)

    # Save feature columns
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(ml_feature_cols, f, indent=2)

    # Save metadata
    metadata = {
        'trained_on': datetime.now().isoformat(),
        'training_years': f"{args.start_year}-{args.end_year}",
        'n_train_games': len(train_df),
        'n_test_games': len(test_df),
        'base_model_mae': float(base_mae),
        'corrected_model_mae': float(corrected_mae),
        'improvement': float(base_mae - corrected_mae),
        'features': ml_feature_cols
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved base_model_config.json")
    print(f"  ✓ Saved correction_model.pkl")
    print(f"  ✓ Saved feature_columns.json")
    print(f"  ✓ Saved metadata.json")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel artifacts saved to: {output_dir}")
    print(f"\nTo use this model for predictions:")
    print(f"  python src/run_weekly_predictions.py --model v1.2 --season 2025 --week 11")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
