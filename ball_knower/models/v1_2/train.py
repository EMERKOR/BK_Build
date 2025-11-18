"""
v1.2 Model Training

Formal training script for Ball Knower v1.2 residual model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from .model import V1_2ResidualModel
from .features import validate_dataset, FEATURE_COLUMNS


def train_v1_2(
    dataset_path: Path = None,
    output_dir: Path = None,
    alpha: float = 1.0,
    test_season: int = 2025,
) -> Path:
    """
    Train v1.2 residual model on canonical dataset.

    Args:
        dataset_path: Path to v1_2_training_dataset.parquet
        output_dir: Directory to save model outputs
        alpha: Ridge regularization parameter
        test_season: Season to use as test set (all prior seasons = training)

    Returns:
        Path to saved model JSON

    Example:
        >>> model_path = train_v1_2()
        >>> print(f"Model saved to {model_path}")
    """

    # Default paths
    if dataset_path is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        dataset_path = PROJECT_ROOT / "data" / "v1_2_training_dataset.parquet"

    if output_dir is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_dir = PROJECT_ROOT / "output"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BALL KNOWER v1.2 - MODEL TRAINING")
    print("="*80)

    # ========================================================================
    # LOAD DATASET
    # ========================================================================

    print(f"\n[1/5] Loading dataset from {dataset_path}...")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    print(f"  Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

    # Validate dataset
    validate_dataset(df)

    # ========================================================================
    # TRAIN/TEST SPLIT
    # ========================================================================

    print(f"\n[2/5] Splitting train/test sets...")
    print(f"  Train: seasons < {test_season}")
    print(f"  Test:  season == {test_season}")

    # Strict time-based split: no leakage
    train_df = df[df['season'] < test_season].copy()
    test_df = df[df['season'] == test_season].copy()

    print(f"  Train set: {len(train_df):,} games")
    print(f"  Test set:  {len(test_df):,} games")

    if len(test_df) == 0:
        print(f"\n  WARNING: No test data for season {test_season}")
        print(f"  Will train on all data and skip test evaluation")
        test_df = None

    # ========================================================================
    # TRAIN MODEL
    # ========================================================================

    print(f"\n[3/5] Training Ridge model (alpha={alpha})...")

    model = V1_2ResidualModel(alpha=alpha)
    model.fit(train_df)

    # Show feature importance
    importance = model.get_feature_importance()
    print("\n  Feature importance (by |coefficient|):")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:25} {row['coefficient']:+.6f}")

    # ========================================================================
    # EVALUATE ON TEST SET
    # ========================================================================

    if test_df is not None:
        print(f"\n[4/5] Evaluating on test set...")

        test_metrics = model.evaluate(test_df, split_name='test')

        print(f"  Test MAE:  {test_metrics['mae']:.3f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.3f}")
        print(f"  Test R²:   {test_metrics['r2']:.3f}")

        # Additional analysis: spread prediction MAE
        # (residual prediction MAE vs final spread prediction MAE)
        test_df_eval = test_df.copy()
        test_df_eval['residual_pred'] = model.predict(test_df)
        test_df_eval['spread_pred'] = test_df_eval['bk_base_spread'] + test_df_eval['residual_pred']
        test_df_eval['spread_error'] = abs(test_df_eval['spread_pred'] - test_df_eval['vegas_closing_spread'])

        spread_mae = test_df_eval['spread_error'].mean()
        print(f"  Spread prediction MAE: {spread_mae:.3f} (vs Vegas line)")

    else:
        print(f"\n[4/5] Skipping test evaluation (no test data)")

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    print(f"\n[5/5] Saving model...")

    # Save model JSON
    model_path = output_dir / "v1_2_model.json"
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Save extended metadata
    metadata_path = output_dir / "v1_2_training_metadata.json"
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': str(dataset_path),
        'dataset_shape': list(df.shape),
        'train_size': len(train_df),
        'test_size': len(test_df) if test_df is not None else 0,
        'test_season': test_season,
        'alpha': alpha,
        'feature_columns': FEATURE_COLUMNS,
        'model_path': str(model_path),
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to: {metadata_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    print(model.summary())

    if test_df is not None:
        print(f"\nKey Metrics:")
        print(f"  Train MAE: {model.metrics['train']['mae']:.3f} points")
        print(f"  Test MAE:  {model.metrics['test']['mae']:.3f} points")
        print(f"  Test R²:   {model.metrics['test']['r2']:.3f}")
        print(f"  Spread prediction MAE: {spread_mae:.3f} points (vs Vegas)")
    else:
        print(f"\nKey Metrics:")
        print(f"  Train MAE: {model.metrics['train']['mae']:.3f} points")
        print(f"  Train R²:  {model.metrics['train']['r2']:.3f}")

    print("\n" + "="*80 + "\n")

    return model_path
