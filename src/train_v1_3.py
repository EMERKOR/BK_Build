"""
Train Ball Knower v1.3 Model

Trains a logistic regression model using v1.3 dataset (v1.2 features + team form).

Model Target:
- Predicts probability that home team covers Vegas spread

Features:
- All v1.2 features (nfelo, rest, division, surface, time zone, QB)
- v1.3 form features (rolling 4-game offensive/defensive efficiency)

Output:
- Trained model saved to models/v1_3/model.pkl
- Feature names saved to models/v1_3/features.json
- Training metrics saved to models/v1_3/metrics.json
"""

import sys
import json
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.datasets import v1_3
from ball_knower.utils import paths, version


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare training data from v1.3 dataset.

    Args:
        df: v1.3 dataset DataFrame

    Returns:
        tuple: (X, y, feature_names) where:
            X: Feature matrix (numpy array)
            y: Target vector (continuous: vegas closing spread)
            feature_names: List of feature column names

    Note:
        For v1.3 training, we predict the Vegas closing spread directly
        rather than classification (cover/no cover) since actual scores
        may not be available in the training data source.
    """
    # Define feature columns (v1.2 + v1.3 form features)
    feature_cols = [
        # v1.2 features
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
        # v1.3 form features
        'offense_form_epa_diff',
        'offense_form_success_diff',
        'defense_form_epa_diff',
        'defense_form_success_diff',
    ]

    # Filter to rows with no missing feature values
    required_cols = feature_cols + ['vegas_closing_spread']
    df_clean = df[required_cols].dropna()

    # Target: Predict Vegas closing spread
    X = df_clean[feature_cols].values
    y = df_clean['vegas_closing_spread'].values

    return X, y, feature_cols


def train_model(X, y):
    """
    Train Ridge regression model to predict Vegas spread.

    Args:
        X: Feature matrix
        y: Target vector (Vegas closing spreads)

    Returns:
        tuple: (model, metrics) where metrics is a dict with training stats
    """
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Ridge regression
    model = Ridge(
        alpha=1.0,  # L2 regularization strength
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'n_features': int(X.shape[1])
    }

    return model, metrics


def save_model(model, feature_names, metrics, output_dir):
    """
    Save trained model and metadata.

    Args:
        model: Trained sklearn model
        feature_names: List of feature names
        metrics: Training metrics dict
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Save feature names
    features_path = output_dir / 'features.json'
    with open(features_path, 'w') as f:
        json.dump({
            'features': feature_names,
            'n_features': len(feature_names)
        }, f, indent=2)
    print(f"✓ Features saved to: {features_path}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")


def main():
    """
    Main training function.
    """
    # Print version banner
    version.print_version_banner("train_v1_3", model_version="v1.3")

    print("\nBall Knower v1.3 Training")
    print("=" * 80)

    # Suppress v1.3 warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)

        # Load v1.3 dataset
        print("\n[1/4] Loading v1.3 dataset (2013-2024)...")
        df = v1_3.build_training_frame(start_year=2013, end_year=2024)
        print(f"  ✓ Loaded {len(df)} games")

        # Prepare training data
        print("\n[2/4] Preparing training data...")
        X, y, feature_names = prepare_training_data(df)
        print(f"  ✓ Prepared {len(X)} samples with {len(feature_names)} features")
        print(f"  ✓ Target range: {y.min():.1f} to {y.max():.1f} (Vegas spread)")

        # Train model
        print("\n[3/4] Training Ridge regression model...")
        model, metrics = train_model(X, y)
        print(f"  ✓ Training complete")
        print(f"    - Train MAE: {metrics['train_mae']:.3f} points")
        print(f"    - Test MAE: {metrics['test_mae']:.3f} points")
        print(f"    - Train RMSE: {metrics['train_rmse']:.3f} points")
        print(f"    - Test RMSE: {metrics['test_rmse']:.3f} points")
        print(f"    - Test R²: {metrics['test_r2']:.3f}")

        # Save model
        print("\n[4/4] Saving model artifacts...")
        output_dir = paths.get_models_dir("v1.3")
        save_model(model, feature_names, metrics, output_dir)

        print("\n" + "=" * 80)
        print("✓ v1.3 model training complete!")
        print(f"  Model directory: {output_dir}")
        print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
