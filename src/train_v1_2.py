#!/usr/bin/env python3
"""
Train Ball Knower v1.2 Model

Trains a Ridge regression model using v1.2 dataset (structural features).

Model Target:
- Predicts Vegas closing spread (market consensus)

Features:
- nfelo_diff: ELO rating differential
- rest_advantage: Combined bye week effects
- div_game: Division game modifier
- surface_mod: Surface differential modifier
- time_advantage: Time zone advantage modifier
- qb_diff: QB adjustment differential

Output:
- Trained model saved to output/models/v1_2/ball_knower_v1_2_model.json
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.datasets import v1_2
from ball_knower.utils import paths, version


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare training data from v1.2 dataset.

    Args:
        df: v1.2 dataset DataFrame

    Returns:
        tuple: (X, y, feature_names) where:
            X: Feature matrix (numpy array)
            y: Target vector (vegas closing spreads)
            feature_names: List of feature column names
    """
    # Define feature columns (matching run_backtest_v1_2)
    feature_cols = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
    ]

    # Filter to rows with no missing feature values
    required_cols = feature_cols + ['vegas_closing_spread']
    df_clean = df[required_cols].dropna()

    # Target: Predict Vegas closing spread
    X = df_clean[feature_cols].values
    y = df_clean['vegas_closing_spread'].values

    return X, y, feature_cols


def train_model(X, y, alpha: float = 1.0):
    """
    Train Ridge regression model to predict Vegas spread.

    Args:
        X: Feature matrix
        y: Target vector (Vegas closing spreads)
        alpha: L2 regularization strength (default: 1.0)

    Returns:
        tuple: (model, metrics) where metrics is a dict with training stats
    """
    # Train Ridge regression
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X, y)

    # Evaluate on training data
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    metrics = {
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'alpha': float(alpha),
    }

    return model, metrics


def save_model_json(model, feature_names, metrics, output_path: Path):
    """
    Save trained model as JSON matching run_backtest_v1_2 expectations.

    Args:
        model: Trained sklearn Ridge model
        feature_names: List of feature names
        metrics: Training metrics dict
        output_path: Path to save JSON file

    Output JSON format:
        {
            "intercept": <float>,
            "coefficients": {
                "nfelo_diff": <float>,
                "rest_advantage": <float>,
                "div_game": <float>,
                "surface_mod": <float>,
                "time_advantage": <float>,
                "qb_diff": <float>
            },
            "metadata": {
                "n_samples": <int>,
                "mae": <float>,
                "rmse": <float>,
                "r2": <float>,
                "alpha": <float>
            }
        }
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build coefficients dictionary
    coefficients = {}
    for i, feature_name in enumerate(feature_names):
        coefficients[feature_name] = float(model.coef_[i])

    # Build output JSON
    model_json = {
        'intercept': float(model.intercept_),
        'coefficients': coefficients,
        'metadata': metrics
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(model_json, f, indent=2)

    print(f"✓ Model saved to: {output_path}")


def main(start_season: int = 2009, end_season: int = 2024, alpha: float = 1.0):
    """
    Main training function.

    Args:
        start_season: Start season for training (default: 2009)
        end_season: End season for training (default: 2024)
        alpha: Ridge regression regularization strength (default: 1.0)

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Print version banner
    version.print_version_banner("train_v1_2", model_version="v1.2")

    print("\nBall Knower v1.2 Training")
    print("=" * 80)

    # Load v1.2 dataset
    print(f"\n[1/4] Loading v1.2 dataset ({start_season}-{end_season})...")
    try:
        df = v1_2.build_training_frame(start_year=start_season, end_year=end_season)
        print(f"  ✓ Loaded {len(df)} games")
    except Exception as e:
        print(f"\n✗ Error loading v1.2 dataset: {e}", file=sys.stderr)
        print(f"  Ensure nfelo data is accessible from:")
        print(f"  https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv")
        return 1

    # Prepare training data
    print("\n[2/4] Preparing training data...")
    try:
        X, y, feature_names = prepare_training_data(df)
        print(f"  ✓ Prepared {len(X)} samples with {len(feature_names)} features")
        print(f"  ✓ Features: {', '.join(feature_names)}")
        print(f"  ✓ Target range: {y.min():.1f} to {y.max():.1f} (Vegas spread)")
    except Exception as e:
        print(f"\n✗ Error preparing training data: {e}", file=sys.stderr)
        return 1

    # Train model
    print(f"\n[3/4] Training Ridge regression model (alpha={alpha})...")
    try:
        model, metrics = train_model(X, y, alpha=alpha)
        print(f"  ✓ Training complete")
        print(f"    - MAE:  {metrics['mae']:.3f} points")
        print(f"    - RMSE: {metrics['rmse']:.3f} points")
        print(f"    - R²:   {metrics['r2']:.3f}")
    except Exception as e:
        print(f"\n✗ Error training model: {e}", file=sys.stderr)
        return 1

    # Save model
    print("\n[4/4] Saving model artifacts...")
    try:
        output_path = paths.get_model_artifact_path("v1.2", "ball_knower_v1_2_model.json")
        save_model_json(model, feature_names, metrics, output_path)

        # Print coefficient summary
        print(f"\n  Coefficients:")
        print(f"    Intercept: {model.intercept_:+.4f}")
        for feature_name, coef in zip(feature_names, model.coef_):
            print(f"    {feature_name:20s}: {coef:+.4f}")

    except Exception as e:
        print(f"\n✗ Error saving model: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 80)
    print("✓ v1.2 model training complete!")
    print(f"  Model file: {output_path}")
    print(f"  Training samples: {metrics['n_samples']}")
    print(f"  Training MAE: {metrics['mae']:.3f} points")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train v1.2 model')
    parser.add_argument('--start-season', type=int, default=2009,
                        help='Start season for training (default: 2009)')
    parser.add_argument('--end-season', type=int, default=2024,
                        help='End season for training (default: 2024)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regression alpha (default: 1.0)')

    args = parser.parse_args()

    sys.exit(main(
        start_season=args.start_season,
        end_season=args.end_season,
        alpha=args.alpha
    ))
