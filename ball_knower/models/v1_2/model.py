"""
v1.2 Residual Model

Ridge regression model for predicting Vegas line residuals.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import get_feature_matrix, get_target_vector, FEATURE_COLUMNS


class V1_2ResidualModel:
    """
    v1.2 Spread Correction Model

    Trains a Ridge regressor to predict:
        residual = vegas_closing_spread - bk_base_spread

    Final prediction:
        bk_spread_pred = bk_base_spread + residual_pred

    This allows the model to learn a correction factor on top of the base
    nfelo spread prediction.

    Attributes:
        model: Trained Ridge regression model
        feature_names: List of feature column names
        metrics: Dict of training/test metrics
        alpha: Ridge regularization parameter

    Example:
        >>> model = V1_2ResidualModel(alpha=1.0)
        >>> model.fit(train_df)
        >>> predictions = model.predict(test_df)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize v1.2 residual model.

        Args:
            alpha: Ridge regularization strength (default: 1.0)
        """
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_names = FEATURE_COLUMNS.copy()
        self.metrics = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'V1_2ResidualModel':
        """
        Train the model on a dataset.

        Args:
            df: DataFrame from v1_2 dataset builder

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If required columns are missing
        """

        print(f"\nTraining v1.2 model (alpha={self.alpha})...")

        # Extract features and target
        X, feature_names = get_feature_matrix(df)
        y = get_target_vector(df)

        # Verify feature names match
        if feature_names != self.feature_names:
            raise ValueError("Feature names mismatch")

        # Train model
        self.model.fit(X, y)
        self._is_fitted = True

        # Compute training metrics
        y_pred = self.model.predict(X)
        self.metrics['train'] = self._compute_metrics(y, y_pred)

        print(f"  Train MAE: {self.metrics['train']['mae']:.3f}")
        print(f"  Train R²:  {self.metrics['train']['r2']:.3f}")

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict residuals for a dataset.

        Args:
            df: DataFrame from v1_2 dataset builder

        Returns:
            Series of predicted residuals

        Raises:
            RuntimeError: If model hasn't been fitted
            ValueError: If required columns are missing
        """

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Extract features
        X, _ = get_feature_matrix(df)

        # Predict residuals
        residuals_pred = self.model.predict(X)

        return pd.Series(residuals_pred, index=df.index, name='residual_pred')

    def predict_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict final spread (base + residual correction).

        Args:
            df: DataFrame from v1_2 dataset builder (must contain 'bk_base_spread')

        Returns:
            Series of predicted spreads

        Raises:
            ValueError: If 'bk_base_spread' column is missing
        """

        if 'bk_base_spread' not in df.columns:
            raise ValueError("DataFrame must contain 'bk_base_spread' column")

        # Predict residual
        residual_pred = self.predict(df)

        # Add residual to base prediction
        spread_pred = df['bk_base_spread'] + residual_pred

        return pd.Series(spread_pred, index=df.index, name='bk_spread_pred')

    def evaluate(self, df: pd.DataFrame, split_name: str = 'test') -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            df: DataFrame from v1_2 dataset builder
            split_name: Name of this data split (e.g., 'train', 'test')

        Returns:
            Dict of metrics

        Raises:
            RuntimeError: If model hasn't been fitted
        """

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        # Get true residuals
        y_true = get_target_vector(df)

        # Predict residuals
        y_pred = self.predict(df)

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred)

        # Store metrics
        self.metrics[split_name] = metrics

        return metrics

    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Compute regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dict of metrics
        """

        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mean_residual': float(np.mean(y_true - y_pred)),
            'n_samples': len(y_true)
        }

    def save(self, path: Path) -> Path:
        """
        Save model parameters to JSON.

        Args:
            path: Output file path (will create parent dirs if needed)

        Returns:
            Path to saved file

        Raises:
            RuntimeError: If model hasn't been fitted
        """

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare model data
        model_data = {
            'model_type': 'V1_2ResidualModel',
            'alpha': self.alpha,
            'intercept': float(self.model.intercept_),
            'coefficients': {
                name: float(coef)
                for name, coef in zip(self.feature_names, self.model.coef_)
            },
            'feature_names': self.feature_names,
            'metrics': self.metrics,
        }

        # Save to JSON
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> 'V1_2ResidualModel':
        """
        Load model from JSON file.

        Args:
            path: Path to saved model JSON

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load JSON
        with open(path, 'r') as f:
            model_data = json.load(f)

        # Verify model type
        if model_data.get('model_type') != 'V1_2ResidualModel':
            raise ValueError("Invalid model type in file")

        # Create model instance
        model = cls(alpha=model_data['alpha'])

        # Restore coefficients
        model.model.intercept_ = model_data['intercept']
        model.model.coef_ = np.array([
            model_data['coefficients'][name]
            for name in model.feature_names
        ])

        # Restore metadata
        model.metrics = model_data.get('metrics', {})
        model._is_fitted = True

        return model

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (absolute coefficient values).

        Returns:
            DataFrame with features sorted by importance

        Raises:
            RuntimeError: If model hasn't been fitted
        """

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })

        return importance.sort_values('abs_coefficient', ascending=False)

    def summary(self) -> str:
        """
        Get a text summary of the model.

        Returns:
            Formatted summary string
        """

        if not self._is_fitted:
            return "Model not fitted yet"

        lines = [
            "="*60,
            "Ball Knower v1.2 Residual Model Summary",
            "="*60,
            f"Alpha (regularization): {self.alpha}",
            f"Intercept: {self.model.intercept_:.4f}",
            "",
            "Feature Coefficients:",
        ]

        for name, coef in zip(self.feature_names, self.model.coef_):
            lines.append(f"  {name:25} {coef:+.6f}")

        if self.metrics:
            lines.append("")
            lines.append("Performance Metrics:")
            for split_name, metrics in self.metrics.items():
                lines.append(f"\n  {split_name.upper()}:")
                lines.append(f"    MAE:  {metrics['mae']:.3f}")
                lines.append(f"    RMSE: {metrics['rmse']:.3f}")
                lines.append(f"    R²:   {metrics['r2']:.3f}")
                lines.append(f"    N:    {metrics['n_samples']:,}")

        lines.append("="*60)

        return "\n".join(lines)
