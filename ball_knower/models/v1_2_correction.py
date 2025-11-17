"""
Ball Knower v1.2: Spread Correction Model

Small, stable ML layer that learns residual adjustments on top of the deterministic
Ball Knower line using only pre-game, leakage-free features.

Model Architecture:
    1. Base deterministic model (v1.0) generates initial spread prediction
    2. ML correction layer learns residuals (actual - predicted)
    3. Final prediction = base prediction + ML correction

Features Used (all canonical, provider-agnostic):
    - Base model prediction
    - Overall rating differential
    - EPA margin differential
    - Offensive/defensive rating differentials
    - QB adjustment differential
    - Home field indicator (binary)
    - Structural factors (divisional game, rest advantage)

Training:
    - Train on historical week ranges
    - Learn to predict residuals (Vegas line - base prediction)
    - Ridge regression for stability and regularization

Usage:
    from ball_knower.models.v1_2_correction import SpreadCorrectionModel
    from src.models import DeterministicSpreadModel

    base_model = DeterministicSpreadModel()
    correction_model = SpreadCorrectionModel(base_model=base_model)

    # Train on weeks 1-10
    correction_model.fit(train_matchups, vegas_lines)

    # Predict on week 11
    corrected_spreads = correction_model.predict(test_matchups)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, List, Tuple
import warnings


class SpreadCorrectionModel:
    """
    v1.2 Spread Correction Model

    Learns residual corrections on top of a deterministic base model.
    Uses only pre-game, leakage-free canonical features.
    """

    def __init__(
        self,
        base_model=None,
        alpha: float = 10.0,
        fit_intercept: bool = True,
        normalize_features: bool = True
    ):
        """
        Initialize the spread correction model.

        Args:
            base_model: Base deterministic model (e.g., DeterministicSpreadModel)
                       If None, will use canonical features directly without base prediction
            alpha: Ridge regression regularization strength (default: 10.0)
            fit_intercept: Whether to fit intercept in Ridge regression (default: True)
            normalize_features: Whether to standardize features before fitting (default: True)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features

        # ML model for learning corrections
        self.correction_model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

        # Feature scaler (if normalization enabled)
        self.scaler = StandardScaler() if normalize_features else None

        # Track which features are available
        self.feature_names = []
        self.is_fitted = False

    def _extract_correction_features(
        self,
        matchups: pd.DataFrame,
        include_base_prediction: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for the correction model from matchup data.

        Args:
            matchups: DataFrame with canonical matchup features
                     Expected columns with _diff suffix:
                     - overall_rating_diff
                     - epa_margin_diff
                     - offensive_rating_diff
                     - defensive_rating_diff
                     - qb_adjustment_diff (optional)

                     Expected binary columns:
                     - is_home (or will create from structure)
                     - div_game (optional)

                     Expected continuous columns:
                     - rest_diff (optional - days of rest advantage)

            include_base_prediction: Whether to include base model prediction as feature

        Returns:
            DataFrame with extracted features for correction model
        """
        features = pd.DataFrame(index=matchups.index)

        # Feature 1: Base model prediction (if available)
        if include_base_prediction and self.base_model is not None:
            if 'base_prediction' in matchups.columns:
                features['base_prediction'] = matchups['base_prediction']
            else:
                warnings.warn(
                    "Base model provided but 'base_prediction' not in matchups. "
                    "Call predict_base() first or include base_prediction column.",
                    UserWarning
                )

        # Feature 2-5: Rating differentials (core canonical features)
        for feature in ['overall_rating_diff', 'epa_margin_diff',
                       'offensive_rating_diff', 'defensive_rating_diff']:
            if feature in matchups.columns:
                features[feature] = matchups[feature]

        # Feature 6: QB adjustment differential (if available)
        if 'qb_adjustment_diff' in matchups.columns:
            features['qb_adjustment_diff'] = matchups['qb_adjustment_diff']

        # Feature 7: Home field indicator
        if 'is_home' in matchups.columns:
            features['is_home'] = matchups['is_home']
        else:
            # Assume all are home games (neutral site = 0.5)
            features['is_home'] = 1.0

        # Feature 8: Divisional game indicator (if available)
        if 'div_game' in matchups.columns:
            features['div_game'] = matchups['div_game']

        # Feature 9: Rest advantage (if available)
        if 'rest_diff' in matchups.columns:
            features['rest_diff'] = matchups['rest_diff']

        # Drop rows with all NaN (no features available)
        features = features.dropna(how='all')

        return features

    def predict_base(self, matchups: pd.DataFrame) -> np.ndarray:
        """
        Generate base model predictions for matchups.

        Args:
            matchups: DataFrame with home/away canonical features
                     Expected columns:
                     - overall_rating_home, overall_rating_away
                     - epa_margin_home, epa_margin_away
                     - offensive_rating_home, offensive_rating_away
                     - defensive_rating_home, defensive_rating_away

        Returns:
            Array of base model predictions
        """
        if self.base_model is None:
            raise ValueError("Base model not provided. Cannot generate base predictions.")

        predictions = []

        for idx, row in matchups.iterrows():
            home_features = {
                col.replace('_home', ''): row[col]
                for col in matchups.columns
                if col.endswith('_home')
            }

            away_features = {
                col.replace('_away', ''): row[col]
                for col in matchups.columns
                if col.endswith('_away')
            }

            pred = self.base_model.predict(home_features, away_features)
            predictions.append(pred)

        return np.array(predictions)

    def fit(
        self,
        matchups: pd.DataFrame,
        vegas_lines: np.ndarray,
        verbose: bool = True
    ) -> 'SpreadCorrectionModel':
        """
        Train the correction model to predict residuals.

        Args:
            matchups: DataFrame with canonical matchup features
            vegas_lines: Array of actual Vegas lines (target to learn)
            verbose: Whether to print training info (default: True)

        Returns:
            Self (for method chaining)
        """
        # Step 1: Generate base predictions if base model provided
        if self.base_model is not None:
            base_predictions = self.predict_base(matchups)
            matchups = matchups.copy()
            matchups['base_prediction'] = base_predictions
        else:
            base_predictions = np.zeros(len(matchups))

        # Step 2: Calculate residuals (what the base model got wrong)
        residuals = vegas_lines - base_predictions

        # Step 3: Extract features for correction model
        X = self._extract_correction_features(matchups, include_base_prediction=True)

        # Ensure same indices
        common_idx = X.index.intersection(matchups.index)
        X = X.loc[common_idx]
        y = residuals[common_idx] if isinstance(residuals, pd.Series) else residuals[common_idx.tolist()]

        # Step 4: Handle missing values
        X = X.fillna(0)  # Simple imputation for missing features

        # Store feature names
        self.feature_names = list(X.columns)

        # Step 5: Normalize features if enabled
        if self.normalize_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Step 6: Fit correction model
        self.correction_model.fit(X_scaled, y)
        self.is_fitted = True

        if verbose:
            # Calculate training metrics
            train_corrections = self.correction_model.predict(X_scaled)
            train_predictions = base_predictions[common_idx.tolist()] + train_corrections
            train_mae = np.mean(np.abs(vegas_lines[common_idx.tolist()] - train_predictions))
            train_rmse = np.sqrt(np.mean((vegas_lines[common_idx.tolist()] - train_predictions) ** 2))

            print(f"\n{'='*70}")
            print(f"CORRECTION MODEL TRAINING")
            print(f"{'='*70}")
            print(f"Training samples: {len(X)}")
            print(f"Features used: {len(self.feature_names)}")
            print(f"  - {', '.join(self.feature_names)}")
            print(f"\nBase model MAE: {np.mean(np.abs(residuals[common_idx.tolist()])):.3f} points")
            print(f"Corrected MAE:  {train_mae:.3f} points")
            print(f"Corrected RMSE: {train_rmse:.3f} points")
            print(f"{'='*70}\n")

        return self

    def predict(self, matchups: pd.DataFrame) -> np.ndarray:
        """
        Generate corrected spread predictions.

        Args:
            matchups: DataFrame with canonical matchup features

        Returns:
            Array of corrected spread predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Step 1: Generate base predictions
        if self.base_model is not None:
            base_predictions = self.predict_base(matchups)
            matchups = matchups.copy()
            matchups['base_prediction'] = base_predictions
        else:
            base_predictions = np.zeros(len(matchups))

        # Step 2: Extract features for correction
        X = self._extract_correction_features(matchups, include_base_prediction=True)

        # Ensure same features as training
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0  # Add missing features as zero

        X = X[self.feature_names]  # Ensure same order
        X = X.fillna(0)  # Handle missing values

        # Step 3: Normalize if enabled
        if self.normalize_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        # Step 4: Predict corrections
        corrections = self.correction_model.predict(X_scaled)

        # Step 5: Apply corrections to base predictions
        corrected_predictions = base_predictions + corrections

        return corrected_predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (coefficients) from the correction model.

        Returns:
            Dict mapping feature names to coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = self.correction_model.coef_

        return dict(zip(self.feature_names, coefficients))

    def evaluate(
        self,
        matchups: pd.DataFrame,
        vegas_lines: np.ndarray,
        actual_margins: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            matchups: DataFrame with canonical matchup features
            vegas_lines: Array of actual Vegas lines
            actual_margins: Optional array of actual game margins (for ATS analysis)

        Returns:
            Dict of evaluation metrics
        """
        predictions = self.predict(matchups)

        # Basic metrics
        mae = np.mean(np.abs(predictions - vegas_lines))
        rmse = np.sqrt(np.mean((predictions - vegas_lines) ** 2))
        correlation = np.corrcoef(predictions, vegas_lines)[0, 1]

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'n_games': len(predictions)
        }

        # ATS metrics if actual margins provided
        if actual_margins is not None:
            # Model picks
            model_picks_home = predictions < 0  # Negative = home favored

            # Actual results
            home_covered = (actual_margins + vegas_lines) > 0

            # ATS accuracy
            ats_correct = (model_picks_home == home_covered).sum()
            ats_accuracy = ats_correct / len(predictions)

            metrics['ats_accuracy'] = ats_accuracy
            metrics['ats_correct'] = int(ats_correct)

        return metrics
