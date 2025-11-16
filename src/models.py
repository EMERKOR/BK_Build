"""
Spread Prediction Models

Model progression:
- v1.0: Deterministic baseline (EPA + ratings + HFA)
- v1.1: Enhanced with structural features (rest, form, matchups)
- v1.2: ML correction layer on top of v1.1

All models predict spread from HOME TEAM perspective.
Negative = home favored, Positive = home underdog
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import HOME_FIELD_ADVANTAGE


# ============================================================================
# MODEL v1.0: DETERMINISTIC BASELINE
# ============================================================================

class DeterministicSpreadModel:
    """
    Simple deterministic spread model based on:
    - EPA differential
    - Power ratings differential
    - Home field advantage

    No ML, just weighted combination of known good predictors.
    """

    def __init__(self, hfa=HOME_FIELD_ADVANTAGE):
        """
        Args:
            hfa (float): Home field advantage in points (default: 2.5)
        """
        self.hfa = hfa
        self.weights = {
            'epa_margin': 100,      # EPA differential * 100 ≈ point spread
            'nfelo_diff': 0.04,     # nfelo difference * 0.04 ≈ point spread
            'substack_ovr_diff': 1.0  # Substack overall rating diff
        }

    def predict(self, home_features, away_features):
        """
        Predict spread from home team perspective.

        Args:
            home_features (dict): Home team features
            away_features (dict): Away team features

        Returns:
            float: Predicted spread (negative = home favored)
        """
        spread = -self.hfa  # Start with HFA (negative because home is favored)

        # Add EPA contribution
        if 'epa_margin' in home_features and 'epa_margin' in away_features:
            epa_diff = home_features['epa_margin'] - away_features['epa_margin']
            spread -= epa_diff * self.weights['epa_margin']

        # Add nfelo contribution
        if 'nfelo' in home_features and 'nfelo' in away_features:
            nfelo_diff = home_features['nfelo'] - away_features['nfelo']
            spread -= nfelo_diff * self.weights['nfelo_diff']

        # Add Substack contribution
        if 'Ovr.' in home_features and 'Ovr.' in away_features:
            ovr_diff = home_features['Ovr.'] - away_features['Ovr.']
            spread -= ovr_diff * self.weights['substack_ovr_diff']

        return spread

    def predict_games(self, games_df):
        """
        Predict spreads for multiple games.

        Args:
            games_df (pd.DataFrame): Games with home/away features

        Returns:
            np.array: Predicted spreads
        """
        predictions = []

        for idx, game in games_df.iterrows():
            home_features = game.filter(regex='.*_home$|^home_').to_dict()
            away_features = game.filter(regex='.*_away$|^away_').to_dict()

            pred = self.predict(home_features, away_features)
            predictions.append(pred)

        return np.array(predictions)


# ============================================================================
# MODEL v1.1: ENHANCED WITH STRUCTURAL FEATURES
# ============================================================================

class EnhancedSpreadModel(DeterministicSpreadModel):
    """
    Extends v1.0 with structural features:
    - Rest advantage
    - Recent form
    - QB adjustments
    """

    def __init__(self, hfa=HOME_FIELD_ADVANTAGE):
        super().__init__(hfa)

        # Additional weights for structural features
        self.weights.update({
            'rest_advantage': 0.3,      # Points per day of rest advantage
            'win_rate_L5': 5.0,         # Recent win rate impact
            'qb_adj_diff': 0.1,         # QB adjustment differential
        })

    def predict(self, home_features, away_features):
        """Predict with structural features."""
        # Start with base v1.0 prediction
        spread = super().predict(home_features, away_features)

        # Add rest advantage
        if 'rest_days' in home_features and 'rest_days' in away_features:
            rest_diff = home_features['rest_days'] - away_features['rest_days']
            spread -= rest_diff * self.weights['rest_advantage']

        # Add recent form
        if 'win_rate_L5' in home_features and 'win_rate_L5' in away_features:
            form_diff = home_features['win_rate_L5'] - away_features['win_rate_L5']
            spread -= form_diff * self.weights['win_rate_L5']

        # Add QB adjustment
        if 'QB Adj' in home_features and 'QB Adj' in away_features:
            qb_diff = home_features['QB Adj'] - away_features['QB Adj']
            spread -= qb_diff * self.weights['qb_adj_diff']

        return spread


# ============================================================================
# MODEL v1.2: ML CORRECTION LAYER
# ============================================================================

class MLCorrectionModel:
    """
    Machine learning model that learns to correct v1.1 predictions.

    This is a SMALL correction layer on top of the deterministic model,
    not a standalone ML model. Helps capture nonlinear effects.
    """

    def __init__(self, base_model=None, ml_model=None):
        """
        Args:
            base_model: Base model (v1.1) to correct
            ml_model: Sklearn model for correction (default: Ridge)
        """
        self.base_model = base_model or EnhancedSpreadModel()
        self.ml_model = ml_model or Ridge(alpha=10.0)
        self.is_fitted = False

    def fit(self, X, y, base_predictions=None):
        """
        Fit ML correction layer.

        Args:
            X (pd.DataFrame): Features
            y (np.array): True spreads
            base_predictions (np.array): Base model predictions (optional)
        """
        if base_predictions is None:
            base_predictions = self.base_model.predict_games(X)

        # Calculate residuals (what the base model got wrong)
        residuals = y - base_predictions

        # Fit ML model to predict residuals
        feature_cols = self._get_ml_features(X)
        self.ml_model.fit(X[feature_cols], residuals)
        self.is_fitted = True

    def predict(self, X):
        """
        Predict spreads with ML correction.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.array: Corrected spread predictions
        """
        # Get base predictions
        base_preds = self.base_model.predict_games(X)

        if not self.is_fitted:
            return base_preds

        # Get ML correction
        feature_cols = self._get_ml_features(X)
        corrections = self.ml_model.predict(X[feature_cols])

        # Return corrected predictions
        return base_preds + corrections

    def _get_ml_features(self, X):
        """Select features for ML model."""
        # Use rolling averages, matchup features, rest, etc.
        ml_features = [
            col for col in X.columns
            if any(pattern in col for pattern in [
                '_L3', '_L5', '_L10',  # Rolling features
                'rest_', 'matchup_', 'qb_', 'coach_'  # Structural features
            ])
        ]
        return ml_features


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, vegas_line=None, model_name="Model"):
    """
    Evaluate spread prediction model.

    Args:
        y_true (np.array): Actual outcomes (home score - away score)
        y_pred (np.array): Predicted spreads
        vegas_line (np.array): Vegas spreads (optional)
        model_name (str): Model name for reporting

    Returns:
        dict: Evaluation metrics
    """
    # Calculate actual spreads (from home perspective)
    # Negative actual = away won by margin, Positive = home won by margin
    # But we're predicting the line, so compare to actual margin

    # For spread prediction, we care about:
    # 1. MAE: How far off are we?
    # 2. RMSE: Penalize large errors
    # 3. ATS win rate: How often do we beat the spread?

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics = {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'n_games': len(y_true)
    }

    # If vegas line provided, calculate edge and ATS performance
    if vegas_line is not None:
        edge = y_pred - vegas_line
        metrics['mean_edge'] = np.mean(np.abs(edge))
        metrics['max_edge'] = np.max(np.abs(edge))

        # ATS: Did we correctly identify which side beats the spread?
        # Actual margin vs predicted margin
        actual_ats = y_true > 0  # Home covered if they won or lost by less than expected
        pred_ats = y_pred > 0

        ats_accuracy = np.mean(actual_ats == pred_ats)
        metrics['ats_accuracy'] = ats_accuracy

    return metrics


def backtest_model(model, train_df, test_df, y_col='actual_margin'):
    """
    Backtest model with time-series cross-validation.

    Args:
        model: Model to backtest
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
        y_col (str): Column with actual outcomes

    Returns:
        dict: Backtest results
    """
    # Fit on training data
    y_train = train_df[y_col].values
    model.fit(train_df, y_train)

    # Predict on test data
    y_test = test_df[y_col].values
    y_pred = model.predict(test_df)

    # Evaluate
    vegas_line = test_df['spread_line'].values if 'spread_line' in test_df.columns else None
    metrics = evaluate_model(y_test, y_pred, vegas_line, model_name=model.__class__.__name__)

    return metrics, y_pred


# ============================================================================
# ROI ANALYSIS
# ============================================================================

def calculate_roi_by_edge(actual_margins, predicted_spreads, vegas_spreads, edge_bins=[0.5, 1, 2, 3, 5]):
    """
    Calculate ROI by edge bin for bet sizing strategy.

    Args:
        actual_margins (np.array): Actual game margins
        predicted_spreads (np.array): Model predictions
        vegas_spreads (np.array): Vegas lines
        edge_bins (list): Edge threshold bins

    Returns:
        pd.DataFrame: ROI analysis by edge bin
    """
    df = pd.DataFrame({
        'actual_margin': actual_margins,
        'pred_spread': predicted_spreads,
        'vegas_spread': vegas_spreads
    })

    # Calculate edge
    df['edge'] = np.abs(df['pred_spread'] - df['vegas_spread'])

    # Determine which side to bet (model vs vegas)
    df['bet_home'] = df['pred_spread'] < df['vegas_spread']  # Bet home if model more favorable to home

    # Calculate if bet won
    # Home covers if actual margin + vegas spread > 0
    df['home_covered'] = (df['actual_margin'] + df['vegas_spread']) > 0
    df['bet_won'] = df['bet_home'] == df['home_covered']

    # Analyze by edge bin
    results = []
    for i, threshold in enumerate(edge_bins):
        if i == 0:
            mask = df['edge'] >= threshold
        else:
            mask = (df['edge'] >= threshold) & (df['edge'] < edge_bins[i-1])

        if mask.sum() == 0:
            continue

        subset = df[mask]
        win_rate = subset['bet_won'].mean()
        n_bets = len(subset)

        # ROI assuming -110 odds (risk 1.1 to win 1.0)
        roi = (win_rate * 1.0 - (1 - win_rate) * 1.1) / 1.1

        results.append({
            'edge_threshold': threshold,
            'n_bets': n_bets,
            'win_rate': win_rate,
            'roi': roi
        })

    return pd.DataFrame(results)
