"""
Test src.models module

Verifies that models:
- Load calibrated weights from JSON when available
- Fall back to defaults when JSON is missing
- Produce numeric predictions from dummy features
- Handle edge cases gracefully
"""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch

from src import models, config


# ============================================================================
# TEST CALIBRATED WEIGHTS LOADING
# ============================================================================

def test_load_calibrated_weights_with_file():
    """
    Test that load_calibrated_weights loads from JSON when file exists.
    """
    # Check if calibrated weights file exists
    weights_file = config.OUTPUT_DIR / "calibrated_weights_v1.json"

    if not weights_file.exists():
        pytest.skip(
            f"Calibrated weights file not found at {weights_file}. "
            "Run calibrate_v1_json.py to generate it, or test will use defaults."
        )

    # Load weights
    calibration = models.load_calibrated_weights(weights_file)

    # Assert it's a dict with expected keys
    assert isinstance(calibration, dict), "Calibration should be a dict"
    assert "hfa" in calibration, "Calibration should have 'hfa' key"
    assert "weights" in calibration, "Calibration should have 'weights' key"

    # Assert hfa is numeric
    assert isinstance(calibration["hfa"], (int, float)), "HFA should be numeric"

    # Assert weights is a dict
    assert isinstance(calibration["weights"], dict), "Weights should be a dict"


def test_load_calibrated_weights_fallback_when_missing():
    """
    Test that load_calibrated_weights falls back to defaults when file is missing.
    """
    # Use a non-existent path
    fake_path = Path("/tmp/nonexistent_calibration.json")

    # Load weights (should fall back to defaults)
    calibration = models.load_calibrated_weights(fake_path)

    # Assert it returns defaults
    assert isinstance(calibration, dict), "Should return dict even when file missing"
    assert "hfa" in calibration, "Should have 'hfa' in defaults"
    assert "weights" in calibration, "Should have 'weights' in defaults"

    # Assert defaults are reasonable
    assert calibration["hfa"] > 0, "Default HFA should be positive"
    assert len(calibration["weights"]) > 0, "Default weights should not be empty"


# ============================================================================
# TEST DeterministicSpreadModel (v1.0)
# ============================================================================

def test_deterministic_spread_model_instantiation():
    """
    Test that DeterministicSpreadModel can be instantiated.
    """
    model = models.DeterministicSpreadModel(use_calibrated=False)

    # Assert model has required attributes
    assert hasattr(model, "hfa"), "Model should have 'hfa' attribute"
    assert hasattr(model, "weights"), "Model should have 'weights' attribute"

    # Assert hfa is numeric
    assert isinstance(model.hfa, (int, float)), "HFA should be numeric"

    # Assert weights is a dict
    assert isinstance(model.weights, dict), "Weights should be a dict"


def test_deterministic_spread_model_prediction():
    """
    Test that DeterministicSpreadModel produces numeric predictions.
    """
    model = models.DeterministicSpreadModel(use_calibrated=False)

    # Create dummy feature dicts
    home_features = {
        'nfelo': 1600,
        'epa_margin': 0.1,
        'Ovr.': 25.0,
    }

    away_features = {
        'nfelo': 1500,
        'epa_margin': -0.05,
        'Ovr.': 20.0,
    }

    # Predict
    prediction = model.predict(home_features, away_features)

    # Assert prediction is numeric
    assert isinstance(prediction, (int, float, np.number)), \
        "Prediction should be numeric"

    # Assert prediction is not NaN
    assert not np.isnan(prediction), "Prediction should not be NaN"

    # Assert prediction is reasonable (within -30 to +30 points)
    assert -30 < prediction < 30, \
        f"Prediction should be within reasonable range, got {prediction}"


def test_deterministic_spread_model_with_calibrated_weights():
    """
    Test that DeterministicSpreadModel can load calibrated weights if available.
    """
    # Try to load calibrated weights
    weights_file = config.OUTPUT_DIR / "calibrated_weights_v1.json"

    if not weights_file.exists():
        pytest.skip(
            "Calibrated weights file not found. "
            "Skipping test with calibrated weights."
        )

    # Instantiate with calibrated weights
    model = models.DeterministicSpreadModel(use_calibrated=True)

    # Create dummy features
    home_features = {
        'nfelo': 1600,
        'epa_margin': 0.1,
        'Ovr.': 25.0,
    }

    away_features = {
        'nfelo': 1500,
        'epa_margin': -0.05,
        'Ovr.': 20.0,
    }

    # Predict
    prediction = model.predict(home_features, away_features)

    # Assert prediction is valid
    assert isinstance(prediction, (int, float, np.number)), \
        "Prediction with calibrated weights should be numeric"
    assert not np.isnan(prediction), \
        "Prediction with calibrated weights should not be NaN"


# ============================================================================
# TEST EnhancedSpreadModel (v1.1)
# ============================================================================

def test_enhanced_spread_model_instantiation():
    """
    Test that EnhancedSpreadModel can be instantiated.
    """
    model = models.EnhancedSpreadModel(use_calibrated=False)

    # Assert model has required attributes
    assert hasattr(model, "hfa"), "Model should have 'hfa' attribute"
    assert hasattr(model, "weights"), "Model should have 'weights' attribute"

    # Assert weights includes enhanced features
    assert 'rest_advantage' in model.weights or \
           'win_rate_L5' in model.weights or \
           'qb_adj_diff' in model.weights, \
        "EnhancedSpreadModel should have enhanced feature weights"


def test_enhanced_spread_model_prediction():
    """
    Test that EnhancedSpreadModel produces numeric predictions.
    """
    model = models.EnhancedSpreadModel(use_calibrated=False)

    # Create dummy feature dicts with enhanced features
    home_features = {
        'nfelo': 1600,
        'epa_margin': 0.1,
        'Ovr.': 25.0,
        'rest_days': 7,
        'win_rate_L5': 0.6,
        'QB Adj': 2.0,
    }

    away_features = {
        'nfelo': 1500,
        'epa_margin': -0.05,
        'Ovr.': 20.0,
        'rest_days': 7,
        'win_rate_L5': 0.4,
        'QB Adj': 0.0,
    }

    # Predict
    prediction = model.predict(home_features, away_features)

    # Assert prediction is numeric
    assert isinstance(prediction, (int, float, np.number)), \
        "Prediction should be numeric"

    # Assert prediction is not NaN
    assert not np.isnan(prediction), "Prediction should not be NaN"

    # Assert prediction is reasonable
    assert -30 < prediction < 30, \
        f"Prediction should be within reasonable range, got {prediction}"


# ============================================================================
# TEST MODEL FALLBACK BEHAVIOR
# ============================================================================

def test_model_fallback_when_calibration_file_missing():
    """
    Test that models don't crash when calibration file is missing.

    They should fall back to hard-coded defaults.
    """
    # Use a non-existent calibration file path
    fake_path = Path("/tmp/nonexistent_calibration.json")

    # Instantiate models with missing file
    model_v1_0 = models.DeterministicSpreadModel(
        use_calibrated=True,
        weights_file=fake_path
    )

    model_v1_1 = models.EnhancedSpreadModel(
        use_calibrated=True,
        weights_file=fake_path
    )

    # Create dummy features
    home_features = {
        'nfelo': 1600,
        'epa_margin': 0.1,
        'Ovr.': 25.0,
        'rest_days': 7,
        'win_rate_L5': 0.6,
        'QB Adj': 2.0,
    }

    away_features = {
        'nfelo': 1500,
        'epa_margin': -0.05,
        'Ovr.': 20.0,
        'rest_days': 7,
        'win_rate_L5': 0.4,
        'QB Adj': 0.0,
    }

    # Both models should still produce valid predictions
    pred_v1_0 = model_v1_0.predict(home_features, away_features)
    pred_v1_1 = model_v1_1.predict(home_features, away_features)

    # Assert predictions are valid
    assert not np.isnan(pred_v1_0), "v1.0 should not return NaN with missing file"
    assert not np.isnan(pred_v1_1), "v1.1 should not return NaN with missing file"


def test_model_prediction_with_missing_features():
    """
    Test that models handle missing features gracefully (skip those features).
    """
    model = models.DeterministicSpreadModel(use_calibrated=False)

    # Features with some missing keys
    home_features = {
        'nfelo': 1600,
        # Missing epa_margin and Ovr.
    }

    away_features = {
        'nfelo': 1500,
        # Missing epa_margin and Ovr.
    }

    # Should still produce a prediction (using only available features)
    prediction = model.predict(home_features, away_features)

    # Assert prediction is valid
    assert isinstance(prediction, (int, float, np.number)), \
        "Should handle missing features gracefully"
    assert not np.isnan(prediction), \
        "Should not return NaN with partial features"
