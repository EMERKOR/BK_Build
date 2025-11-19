"""
Test calibrated weights loading behavior with strict error handling.

Tests verify that:
1. When use_calibrated=True and JSON exists and is valid → Model initializes successfully
2. When use_calibrated=True and JSON is missing → Raises RuntimeError
3. When use_calibrated=True and JSON is invalid → Raises RuntimeError
4. When use_calibrated=False → Model uses defaults without touching JSON
"""

import json
import pytest
from pathlib import Path
from src.models import DeterministicSpreadModel, EnhancedSpreadModel


class TestCalibratedWeightsLoading:
    """Test strict calibration weight loading behavior."""

    @pytest.fixture
    def valid_calibration_json(self):
        """Valid calibration file content."""
        return {
            "hfa": 2.8,
            "weights": {
                "epa_margin": 38.0,
                "nfelo_diff": 0.025,
                "substack_ovr_diff": 0.6,
                "rest_advantage": 0.35,
                "win_rate_L5": 5.5,
                "qb_adj_diff": 0.12
            },
            "metadata": {
                "calibrated_on": "2015-2024",
                "mae_vs_vegas": 10.2,
                "n_games": 2800
            }
        }

    @pytest.fixture
    def minimal_calibration_json(self):
        """Minimal valid calibration (just weights)."""
        return {
            "weights": {
                "epa_margin": 35.0,
                "nfelo_diff": 0.02
            }
        }

    @pytest.fixture
    def invalid_json_missing_weights(self):
        """Invalid calibration: missing 'weights' key."""
        return {
            "hfa": 2.5,
            "metadata": {
                "calibrated_on": "2015-2024"
            }
        }

    # =========================================================================
    # TEST: use_calibrated=True with VALID calibration file
    # =========================================================================

    def test_deterministic_model_loads_valid_calibration(self, tmp_path, valid_calibration_json):
        """Test v1.0 model loads valid calibration file successfully."""
        # Create calibration file
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump(valid_calibration_json, f)

        # Initialize model with calibration
        model = DeterministicSpreadModel(weights_file=calib_file, use_calibrated=True)

        # Verify it loaded the calibrated values
        assert model.hfa == 2.8
        assert model.weights['epa_margin'] == 38.0
        assert model.weights['nfelo_diff'] == 0.025
        assert model.weights['substack_ovr_diff'] == 0.6

    def test_enhanced_model_loads_valid_calibration(self, tmp_path, valid_calibration_json):
        """Test v1.1 model loads valid calibration file successfully."""
        # Create calibration file
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump(valid_calibration_json, f)

        # Initialize model with calibration
        model = EnhancedSpreadModel(weights_file=calib_file, use_calibrated=True)

        # Verify it loaded the calibrated values (both base and enhanced)
        assert model.hfa == 2.8
        assert model.weights['epa_margin'] == 38.0
        assert model.weights['rest_advantage'] == 0.35
        assert model.weights['win_rate_L5'] == 5.5
        assert model.weights['qb_adj_diff'] == 0.12

    # =========================================================================
    # TEST: use_calibrated=True with MISSING calibration file
    # =========================================================================

    def test_deterministic_model_raises_error_when_calibration_missing(self, tmp_path):
        """Test v1.0 model raises RuntimeError when calibration file is missing."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(RuntimeError) as exc_info:
            DeterministicSpreadModel(weights_file=nonexistent_file, use_calibrated=True)

        # Verify error message contains expected text
        error_msg = str(exc_info.value)
        assert "Calibration file not found" in error_msg
        assert str(nonexistent_file) in error_msg
        assert "use_calibrated=True" in error_msg

    def test_enhanced_model_raises_error_when_calibration_missing(self, tmp_path):
        """Test v1.1 model raises RuntimeError when calibration file is missing."""
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(RuntimeError) as exc_info:
            EnhancedSpreadModel(weights_file=nonexistent_file, use_calibrated=True)

        error_msg = str(exc_info.value)
        assert "Calibration file not found" in error_msg

    # =========================================================================
    # TEST: use_calibrated=True with INVALID calibration file
    # =========================================================================

    def test_model_raises_error_on_malformed_json(self, tmp_path):
        """Test model raises RuntimeError when JSON is malformed."""
        calib_file = tmp_path / "malformed.json"
        with open(calib_file, 'w') as f:
            f.write("{ invalid json here!!! }")

        with pytest.raises(RuntimeError) as exc_info:
            DeterministicSpreadModel(weights_file=calib_file, use_calibrated=True)

        error_msg = str(exc_info.value)
        assert "Failed to parse calibration file" in error_msg
        assert str(calib_file) in error_msg

    def test_model_raises_error_on_missing_weights_key(self, tmp_path, invalid_json_missing_weights):
        """Test model raises RuntimeError when 'weights' key is missing."""
        calib_file = tmp_path / "invalid.json"
        with open(calib_file, 'w') as f:
            json.dump(invalid_json_missing_weights, f)

        with pytest.raises(RuntimeError) as exc_info:
            DeterministicSpreadModel(weights_file=calib_file, use_calibrated=True)

        error_msg = str(exc_info.value)
        assert "Invalid calibration file" in error_msg
        assert "missing 'weights' key" in error_msg

    def test_model_raises_error_on_empty_file(self, tmp_path):
        """Test model raises RuntimeError when file is empty."""
        calib_file = tmp_path / "empty.json"
        calib_file.touch()  # Create empty file

        with pytest.raises(RuntimeError) as exc_info:
            DeterministicSpreadModel(weights_file=calib_file, use_calibrated=True)

        error_msg = str(exc_info.value)
        assert "Failed to parse calibration file" in error_msg

    # =========================================================================
    # TEST: use_calibrated=False (non-calibrated mode)
    # =========================================================================

    def test_deterministic_model_uses_defaults_when_calibration_disabled(self, tmp_path):
        """Test v1.0 model uses defaults when use_calibrated=False."""
        # Even if a calibration file exists, it should NOT be used
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump({"hfa": 999.0, "weights": {"epa_margin": 999.0}}, f)

        # Initialize with use_calibrated=False
        model = DeterministicSpreadModel(weights_file=calib_file, use_calibrated=False)

        # Should use defaults, NOT the file values
        assert model.hfa != 999.0
        assert model.weights['epa_margin'] == 35  # Default value

    def test_enhanced_model_uses_defaults_when_calibration_disabled(self, tmp_path):
        """Test v1.1 model uses defaults when use_calibrated=False."""
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump({"hfa": 999.0, "weights": {"rest_advantage": 999.0}}, f)

        model = EnhancedSpreadModel(weights_file=calib_file, use_calibrated=False)

        # Should use defaults
        assert model.weights['rest_advantage'] == 0.3  # Default value
        assert model.weights['epa_margin'] == 35

    def test_model_works_when_file_missing_and_calibration_disabled(self, tmp_path):
        """Test model works fine when use_calibrated=False even if file doesn't exist."""
        nonexistent_file = tmp_path / "does_not_exist.json"

        # Should NOT raise error when use_calibrated=False
        model = DeterministicSpreadModel(weights_file=nonexistent_file, use_calibrated=False)

        # Should have default weights
        assert model.weights['epa_margin'] == 35
        assert model.hfa == 2.5  # Default HFA from config

    # =========================================================================
    # TEST: Model prediction functionality still works
    # =========================================================================

    def test_model_can_predict_with_calibrated_weights(self, tmp_path, valid_calibration_json):
        """Test that model can make predictions after loading calibration."""
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump(valid_calibration_json, f)

        model = EnhancedSpreadModel(weights_file=calib_file, use_calibrated=True)

        # Make a prediction
        home_features = {
            'nfelo': 1600,
            'epa_margin': 0.1,
            'Ovr.': 25.0,
            'rest_days': 7,
            'win_rate_L5': 0.6,
            'QB Adj': 2.0
        }
        away_features = {
            'nfelo': 1500,
            'epa_margin': -0.05,
            'Ovr.': 20.0,
            'rest_days': 7,
            'win_rate_L5': 0.4,
            'QB Adj': 0.0
        }

        prediction = model.predict(home_features, away_features)

        # Should return a valid prediction (negative = home favored)
        assert isinstance(prediction, (int, float))
        assert -50 < prediction < 50  # Reasonable spread range

    def test_model_can_predict_without_calibration(self):
        """Test that model can make predictions in non-calibrated mode."""
        model = DeterministicSpreadModel(use_calibrated=False)

        home_features = {
            'nfelo': 1600,
            'epa_margin': 0.1,
            'Ovr.': 25.0
        }
        away_features = {
            'nfelo': 1500,
            'epa_margin': -0.05,
            'Ovr.': 20.0
        }

        prediction = model.predict(home_features, away_features)

        # Should return a valid prediction
        assert isinstance(prediction, (int, float))
        assert -50 < prediction < 50


class TestBackwardCompatibility:
    """Test that existing code patterns still work."""

    def test_default_initialization_without_file(self):
        """Test that models can be initialized with default behavior (backward compat)."""
        # This should work if the default calibration file exists in output/
        # But if it doesn't exist and use_calibrated=True (default), it should raise
        # To maintain backward compat, we test use_calibrated=False
        model = DeterministicSpreadModel(use_calibrated=False)
        assert model is not None
        assert model.weights['epa_margin'] == 35

    def test_custom_hfa_override(self, tmp_path):
        """Test that custom HFA parameter still works."""
        calib_file = tmp_path / "calibration.json"
        with open(calib_file, 'w') as f:
            json.dump({"hfa": 2.8, "weights": {"epa_margin": 35}}, f)

        # HFA should be overridden by the parameter
        model = DeterministicSpreadModel(hfa=3.5, weights_file=calib_file, use_calibrated=True)
        assert model.hfa == 3.5  # Override should work
