"""
Test src/train_v1_2.py training pipeline

Tests for v1.2 model training:
- Training script runs successfully on restricted data
- Model artifact is created in expected location
- Model artifact has expected JSON structure
- Backtest can load and use the trained model
"""

import pytest
import subprocess
import json
from pathlib import Path
import pandas as pd


# ============================================================================
# TEST V1.2 TRAINING SCRIPT
# ============================================================================

def test_train_v1_2_script_creates_artifact():
    """
    Test that train_v1_2.py creates model artifact file.

    Uses a restricted season range (2019-2019) to keep test fast.
    """
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "src" / "train_v1_2.py"
    output_dir = project_root / "output" / "models" / "v1_2"
    model_file = output_dir / "ball_knower_v1_2_model.json"

    # Remove existing model file if present
    if model_file.exists():
        model_file.unlink()

    # Run training script with restricted data (single season for speed)
    result = subprocess.run(
        [
            "python",
            str(train_script),
            "--start-season", "2019",
            "--end-season", "2019",
            "--alpha", "1.0"
        ],
        capture_output=True,
        text=True,
        timeout=120  # 2 minutes max (downloads nfelo data)
    )

    # Assert training succeeded
    assert result.returncode == 0, \
        f"Training script failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"

    # Assert model file was created
    assert model_file.exists(), \
        f"Model file not created at {model_file}"

    # Load and validate model JSON
    with open(model_file, 'r') as f:
        model_json = json.load(f)

    # Assert required keys exist
    assert 'intercept' in model_json, "Model JSON missing 'intercept' key"
    assert 'coefficients' in model_json, "Model JSON missing 'coefficients' key"

    # Assert coefficients has expected features
    expected_features = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
    ]

    for feature in expected_features:
        assert feature in model_json['coefficients'], \
            f"Model JSON coefficients missing feature: {feature}"

    # Assert intercept and coefficients are numeric
    assert isinstance(model_json['intercept'], (int, float)), \
        "Intercept should be numeric"

    for feature, coef in model_json['coefficients'].items():
        assert isinstance(coef, (int, float)), \
            f"Coefficient for {feature} should be numeric"

    # Assert metadata exists and has expected keys
    if 'metadata' in model_json:
        metadata = model_json['metadata']
        assert 'n_samples' in metadata, "Metadata missing 'n_samples'"
        assert 'mae' in metadata, "Metadata missing 'mae'"
        assert 'rmse' in metadata, "Metadata missing 'rmse'"

        # Assert metrics are reasonable
        assert metadata['n_samples'] > 0, "Should have training samples"
        assert 0 < metadata['mae'] < 50, "MAE should be reasonable"
        assert 0 < metadata['rmse'] < 50, "RMSE should be reasonable"


def test_train_v1_2_via_bk_build_cli():
    """
    Test that train-v1-2 subcommand works via bk_build.py CLI.
    """
    project_root = Path(__file__).resolve().parents[1]
    cli_script = project_root / "src" / "bk_build.py"
    output_dir = project_root / "output" / "models" / "v1_2"
    model_file = output_dir / "ball_knower_v1_2_model.json"

    # Remove existing model file if present
    if model_file.exists():
        model_file.unlink()

    # Run via bk_build CLI
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "train-v1-2",
            "--start-season", "2019",
            "--end-season", "2019",
            "--alpha", "1.0"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    # Assert CLI succeeded
    assert result.returncode == 0, \
        f"bk_build train-v1-2 failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"

    # Assert model file was created
    assert model_file.exists(), \
        f"Model file not created at {model_file}"


# ============================================================================
# TEST V1.2 TRAINING + BACKTEST INTEGRATION
# ============================================================================

def test_train_then_backtest_v1_2():
    """
    Integration test: Train v1.2, then run backtest to verify it loads correctly.

    Tests the full pipeline:
    1. Train v1.2 model
    2. Use run_backtest_v1_2 to verify model can be loaded and used
    """
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "src" / "train_v1_2.py"
    backtest_script = project_root / "src" / "run_backtests.py"
    model_file = project_root / "output" / "models" / "v1_2" / "ball_knower_v1_2_model.json"
    backtest_output = project_root / "output" / "test_backtest_v1_2_integration.csv"

    # Clean up previous test artifacts
    if model_file.exists():
        model_file.unlink()
    if backtest_output.exists():
        backtest_output.unlink()

    # Step 1: Train model on restricted data
    train_result = subprocess.run(
        [
            "python",
            str(train_script),
            "--start-season", "2019",
            "--end-season", "2019",
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert train_result.returncode == 0, \
        f"Training failed:\n{train_result.stdout}\n{train_result.stderr}"

    assert model_file.exists(), \
        f"Model file not created at {model_file}"

    # Step 2: Run backtest using the trained model
    backtest_result = subprocess.run(
        [
            "python",
            str(backtest_script),
            "--model", "v1.2",
            "--start-season", "2019",
            "--end-season", "2019",
            "--edge-threshold", "0.0",
            "--output", str(backtest_output)
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert backtest_result.returncode == 0, \
        f"Backtest failed:\n{backtest_result.stdout}\n{backtest_result.stderr}"

    assert backtest_output.exists(), \
        f"Backtest output not created at {backtest_output}"

    # Step 3: Validate backtest output
    df = pd.read_csv(backtest_output)

    # Assert it has exactly 1 row (single season)
    assert len(df) == 1, \
        f"Backtest should have 1 row for single season, got {len(df)}"

    # Assert required columns exist
    required_cols = ['season', 'model', 'n_games', 'mae_vs_vegas', 'rmse_vs_vegas']
    for col in required_cols:
        assert col in df.columns, \
            f"Backtest output missing column: {col}"

    # Assert values are reasonable
    assert df['season'].iloc[0] == 2019
    assert df['model'].iloc[0] == 'v1.2'
    assert df['n_games'].iloc[0] > 0
    assert 0 < df['mae_vs_vegas'].iloc[0] < 20  # MAE should be reasonable

    # Clean up test artifacts
    if backtest_output.exists():
        backtest_output.unlink()


# ============================================================================
# TEST MODEL JSON STRUCTURE
# ============================================================================

def test_model_json_structure_matches_backtest_expectations():
    """
    Test that trained model JSON structure exactly matches what backtest expects.

    This is a critical test to ensure training and backtest are compatible.
    """
    project_root = Path(__file__).resolve().parents[1]
    model_file = project_root / "output" / "models" / "v1_2" / "ball_knower_v1_2_model.json"

    # Skip if model doesn't exist (run train test first)
    if not model_file.exists():
        pytest.skip(f"Model file not found at {model_file}. Run training test first.")

    # Load model JSON
    with open(model_file, 'r') as f:
        model_json = json.load(f)

    # Assert structure matches run_backtest_v1_2 expectations
    # From run_backtests.py lines 177-186:
    # intercept = model_params['intercept']
    # coefs = model_params['coefficients']
    # df['bk_v1_2_spread'] = intercept + \
    #     (df['nfelo_diff'] * coefs['nfelo_diff']) + ...

    assert 'intercept' in model_json
    assert 'coefficients' in model_json
    assert isinstance(model_json['intercept'], (int, float))
    assert isinstance(model_json['coefficients'], dict)

    # Assert all expected coefficients are present
    expected_coefs = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
    ]

    for coef_name in expected_coefs:
        assert coef_name in model_json['coefficients'], \
            f"Missing coefficient: {coef_name}"
        assert isinstance(model_json['coefficients'][coef_name], (int, float)), \
            f"Coefficient {coef_name} should be numeric"
