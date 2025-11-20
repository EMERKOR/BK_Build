"""
Test src/run_backtests.py CLI

Smoke tests for the unified backtest driver:
- Verify CLI runs without errors
- Check output file is created
- Validate output CSV structure
"""

import pytest
import subprocess
import pandas as pd
from pathlib import Path


# ============================================================================
# TEST BACKTEST CLI - v1.0
# ============================================================================

def test_backtest_cli_v1_0_smoke_test():
    """
    Smoke test for run_backtests.py with v1.0 model.

    Runs a single season (2019) to keep test fast.
    """
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_backtest_v1_0_2019.csv"

    # Remove output file if it exists from previous test
    if output_path.exists():
        output_path.unlink()

    # Run backtest CLI
    cli_script = project_root / "src" / "run_backtests.py"
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "--start-season", "2019",
            "--end-season", "2019",
            "--model", "v1.0",
            "--edge-threshold", "0.5",
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True,
        timeout=60  # 60 second timeout (downloads nfelo data)
    )

    # Assert CLI succeeded
    assert result.returncode == 0, \
        f"CLI failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"

    # Assert output file was created
    assert output_path.exists(), \
        f"Output file not created at {output_path}"

    # Read output CSV
    df = pd.read_csv(output_path)

    # Assert it has exactly 1 row (single season)
    assert len(df) == 1, \
        f"Output should have 1 row for single season, got {len(df)}"

    # Assert required columns exist
    required_cols = [
        "season",
        "model",
        "edge_threshold",
        "n_games",
        "n_bets",
        "mae_vs_vegas",
    ]

    for col in required_cols:
        assert col in df.columns, \
            f"Output CSV missing required column: {col}"

    # Assert values are reasonable
    assert df['season'].iloc[0] == 2019, "Season should be 2019"
    assert df['model'].iloc[0] == 'v1.0', "Model should be v1.0"
    assert df['edge_threshold'].iloc[0] == 0.5, "Edge threshold should be 0.5"
    assert df['n_games'].iloc[0] > 0, "Should have games"
    assert df['n_bets'].iloc[0] >= 0, "Should have non-negative bets"
    assert 0 < df['mae_vs_vegas'].iloc[0] < 20, "MAE should be reasonable"

    # Clean up test output file
    if output_path.exists():
        output_path.unlink()


# ============================================================================
# TEST BACKTEST CLI - v1.2
# ============================================================================

def test_backtest_cli_v1_2_smoke_test():
    """
    Smoke test for run_backtests.py with v1.2 model.

    Requires that ball_knower_v1_2_model.json exists in output/.
    """
    # Check if v1.2 model file exists
    project_root = Path(__file__).resolve().parents[1]
    model_file = project_root / "output" / "ball_knower_v1_2_model.json"

    if not model_file.exists():
        pytest.skip(
            f"v1.2 model file not found at {model_file}. "
            "Run ball_knower_v1_2.py to train the model first, "
            "or skip this test."
        )

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_backtest_v1_2_2019.csv"

    # Remove output file if it exists from previous test
    if output_path.exists():
        output_path.unlink()

    # Run backtest CLI
    cli_script = project_root / "src" / "run_backtests.py"
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "--start-season", "2019",
            "--end-season", "2019",
            "--model", "v1.2",
            "--edge-threshold", "0.5",
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Assert CLI succeeded
    assert result.returncode == 0, \
        f"CLI failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"

    # Assert output file was created
    assert output_path.exists(), \
        f"Output file not created at {output_path}"

    # Read output CSV
    df = pd.read_csv(output_path)

    # Assert it has exactly 1 row
    assert len(df) == 1, \
        f"Output should have 1 row for single season, got {len(df)}"

    # Assert required columns exist
    required_cols = [
        "season",
        "model",
        "edge_threshold",
        "n_games",
        "n_bets",
        "mae_vs_vegas",
    ]

    for col in required_cols:
        assert col in df.columns, \
            f"Output CSV missing required column: {col}"

    # Assert values are reasonable
    assert df['season'].iloc[0] == 2019, "Season should be 2019"
    assert df['model'].iloc[0] == 'v1.2', "Model should be v1.2"
    assert df['edge_threshold'].iloc[0] == 0.5, "Edge threshold should be 0.5"
    assert df['n_games'].iloc[0] > 0, "Should have games"

    # Clean up test output file
    if output_path.exists():
        output_path.unlink()


# ============================================================================
# TEST BACKTEST CLI - v1.3
# ============================================================================

def test_backtest_cli_v1_3_smoke_test():
    """
    Smoke test for run_backtests.py with v1.3 model.

    Requires that model.pkl and features.json exist in output/models/v1_3/.
    Tests only 2019 to keep test fast.
    """
    # Check if v1.3 model files exist
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "output" / "models" / "v1_3"
    model_file = model_dir / "model.pkl"
    features_file = model_dir / "features.json"

    if not model_file.exists() or not features_file.exists():
        pytest.skip(
            f"v1.3 model files not found at {model_dir}. "
            "Run 'python src/bk_build.py train-v1-3' to train the model first, "
            "or skip this test."
        )

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_backtest_v1_3_2019.csv"

    # Remove output file if it exists from previous test
    if output_path.exists():
        output_path.unlink()

    # Run backtest CLI
    cli_script = project_root / "src" / "run_backtests.py"
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "--start-season", "2019",
            "--end-season", "2019",
            "--model", "v1.3",
            "--edge-threshold", "0.5",
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True,
        timeout=120  # v1.3 may take longer due to dataset building
    )

    # Assert CLI succeeded
    assert result.returncode == 0, \
        f"CLI failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"

    # Assert output file was created
    assert output_path.exists(), \
        f"Output file not created at {output_path}"

    # Read output CSV
    df = pd.read_csv(output_path)

    # Assert it has exactly 1 row
    assert len(df) == 1, \
        f"Output should have 1 row for single season, got {len(df)}"

    # Assert required columns exist
    required_cols = [
        "season",
        "model",
        "edge_threshold",
        "n_games",
        "n_bets",
        "mae_vs_vegas",
        "rmse_vs_vegas",
        "mean_edge",
    ]

    for col in required_cols:
        assert col in df.columns, \
            f"Output CSV missing required column: {col}"

    # Assert values are reasonable
    assert df['season'].iloc[0] == 2019, "Season should be 2019"
    assert df['model'].iloc[0] == 'v1.3', "Model should be v1.3"
    assert df['edge_threshold'].iloc[0] == 0.5, "Edge threshold should be 0.5"
    assert df['n_games'].iloc[0] > 0, "Should have games"
    assert df['n_bets'].iloc[0] >= 0, "Should have non-negative bets"
    assert 0 < df['mae_vs_vegas'].iloc[0] < 20, "MAE should be reasonable"

    # Clean up test output file
    if output_path.exists():
        output_path.unlink()


def test_backtest_v1_3_rejects_early_seasons():
    """
    Test that v1.3 backtest properly rejects seasons before 2013.

    v1.3 requires team-week EPA data which starts in 2013.
    """
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "output" / "models" / "v1_3"
    model_file = model_dir / "model.pkl"

    if not model_file.exists():
        pytest.skip("v1.3 model not found. Skip early season test.")

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_backtest_v1_3_2010.csv"

    # Try to run backtest for 2010 (before EPA data availability)
    cli_script = project_root / "src" / "run_backtests.py"
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "--start-season", "2010",
            "--end-season", "2010",
            "--model", "v1.3",
            "--edge-threshold", "0.0",
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True,
        timeout=30
    )

    # Assert CLI failed with appropriate error
    assert result.returncode != 0, \
        "v1.3 backtest should reject seasons before 2013"

    # Assert error message mentions 2013 requirement
    error_output = result.stdout + result.stderr
    assert "2013" in error_output, \
        "Error message should mention 2013 as minimum season for v1.3"

    # Clean up if output file was somehow created
    if output_path.exists():
        output_path.unlink()


# ============================================================================
# TEST CLI HELP
# ============================================================================

def test_backtest_cli_help():
    """
    Test that CLI --help flag works.
    """
    project_root = Path(__file__).resolve().parents[1]
    cli_script = project_root / "src" / "run_backtests.py"

    # Run with --help
    result = subprocess.run(
        ["python", str(cli_script), "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )

    # Assert success
    assert result.returncode == 0, \
        f"CLI --help failed with return code {result.returncode}"

    # Assert help text contains expected keywords
    help_text = result.stdout + result.stderr
    assert "--start-season" in help_text, "Help text should mention --start-season"
    assert "--end-season" in help_text, "Help text should mention --end-season"
    assert "--model" in help_text, "Help text should mention --model"
    assert "--edge-threshold" in help_text, "Help text should mention --edge-threshold"
