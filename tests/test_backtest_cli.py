"""
Test src/run_backtests.py CLI

Smoke tests for the unified backtest driver:
- Verify CLI runs without errors
- Check output file is created
- Validate output CSV structure
- Regression test for coefficient sign bug
"""

import pytest
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))


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


# ============================================================================
# REGRESSION TEST: COEFFICIENT SIGN BUG
# ============================================================================

def test_v1_0_nfelo_sign_behavior():
    """
    Regression test for v1.0 coefficient sign bug.

    Ensures that the nfelo→spread mapping has the correct sign:
    - Higher nfelo_diff (stronger home team) → more NEGATIVE spread (bigger home favorite)
    - Lower nfelo_diff (weaker home team) → more POSITIVE spread (bigger home underdog)

    This test caught the bug where coefficients were:
    - WRONG: NFELO_COEF = +0.0447 (positive correlation - backwards!)
    - RIGHT: NFELO_COEF = -0.042 (negative correlation - correct!)
    """
    # Import the v1.0 backtest function to access coefficients
    from run_backtests import run_backtest_v1_0

    # Read the source code to extract coefficients
    import inspect
    source = inspect.getsource(run_backtest_v1_0)

    # Extract coefficient values from source
    import re
    nfelo_match = re.search(r'NFELO_COEF\s*=\s*([-+]?\d+\.?\d*)', source)
    intercept_match = re.search(r'INTERCEPT\s*=\s*([-+]?\d+\.?\d*)', source)

    assert nfelo_match is not None, "Could not find NFELO_COEF in source code"
    assert intercept_match is not None, "Could not find INTERCEPT in source code"

    nfelo_coef = float(nfelo_match.group(1))
    intercept = float(intercept_match.group(1))

    print(f"\nExtracted coefficients from v1.0 model:")
    print(f"  NFELO_COEF = {nfelo_coef}")
    print(f"  INTERCEPT = {intercept}")

    # Test sign behavior with synthetic nfelo differences
    # nfelo_diff = starting_nfelo_home - starting_nfelo_away

    # Scenario 1: Home team is MUCH stronger
    strong_home_nfelo_diff = 100  # Home team has +100 ELO advantage
    strong_home_spread = intercept + (strong_home_nfelo_diff * nfelo_coef)

    # Scenario 2: Home team is MUCH weaker
    weak_home_nfelo_diff = -100  # Home team has -100 ELO disadvantage
    weak_home_spread = intercept + (weak_home_nfelo_diff * nfelo_coef)

    print(f"\nSign behavior test:")
    print(f"  Strong home (nfelo_diff=+100): predicted spread = {strong_home_spread:.2f}")
    print(f"  Weak home (nfelo_diff=-100): predicted spread = {weak_home_spread:.2f}")

    # CRITICAL ASSERTION: Strong home team should have MORE NEGATIVE spread
    # (i.e., bigger favorite)
    assert strong_home_spread < weak_home_spread, \
        f"Sign bug detected! Strong home team (spread={strong_home_spread:.2f}) " \
        f"should have MORE NEGATIVE spread than weak home team (spread={weak_home_spread:.2f}). " \
        f"This indicates NFELO_COEF has the wrong sign."

    # Additional check: Coefficient should be negative
    assert nfelo_coef < 0, \
        f"NFELO_COEF should be negative (found {nfelo_coef}). " \
        f"Positive coefficient produces inverted predictions."

    print(f"  ✓ Sign behavior is correct!")
    print(f"    Stronger home team → more negative spread (bigger favorite)")

    # Verify realistic magnitude (should be small, around 0.04)
    assert abs(nfelo_coef) < 0.1, \
        f"NFELO_COEF magnitude seems unrealistic: {nfelo_coef}. " \
        f"Expected magnitude around 0.04"
