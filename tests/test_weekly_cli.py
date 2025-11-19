"""
Test run_weekly_predictions.py CLI

Smoke tests for the weekly predictions CLI:
- Verify CLI runs without errors
- Check output file is created
- Validate output CSV structure
"""

import pytest
import subprocess
import pandas as pd
from pathlib import Path


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_current_season_data_exists(season: int = 2025, week: int = 11) -> bool:
    """
    Check if current-season data files exist for the specified season/week.
    """
    data_dir = Path(__file__).resolve().parents[1] / "data" / "current_season"

    expected_files = [
        f"power_ratings_nfelo_{season}_week_{week}.csv",
        f"epa_tiers_nfelo_{season}_week_{week}.csv",
        f"strength_of_schedule_nfelo_{season}_week_{week}.csv",
        f"power_ratings_substack_{season}_week_{week}.csv",
        f"qb_epa_substack_{season}_week_{week}.csv",
        f"weekly_projections_ppg_substack_{season}_week_{week}.csv",
    ]

    for filename in expected_files:
        filepath = data_dir / filename
        if not filepath.exists():
            return False

    return True


# ============================================================================
# TEST WEEKLY PREDICTIONS CLI
# ============================================================================

def test_weekly_predictions_cli_smoke_test():
    """
    Smoke test for run_weekly_predictions.py CLI.

    Verifies:
    - CLI runs without errors
    - Output file is created
    - Output CSV has expected columns

    Uses fixture data from tests/fixtures/current_season/.
    """
    season = 2025
    week = 1

    # Define output path
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"test_predictions_{season}_week_{week}.csv"

    # Remove output file if it exists from previous test
    if output_path.exists():
        output_path.unlink()

    # Run CLI
    cli_script = project_root / "src" / "run_weekly_predictions.py"
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "--season", str(season),
            "--week", str(week),
            "--output", str(output_path)
        ],
        capture_output=True,
        text=True,
        timeout=30  # 30 second timeout
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

    # Assert it's not empty
    assert len(df) > 0, "Output CSV should not be empty"

    # Assert required columns exist
    required_cols = ["game_id", "bk_line", "vegas_line", "edge"]

    for col in required_cols:
        assert col in df.columns, \
            f"Output CSV missing required column: {col}"

    # Assert numeric columns are numeric
    for col in ["bk_line", "vegas_line", "edge"]:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"Column '{col}' should be numeric"

    # Clean up test output file
    if output_path.exists():
        output_path.unlink()


def test_weekly_predictions_cli_with_default_output():
    """
    Test that CLI works with default output path.

    Uses fixture data from tests/fixtures/current_season/.
    """
    season = 2025
    week = 1

    project_root = Path(__file__).resolve().parents[1]
    cli_script = project_root / "src" / "run_weekly_predictions.py"

    # Run CLI without --output flag (uses default)
    result = subprocess.run(
        [
            "python",
            str(cli_script),
            "--season", str(season),
            "--week", str(week),
        ],
        capture_output=True,
        text=True,
        timeout=30
    )

    # Assert CLI succeeded
    assert result.returncode == 0, \
        f"CLI failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"

    # Default output path should be output/predictions_{season}_week_{week}.csv
    default_output = project_root / "output" / f"predictions_{season}_week_{week}.csv"

    # Assert output file exists
    assert default_output.exists(), \
        f"Default output file not created at {default_output}"

    # Clean up
    if default_output.exists():
        default_output.unlink()


def test_weekly_predictions_cli_help():
    """
    Test that CLI --help flag works.
    """
    project_root = Path(__file__).resolve().parents[1]
    cli_script = project_root / "src" / "run_weekly_predictions.py"

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
    assert "--season" in result.stdout or "--season" in result.stderr, \
        "Help text should mention --season flag"
    assert "--week" in result.stdout or "--week" in result.stderr, \
        "Help text should mention --week flag"
