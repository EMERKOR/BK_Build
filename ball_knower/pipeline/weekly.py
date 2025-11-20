"""
Weekly Pipeline Orchestration

High-level orchestration for running the complete weekly workflow:
- Data validation
- Predictions
- Optional backtest
- Optional PredictionTracker export

This module wraps existing functionality without duplicating model logic.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import warnings

# Add project root to path for src imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ball_knower.utils import paths
from src import check_weekly_data, run_weekly_predictions


def run_weekly_pipeline(
    season: int,
    week: int,
    model_version: str = "v1.3",
    run_backtest: bool = False,
    export_predictiontracker: bool = False,
) -> Dict:
    """
    High-level weekly pipeline orchestrator.

    Runs the complete weekly workflow:
    1. Validate data integrity
    2. Generate predictions
    3. Optionally run backtest
    4. Optionally export to PredictionTracker

    Args:
        season: NFL season year
        week: Week number
        model_version: Model version to use (default: "v1.3")
        run_backtest: Whether to run backtest for this week (default: False)
        export_predictiontracker: Whether to export to PredictionTracker (default: False)

    Returns:
        Dictionary with paths and summary:
            {
                "season": int,
                "week": int,
                "model_version": str,
                "predictions_path": Path,
                "backtest_path": Path or None,
                "predictiontracker_path": Path or None,
                "warnings": List[str]
            }

    Raises:
        RuntimeError: If data validation fails or predictions cannot be generated
    """
    result = {
        "season": season,
        "week": week,
        "model_version": model_version,
        "predictions_path": None,
        "backtest_path": None,
        "predictiontracker_path": None,
        "warnings": []
    }

    print(f"\n{'='*80}")
    print(f"WEEKLY PIPELINE: Season {season}, Week {week}")
    print(f"Model: {model_version}")
    print(f"{'='*80}")

    # Step 1: Check data integrity
    print(f"\n[1/4] Validating weekly data...")
    check_result = check_weekly_data.check_weekly_data(season, week)

    if not check_result["all_required_ok"]:
        missing = [name for name, success in check_result["checks"]["required"] if not success]
        error_msg = f"Data validation failed. Missing required files: {', '.join(missing)}"
        raise RuntimeError(error_msg)

    print("  ✓ All required data files present and valid")

    # Track optional files that are missing
    missing_optional = [name for name, success in check_result["checks"]["optional"] if not success]
    if missing_optional:
        warning_msg = f"Optional files missing: {', '.join(missing_optional)}"
        result["warnings"].append(warning_msg)
        warnings.warn(warning_msg, UserWarning)

    # Step 2: Generate predictions
    print(f"\n[2/4] Generating predictions...")

    try:
        # Load weekly data
        team_ratings, matchups, status = run_weekly_predictions.load_weekly_data(season, week)

        # Build feature matrix
        feature_df = run_weekly_predictions.build_feature_matrix(matchups, team_ratings)

        # Generate predictions
        predictions = run_weekly_predictions.generate_predictions(
            feature_df,
            model_version=model_version
        )

        # Save predictions
        predictions_df, output_file = run_weekly_predictions.save_predictions(
            predictions,
            season=season,
            week=week,
            output_path=None  # Use default path
        )

        result["predictions_path"] = output_file
        print(f"  ✓ Predictions saved to: {output_file}")
        print(f"  ✓ Generated {len(predictions_df)} game predictions")

    except Exception as e:
        error_msg = f"Prediction generation failed: {str(e)}"
        raise RuntimeError(error_msg) from e

    # Step 3: Optional backtest
    if run_backtest:
        print(f"\n[3/4] Running backtest for Week {week}...")
        print("  ⚠ Single-week backtest not yet implemented")
        print("  ⚠ Skipping backtest step")
        result["warnings"].append("Backtest requested but not yet implemented")
        result["backtest_path"] = None
    else:
        print(f"\n[3/4] Skipping backtest (not requested)")

    # Step 4: Optional PredictionTracker export
    if export_predictiontracker:
        print(f"\n[4/4] Exporting to PredictionTracker format...")
        print("  ⚠ PredictionTracker export not yet implemented")
        print("  ⚠ Skipping export step")
        result["warnings"].append("PredictionTracker export requested but not yet implemented")
        result["predictiontracker_path"] = None
    else:
        print(f"\n[4/4] Skipping PredictionTracker export (not requested)")

    # Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Predictions: {result['predictions_path']}")
    if result["warnings"]:
        print(f"\nWarnings:")
        for warning in result["warnings"]:
            print(f"  ⚠ {warning}")
    print(f"{'='*80}\n")

    return result
