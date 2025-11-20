"""
Centralized Path Utilities for Ball Knower

Provides consistent path management across all Ball Knower scripts.
All functions use pathlib.Path and create directories automatically where appropriate.
"""

from pathlib import Path
from typing import Optional
from ball_knower import config


def get_output_dir() -> Path:
    """
    Get the main output directory.

    Returns:
        Path to output directory (e.g., PROJECT_ROOT/output/)

    Creates the directory if it doesn't exist.
    """
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def get_models_dir(version: Optional[str] = None) -> Path:
    """
    Get the models directory, optionally for a specific version.

    Args:
        version: Model version string (e.g., "v1_0", "v1_2"). If None, returns base models dir.

    Returns:
        Path to models directory (e.g., output/models/v1_2/)

    Creates the directory if it doesn't exist.
    """
    base_dir = get_output_dir() / 'models'

    if version is None:
        models_dir = base_dir
    else:
        # Normalize version string (v1.2 -> v1_2, v1_2 -> v1_2)
        version_normalized = version.replace('.', '_')
        models_dir = base_dir / version_normalized

    models_dir.mkdir(exist_ok=True, parents=True)
    return models_dir


def get_backtests_dir(version: Optional[str] = None) -> Path:
    """
    Get the backtests directory, optionally for a specific version.

    Args:
        version: Model version string (e.g., "v1_0", "v1_2"). If None, returns base backtests dir.

    Returns:
        Path to backtests directory (e.g., output/backtests/v1_2/)

    Creates the directory if it doesn't exist.
    """
    base_dir = get_output_dir() / 'backtests'

    if version is None:
        backtests_dir = base_dir
    else:
        # Normalize version string (v1.2 -> v1_2, v1_2 -> v1_2)
        version_normalized = version.replace('.', '_')
        backtests_dir = base_dir / version_normalized

    backtests_dir.mkdir(exist_ok=True, parents=True)
    return backtests_dir


def get_predictiontracker_dir() -> Path:
    """
    Get the PredictionTracker export directory.

    Returns:
        Path to PredictionTracker directory (e.g., output/predictiontracker/)

    Creates the directory if it doesn't exist.
    """
    pt_dir = get_output_dir() / 'predictiontracker'
    pt_dir.mkdir(exist_ok=True, parents=True)
    return pt_dir


def get_predictions_dir(version: Optional[str] = None) -> Path:
    """
    Get the predictions directory, optionally for a specific version.

    Args:
        version: Model version string (e.g., "v1_0", "v1_2"). If None, returns base predictions dir.

    Returns:
        Path to predictions directory (e.g., output/predictions/v1_2/)

    Creates the directory if it doesn't exist.
    """
    base_dir = get_output_dir() / 'predictions'

    if version is None:
        predictions_dir = base_dir
    else:
        # Normalize version string (v1.2 -> v1_2, v1_2 -> v1_2)
        version_normalized = version.replace('.', '_')
        predictions_dir = base_dir / version_normalized

    predictions_dir.mkdir(exist_ok=True, parents=True)
    return predictions_dir


def get_model_weights_path(version: str, format: str = "json") -> Path:
    """
    Get the path to a model's weights file.

    Args:
        version: Model version (e.g., "v1_2", "v1.2")
        format: File format, either "json" or "txt" (default: "json")

    Returns:
        Path to weights file (e.g., output/models/v1_2/weights.json)
    """
    if format not in ["json", "txt"]:
        raise ValueError(f"Invalid format '{format}'. Must be 'json' or 'txt'.")

    version_normalized = version.replace('.', '_')
    models_dir = get_models_dir(version)

    return models_dir / f"weights.{format}"


def get_model_artifact_path(version: str, artifact_name: str) -> Path:
    """
    Get the path to a specific model artifact.

    Args:
        version: Model version (e.g., "v1_2", "v1.2")
        artifact_name: Name of the artifact file (e.g., "model.json", "scaler.pkl")

    Returns:
        Path to artifact (e.g., output/models/v1_2/model.json)
    """
    models_dir = get_models_dir(version)
    return models_dir / artifact_name


def get_backtest_results_path(
    version: str,
    start_season: int,
    end_season: int,
    suffix: str = ""
) -> Path:
    """
    Get the path to backtest results file.

    Args:
        version: Model version (e.g., "v1_2")
        start_season: Start season year
        end_season: End season year
        suffix: Optional suffix to add before extension (e.g., "_detailed")

    Returns:
        Path to backtest results (e.g., output/backtests/v1_2/backtest_2019_2024.csv)
    """
    backtests_dir = get_backtests_dir(version)
    filename = f"backtest_{start_season}_{end_season}{suffix}.csv"
    return backtests_dir / filename


def get_prediction_file_path(
    version: str,
    season: int,
    week: int
) -> Path:
    """
    Get the path to a weekly prediction file.

    Args:
        version: Model version (e.g., "v1_2")
        season: Season year
        week: Week number

    Returns:
        Path to predictions file (e.g., output/predictions/v1_2/predictions_2025_week_11.csv)
    """
    predictions_dir = get_predictions_dir(version)
    filename = f"predictions_{season}_week_{week}.csv"
    return predictions_dir / filename


def get_predictiontracker_export_path(
    version: str,
    start_season: int,
    end_season: int
) -> Path:
    """
    Get the path to PredictionTracker export file.

    Args:
        version: Model version (e.g., "v1_2")
        start_season: Start season year
        end_season: End season year

    Returns:
        Path to PredictionTracker CSV (e.g., output/predictiontracker/v1_2_2019_2024.csv)
    """
    pt_dir = get_predictiontracker_dir()
    filename = f"{version.replace('.', '_')}_{start_season}_{end_season}.csv"
    return pt_dir / filename
