"""
Test ball_knower.pipeline.weekly module

Tests the weekly pipeline orchestration without heavy IO or long runtimes.
Uses mocks to stub underlying data check, prediction, backtest, and export functions.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import argparse

from ball_knower.pipeline import weekly


# ============================================================================
# TEST run_weekly_pipeline FUNCTION
# ============================================================================

@patch('ball_knower.pipeline.weekly.run_weekly_predictions.save_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.generate_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.build_feature_matrix')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_basic(
    mock_check_data,
    mock_load_weekly_data,
    mock_build_feature_matrix,
    mock_generate_predictions,
    mock_save_predictions
):
    """
    Test that run_weekly_pipeline calls expected components with correct parameters.
    """
    # Mock check_weekly_data to return success
    mock_check_data.return_value = {
        "all_required_ok": True,
        "checks": {
            "required": [("power_ratings_nfelo", True), ("epa_tiers_nfelo", True)],
            "optional": [("sos_nfelo", False)]
        },
        "return_code": 0
    }

    # Mock run_weekly_predictions components
    mock_team_ratings = MagicMock()
    mock_matchups = MagicMock()
    mock_feature_df = MagicMock()
    mock_predictions = MagicMock()
    mock_predictions_df = MagicMock()
    mock_output_file = Path("output/predictions/v1_3/predictions_2025_week_11.csv")

    mock_load_weekly_data.return_value = (mock_team_ratings, mock_matchups, {})
    mock_build_feature_matrix.return_value = mock_feature_df
    mock_generate_predictions.return_value = mock_predictions
    mock_save_predictions.return_value = (mock_predictions_df, mock_output_file)

    # Run pipeline
    result = weekly.run_weekly_pipeline(
        season=2025,
        week=11,
        model_version="v1.3",
        run_backtest=False,
        export_predictiontracker=False
    )

    # Verify check_weekly_data was called
    mock_check_data.assert_called_once_with(2025, 11)

    # Verify run_weekly_predictions components were called
    mock_load_weekly_data.assert_called_once_with(2025, 11)
    mock_build_feature_matrix.assert_called_once_with(mock_matchups, mock_team_ratings)
    mock_generate_predictions.assert_called_once_with(
        mock_feature_df,
        model_version="v1.3"
    )
    mock_save_predictions.assert_called_once()

    # Verify returned dict has expected structure
    assert isinstance(result, dict)
    assert result["season"] == 2025
    assert result["week"] == 11
    assert result["model_version"] == "v1.3"
    assert result["predictions_path"] == mock_output_file
    assert result["backtest_path"] is None
    assert result["predictiontracker_path"] is None
    assert isinstance(result["warnings"], list)


@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_data_validation_failure(mock_check_data, mock_load_weekly_data):
    """
    Test that run_weekly_pipeline raises RuntimeError when data validation fails.
    """
    # Mock check_weekly_data to return failure
    mock_check_data.return_value = {
        "all_required_ok": False,
        "checks": {
            "required": [("power_ratings_nfelo", False), ("epa_tiers_nfelo", True)],
            "optional": []
        },
        "return_code": 1
    }

    # Run pipeline and expect RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        weekly.run_weekly_pipeline(
            season=2025,
            week=11,
            model_version="v1.3"
        )

    # Verify error message mentions missing files
    assert "Data validation failed" in str(exc_info.value)
    assert "power_ratings_nfelo" in str(exc_info.value)

    # Verify check_weekly_data was called
    mock_check_data.assert_called_once_with(2025, 11)

    # Verify predictions were NOT called (pipeline should exit early)
    mock_load_weekly_data.assert_not_called()


@patch('ball_knower.pipeline.weekly.run_weekly_predictions.save_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.generate_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.build_feature_matrix')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_with_optional_files_missing(
    mock_check_data,
    mock_load_weekly_data,
    mock_build_feature_matrix,
    mock_generate_predictions,
    mock_save_predictions
):
    """
    Test that run_weekly_pipeline adds warnings when optional files are missing.
    """
    # Mock check_weekly_data to return success with missing optional files
    mock_check_data.return_value = {
        "all_required_ok": True,
        "checks": {
            "required": [("power_ratings_nfelo", True), ("epa_tiers_nfelo", True)],
            "optional": [("sos_nfelo", False), ("qb_epa_substack", False)]
        },
        "return_code": 0
    }

    # Mock run_weekly_predictions components
    mock_team_ratings = MagicMock()
    mock_matchups = MagicMock()
    mock_feature_df = MagicMock()
    mock_predictions_obj = MagicMock()
    mock_predictions_df = MagicMock()
    mock_output_file = Path("output/predictions/v1_3/predictions_2025_week_11.csv")

    mock_load_weekly_data.return_value = (mock_team_ratings, mock_matchups, {})
    mock_build_feature_matrix.return_value = mock_feature_df
    mock_generate_predictions.return_value = mock_predictions_obj
    mock_save_predictions.return_value = (mock_predictions_df, mock_output_file)

    # Run pipeline
    result = weekly.run_weekly_pipeline(
        season=2025,
        week=11,
        model_version="v1.3"
    )

    # Verify warnings are present for missing optional files
    assert len(result["warnings"]) > 0
    assert any("sos_nfelo" in warning for warning in result["warnings"])
    assert any("qb_epa_substack" in warning for warning in result["warnings"])


@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_prediction_failure(mock_check_data, mock_load_weekly_data):
    """
    Test that run_weekly_pipeline raises RuntimeError when prediction generation fails.
    """
    # Mock check_weekly_data to return success
    mock_check_data.return_value = {
        "all_required_ok": True,
        "checks": {
            "required": [("power_ratings_nfelo", True)],
            "optional": []
        },
        "return_code": 0
    }

    # Mock run_weekly_predictions to raise exception
    mock_load_weekly_data.side_effect = Exception("Data loading failed")

    # Run pipeline and expect RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        weekly.run_weekly_pipeline(
            season=2025,
            week=11,
            model_version="v1.3"
        )

    # Verify error message mentions prediction failure
    assert "Prediction generation failed" in str(exc_info.value)


@patch('ball_knower.pipeline.weekly.run_weekly_predictions.save_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.generate_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.build_feature_matrix')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_with_backtest_flag(
    mock_check_data,
    mock_load_weekly_data,
    mock_build_feature_matrix,
    mock_generate_predictions,
    mock_save_predictions
):
    """
    Test that run_weekly_pipeline handles backtest flag (currently not implemented).
    """
    # Mock check_weekly_data to return success
    mock_check_data.return_value = {
        "all_required_ok": True,
        "checks": {
            "required": [("power_ratings_nfelo", True)],
            "optional": []
        },
        "return_code": 0
    }

    # Mock run_weekly_predictions
    mock_team_ratings = MagicMock()
    mock_matchups = MagicMock()
    mock_feature_df = MagicMock()
    mock_predictions_obj = MagicMock()
    mock_predictions_df = MagicMock()
    mock_output_file = Path("output/predictions/v1_3/predictions_2025_week_11.csv")

    mock_load_weekly_data.return_value = (mock_team_ratings, mock_matchups, {})
    mock_build_feature_matrix.return_value = mock_feature_df
    mock_generate_predictions.return_value = mock_predictions_obj
    mock_save_predictions.return_value = (mock_predictions_df, mock_output_file)

    # Run pipeline with backtest flag
    result = weekly.run_weekly_pipeline(
        season=2025,
        week=11,
        model_version="v1.3",
        run_backtest=True  # Request backtest
    )

    # Verify backtest path is None (not yet implemented)
    assert result["backtest_path"] is None

    # Verify warning about backtest not implemented
    assert any("Backtest" in warning and "not yet implemented" in warning for warning in result["warnings"])


@patch('ball_knower.pipeline.weekly.run_weekly_predictions.save_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.generate_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.build_feature_matrix')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_with_export_flag(
    mock_check_data,
    mock_load_weekly_data,
    mock_build_feature_matrix,
    mock_generate_predictions,
    mock_save_predictions
):
    """
    Test that run_weekly_pipeline handles export-predictiontracker flag (currently not implemented).
    """
    # Mock check_weekly_data to return success
    mock_check_data.return_value = {
        "all_required_ok": True,
        "checks": {
            "required": [("power_ratings_nfelo", True)],
            "optional": []
        },
        "return_code": 0
    }

    # Mock run_weekly_predictions
    mock_team_ratings = MagicMock()
    mock_matchups = MagicMock()
    mock_feature_df = MagicMock()
    mock_predictions_obj = MagicMock()
    mock_predictions_df = MagicMock()
    mock_output_file = Path("output/predictions/v1_3/predictions_2025_week_11.csv")

    mock_load_weekly_data.return_value = (mock_team_ratings, mock_matchups, {})
    mock_build_feature_matrix.return_value = mock_feature_df
    mock_generate_predictions.return_value = mock_predictions_obj
    mock_save_predictions.return_value = (mock_predictions_df, mock_output_file)

    # Run pipeline with export flag
    result = weekly.run_weekly_pipeline(
        season=2025,
        week=11,
        model_version="v1.3",
        export_predictiontracker=True  # Request export
    )

    # Verify export path is None (not yet implemented)
    assert result["predictiontracker_path"] is None

    # Verify warning about export not implemented
    assert any("PredictionTracker" in warning and "not yet implemented" in warning for warning in result["warnings"])


@patch('ball_knower.pipeline.weekly.run_weekly_predictions.save_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.generate_predictions')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.build_feature_matrix')
@patch('ball_knower.pipeline.weekly.run_weekly_predictions.load_weekly_data')
@patch('ball_knower.pipeline.weekly.check_weekly_data.check_weekly_data')
def test_run_weekly_pipeline_different_model_versions(
    mock_check_data,
    mock_load_weekly_data,
    mock_build_feature_matrix,
    mock_generate_predictions,
    mock_save_predictions
):
    """
    Test that run_weekly_pipeline passes model_version correctly to generate_predictions.
    """
    # Mock check_weekly_data to return success
    mock_check_data.return_value = {
        "all_required_ok": True,
        "checks": {
            "required": [("power_ratings_nfelo", True)],
            "optional": []
        },
        "return_code": 0
    }

    # Mock run_weekly_predictions
    mock_team_ratings = MagicMock()
    mock_matchups = MagicMock()
    mock_feature_df = MagicMock()
    mock_predictions_obj = MagicMock()
    mock_predictions_df = MagicMock()
    mock_output_file = Path("output/predictions/v1_2/predictions_2025_week_11.csv")

    mock_load_weekly_data.return_value = (mock_team_ratings, mock_matchups, {})
    mock_build_feature_matrix.return_value = mock_feature_df
    mock_generate_predictions.return_value = mock_predictions_obj
    mock_save_predictions.return_value = (mock_predictions_df, mock_output_file)

    # Run pipeline with v1.2 model
    result = weekly.run_weekly_pipeline(
        season=2025,
        week=11,
        model_version="v1.2"
    )

    # Verify model_version was passed to generate_predictions
    mock_generate_predictions.assert_called_once_with(
        mock_feature_df,
        model_version="v1.2"
    )

    # Verify result contains correct model version
    assert result["model_version"] == "v1.2"


# ============================================================================
# TEST CLI SUBCOMMAND ARGUMENT PARSING
# ============================================================================

def test_bk_build_weekly_pipeline_parser():
    """
    Test that bk_build.py weekly-pipeline subcommand parses arguments correctly.
    """
    from src.bk_build import create_parser

    parser = create_parser()

    # Test minimal args (only --week is required)
    args = parser.parse_args(['weekly-pipeline', '--week', '11'])
    assert args.command == 'weekly-pipeline'
    assert args.week == 11
    assert args.model_version == 'v1.3'  # Default
    assert args.backtest is False  # Default
    assert args.export_predictiontracker is False  # Default

    # Test all args
    args = parser.parse_args([
        'weekly-pipeline',
        '--season', '2024',
        '--week', '18',
        '--model-version', 'v1.2',
        '--backtest',
        '--export-predictiontracker'
    ])
    assert args.command == 'weekly-pipeline'
    assert args.season == 2024
    assert args.week == 18
    assert args.model_version == 'v1.2'
    assert args.backtest is True
    assert args.export_predictiontracker is True


@patch('ball_knower.pipeline.weekly.run_weekly_pipeline')
def test_cmd_weekly_pipeline_invokes_pipeline(mock_pipeline):
    """
    Test that cmd_weekly_pipeline calls run_weekly_pipeline with correct parameters.
    """
    from src.bk_build import cmd_weekly_pipeline

    # Mock return value
    mock_pipeline.return_value = {
        "season": 2025,
        "week": 11,
        "model_version": "v1.3",
        "predictions_path": Path("output/predictions_2025_week_11.csv"),
        "backtest_path": None,
        "predictiontracker_path": None,
        "warnings": []
    }

    # Create args namespace
    args = argparse.Namespace(
        season=2025,
        week=11,
        model_version="v1.3",
        backtest=False,
        export_predictiontracker=False
    )

    # Call cmd function
    exit_code = cmd_weekly_pipeline(args)

    # Verify pipeline was called with correct args
    mock_pipeline.assert_called_once_with(
        season=2025,
        week=11,
        model_version="v1.3",
        run_backtest=False,
        export_predictiontracker=False
    )

    # Verify exit code is 0 (success)
    assert exit_code == 0


@patch('ball_knower.pipeline.weekly.run_weekly_pipeline')
def test_cmd_weekly_pipeline_handles_runtime_error(mock_pipeline):
    """
    Test that cmd_weekly_pipeline returns exit code 1 when pipeline raises RuntimeError.
    """
    from src.bk_build import cmd_weekly_pipeline

    # Mock pipeline to raise RuntimeError
    mock_pipeline.side_effect = RuntimeError("Data validation failed")

    # Create args namespace
    args = argparse.Namespace(
        season=2025,
        week=11,
        model_version="v1.3",
        backtest=False,
        export_predictiontracker=False
    )

    # Call cmd function
    exit_code = cmd_weekly_pipeline(args)

    # Verify exit code is 1 (failure)
    assert exit_code == 1


# ============================================================================
# TEST check-weekly-data CLI SUBCOMMAND
# ============================================================================

def test_bk_build_check_weekly_data_parser():
    """
    Test that bk_build.py check-weekly-data subcommand parses arguments correctly.
    """
    from src.bk_build import create_parser

    parser = create_parser()

    # Test minimal args (only --week is required)
    args = parser.parse_args(['check-weekly-data', '--week', '11'])
    assert args.command == 'check-weekly-data'
    assert args.week == 11
    # season should default to CURRENT_SEASON (2025)
    assert hasattr(args, 'season')

    # Test all args
    args = parser.parse_args(['check-weekly-data', '--season', '2024', '--week', '18'])
    assert args.command == 'check-weekly-data'
    assert args.season == 2024
    assert args.week == 18


@patch('src.check_weekly_data.check_weekly_data')
def test_cmd_check_weekly_data_invokes_checker(mock_check):
    """
    Test that cmd_check_weekly_data calls check_weekly_data with correct parameters.
    """
    from src.bk_build import cmd_check_weekly_data

    # Mock return value
    mock_check.return_value = {
        "all_required_ok": True,
        "checks": {"required": [], "optional": []},
        "return_code": 0
    }

    # Create args namespace
    args = argparse.Namespace(
        season=2025,
        week=11
    )

    # Call cmd function
    exit_code = cmd_check_weekly_data(args)

    # Verify check_weekly_data was called with correct args
    mock_check.assert_called_once_with(2025, 11)

    # Verify exit code matches return_code from check
    assert exit_code == 0


@patch('src.check_weekly_data.check_weekly_data')
def test_cmd_check_weekly_data_returns_failure_code(mock_check):
    """
    Test that cmd_check_weekly_data returns exit code 1 when validation fails.
    """
    from src.bk_build import cmd_check_weekly_data

    # Mock return value with failure
    mock_check.return_value = {
        "all_required_ok": False,
        "checks": {"required": [], "optional": []},
        "return_code": 1
    }

    # Create args namespace
    args = argparse.Namespace(
        season=2025,
        week=11
    )

    # Call cmd function
    exit_code = cmd_check_weekly_data(args)

    # Verify exit code is 1 (failure)
    assert exit_code == 1
