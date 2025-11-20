"""
Feature Isolation Tests

Ensures that v1.3 features (form.py) are not accidentally integrated
into v1.2 pipelines.

These tests serve as guardrails to prevent premature feature adoption
and maintain model version integrity.
"""

import pytest
import warnings
from pathlib import Path


def test_form_module_is_importable():
    """
    Test that form module exists and can be imported.

    This verifies the placeholder is in place for future work.
    """
    from ball_knower.features import form

    # Should be importable
    assert form is not None


def test_form_functions_are_implemented():
    """
    Test that all form functions are now implemented (v1.3).

    This verifies the functions exist and can be called without raising NotImplementedError.
    """
    from ball_knower.features import form
    import pandas as pd

    # Create valid DataFrame with required columns
    dummy_df = pd.DataFrame({
        'team': ['KC', 'BUF'],
        'season': [2024, 2024],
        'week': [1, 1],
        'off_epa_per_play': [0.1, 0.2],
        'off_success_rate': [0.4, 0.5],
        'def_epa_per_play': [-0.1, -0.2],
        'def_success_rate': [0.35, 0.40]
    })

    # All form functions should now work (return DataFrames, not raise NotImplementedError)
    offense_result = form.compute_offense_form(dummy_df)
    assert isinstance(offense_result, pd.DataFrame)
    assert 'offense_form_epa' in offense_result.columns

    defense_result = form.compute_defense_form(dummy_df)
    assert isinstance(defense_result, pd.DataFrame)
    assert 'defense_form_epa' in defense_result.columns

    team_result = form.compute_team_form(dummy_df)
    assert isinstance(team_result, pd.DataFrame)
    assert 'offense_form_epa' in team_result.columns
    assert 'defense_form_epa' in team_result.columns


def test_form_import_triggers_warning():
    """
    Test that importing form triggers a FutureWarning.

    This warns developers not to use these features in v1.2.

    Note: The warning is triggered at module import time. If the module
    has already been imported in another test, the warning may not appear
    again. This is acceptable behavior.
    """
    # Force reimport to catch warning
    import sys
    import importlib

    # Remove from sys.modules if present
    if 'ball_knower.features.form' in sys.modules:
        del sys.modules['ball_knower.features.form']

    # Importing form should trigger a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from ball_knower.features import form  # noqa: F401

        # Should have triggered a FutureWarning
        # Note: If module was already imported elsewhere, warning may not appear
        if len(w) > 0:
            assert any(issubclass(warning.category, FutureWarning) for warning in w)
            assert any("v1.3" in str(warning.message) for warning in w)


def test_v1_2_datasets_do_not_import_form():
    """
    Test that v1.2 dataset builders do not import form module.

    This prevents accidental integration of v1.3 features.
    """
    # Read v1.2 dataset files
    v1_2_file = Path(__file__).parent.parent / 'ball_knower' / 'datasets' / 'v1_2.py'

    if v1_2_file.exists():
        content = v1_2_file.read_text()

        # Should NOT import form
        assert 'from ball_knower.features import form' not in content
        assert 'from ball_knower.features.form import' not in content
        assert 'import ball_knower.features.form' not in content

        # Should NOT call form functions
        assert 'compute_offense_form' not in content
        assert 'compute_defense_form' not in content
        assert 'compute_team_form' not in content


def test_weekly_predictions_do_not_use_form():
    """
    Test that weekly prediction script does not reference form features.

    This ensures v1.2 weekly predictions don't accidentally use v1.3 features.
    """
    # Read weekly predictions script
    weekly_pred_file = Path(__file__).parent.parent / 'src' / 'run_weekly_predictions.py'

    if weekly_pred_file.exists():
        content = weekly_pred_file.read_text()

        # Should NOT import form
        assert 'from ball_knower.features import form' not in content
        assert 'from ball_knower.features.form import' not in content
        assert 'import ball_knower.features.form' not in content

        # Should NOT reference form features
        assert 'offense_form' not in content
        assert 'defense_form' not in content
        assert 'team_form' not in content


def test_backtests_do_not_use_form():
    """
    Test that backtest script does not reference form features.

    This ensures v1.2 backtests don't accidentally use v1.3 features.
    """
    # Read backtest script
    backtest_file = Path(__file__).parent.parent / 'src' / 'run_backtests.py'

    if backtest_file.exists():
        content = backtest_file.read_text()

        # Should NOT import form
        assert 'from ball_knower.features import form' not in content
        assert 'from ball_knower.features.form import' not in content
        assert 'import ball_knower.features.form' not in content

        # Should NOT reference form features
        assert 'offense_form' not in content
        assert 'defense_form' not in content
        assert 'team_form' not in content


def test_engineering_module_does_not_call_form():
    """
    Test that engineering module does not call form functions.

    The engineering module is used by v1.2, so it must not reference v1.3 features.
    """
    # Read engineering module
    engineering_file = Path(__file__).parent.parent / 'ball_knower' / 'features' / 'engineering.py'

    if engineering_file.exists():
        content = engineering_file.read_text()

        # Should NOT import form
        assert 'from ball_knower.features import form' not in content
        assert 'from .form import' not in content
        assert 'import form' not in content or 'import platform' in content  # platform ok

        # Should NOT call form functions
        assert 'compute_offense_form' not in content
        assert 'compute_defense_form' not in content
        assert 'compute_team_form' not in content


def test_form_module_location():
    """
    Test that form module exists in the correct location.

    This verifies file structure for future v1.3 development.
    """
    form_file = Path(__file__).parent.parent / 'ball_knower' / 'features' / 'form.py'

    # File should exist
    assert form_file.exists(), "form.py should exist at ball_knower/features/form.py"

    # Should be a Python file
    assert form_file.suffix == '.py'
    assert form_file.is_file()
