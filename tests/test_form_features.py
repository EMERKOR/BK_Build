"""
Comprehensive Tests for Team Form Features

Tests the leak-free form feature implementations in ball_knower.features.form.
"""

import pytest
import pandas as pd
import numpy as np
import warnings


def test_compute_offense_form_basic():
    """
    Test basic offense form computation.
    """
    from ball_knower.features.form import compute_offense_form

    # Create sample team-week data
    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024, 2024, 2024],
        'week': [1, 2, 3, 4, 5],
        'off_epa_per_play': [0.1, 0.2, 0.3, 0.4, 0.5],
        'off_success_rate': [0.40, 0.45, 0.50, 0.55, 0.60]
    })

    # Suppress warning about v1.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_offense_form(df, window=4)

    # Check output columns
    assert 'team' in result.columns
    assert 'season' in result.columns
    assert 'week' in result.columns
    assert 'offense_form_epa' in result.columns
    assert 'offense_form_success' in result.columns

    # Check shape
    assert len(result) == len(df)


def test_compute_offense_form_leak_free():
    """
    Test that offense form is leak-free (current game excluded).

    Rolling calculation should use .shift(1), so week 1 should be NaN,
    and week 2 should use only week 1 data.
    """
    from ball_knower.features.form import compute_offense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024, 2024],
        'week': [1, 2, 3, 4],
        'off_epa_per_play': [0.1, 0.2, 0.3, 0.4],
        'off_success_rate': [0.40, 0.50, 0.60, 0.70]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_offense_form(df, window=4)

    # Week 1: Should be NaN (no prior data)
    assert pd.isna(result.loc[0, 'offense_form_epa'])
    assert pd.isna(result.loc[0, 'offense_form_success'])

    # Week 2: Should use only week 1 (0.1)
    assert result.loc[1, 'offense_form_epa'] == pytest.approx(0.1)
    assert result.loc[1, 'offense_form_success'] == pytest.approx(0.40)

    # Week 3: Should use weeks 1-2 average (0.1 + 0.2) / 2 = 0.15
    assert result.loc[2, 'offense_form_epa'] == pytest.approx(0.15)
    assert result.loc[2, 'offense_form_success'] == pytest.approx(0.45)

    # Week 4: Should use weeks 1-3 average (0.1 + 0.2 + 0.3) / 3 = 0.2
    assert result.loc[3, 'offense_form_epa'] == pytest.approx(0.2)
    assert result.loc[3, 'offense_form_success'] == pytest.approx(0.5)


def test_compute_defense_form_basic():
    """
    Test basic defense form computation.
    """
    from ball_knower.features.form import compute_defense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024, 2024, 2024],
        'week': [1, 2, 3, 4, 5],
        'def_epa_per_play': [-0.1, -0.2, -0.3, -0.4, -0.5],
        'def_success_rate': [0.35, 0.40, 0.45, 0.50, 0.55]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_defense_form(df, window=4)

    # Check output columns
    assert 'team' in result.columns
    assert 'defense_form_epa' in result.columns
    assert 'defense_form_success' in result.columns

    # Check shape
    assert len(result) == len(df)


def test_compute_defense_form_leak_free():
    """
    Test that defense form is leak-free (current game excluded).
    """
    from ball_knower.features.form import compute_defense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024, 2024],
        'week': [1, 2, 3, 4],
        'def_epa_per_play': [-0.1, -0.2, -0.3, -0.4],
        'def_success_rate': [0.30, 0.40, 0.50, 0.60]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_defense_form(df, window=4)

    # Week 1: Should be NaN (no prior data)
    assert pd.isna(result.loc[0, 'defense_form_epa'])

    # Week 2: Should use only week 1 (-0.1)
    assert result.loc[1, 'defense_form_epa'] == pytest.approx(-0.1)

    # Week 3: Should use weeks 1-2 average (-0.1 + -0.2) / 2 = -0.15
    assert result.loc[2, 'defense_form_epa'] == pytest.approx(-0.15)


def test_compute_team_form_combined():
    """
    Test that compute_team_form combines offense and defense correctly.
    """
    from ball_knower.features.form import compute_team_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024],
        'week': [1, 2, 3],
        'off_epa_per_play': [0.1, 0.2, 0.3],
        'off_success_rate': [0.40, 0.50, 0.60],
        'def_epa_per_play': [-0.1, -0.2, -0.3],
        'def_success_rate': [0.35, 0.40, 0.45]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_team_form(df, window=4)

    # Check all columns present
    assert 'offense_form_epa' in result.columns
    assert 'offense_form_success' in result.columns
    assert 'defense_form_epa' in result.columns
    assert 'defense_form_success' in result.columns

    # Check shape
    assert len(result) == len(df)


def test_rolling_window_parameter():
    """
    Test that the window parameter works correctly.
    """
    from ball_knower.features.form import compute_offense_form

    df = pd.DataFrame({
        'team': ['KC'] * 6,
        'season': [2024] * 6,
        'week': [1, 2, 3, 4, 5, 6],
        'off_epa_per_play': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'off_success_rate': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result_window_2 = compute_offense_form(df, window=2)
        result_window_4 = compute_offense_form(df, window=4)

    # Week 4 with window=2 should average weeks 2-3
    # (0.2 + 0.3) / 2 = 0.25
    assert result_window_2.loc[3, 'offense_form_epa'] == pytest.approx(0.25)

    # Week 5 with window=4 should average weeks 1-4
    # (0.1 + 0.2 + 0.3 + 0.4) / 4 = 0.25
    assert result_window_4.loc[4, 'offense_form_epa'] == pytest.approx(0.25)


def test_multiple_teams():
    """
    Test that form calculations are correctly grouped by team.
    """
    from ball_knower.features.form import compute_offense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'BUF', 'BUF'],
        'season': [2024, 2024, 2024, 2024],
        'week': [1, 2, 1, 2],
        'off_epa_per_play': [0.1, 0.2, 0.3, 0.4],
        'off_success_rate': [0.4, 0.5, 0.6, 0.7]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_offense_form(df, window=4)

    # KC week 1: NaN (no prior data)
    kc_week1 = result[(result['team'] == 'KC') & (result['week'] == 1)]
    assert len(kc_week1) == 1
    assert pd.isna(kc_week1['offense_form_epa'].values[0])

    # KC week 2: should be 0.1 (only KC week 1)
    kc_week2 = result[(result['team'] == 'KC') & (result['week'] == 2)]
    assert len(kc_week2) == 1
    assert kc_week2['offense_form_epa'].values[0] == pytest.approx(0.1)

    # BUF week 2: should be 0.3 (only BUF week 1)
    buf_week2 = result[(result['team'] == 'BUF') & (result['week'] == 2)]
    assert len(buf_week2) == 1
    assert buf_week2['offense_form_epa'].values[0] == pytest.approx(0.3)


def test_missing_offensive_columns():
    """
    Test graceful handling of missing offensive columns.
    """
    from ball_knower.features.form import compute_offense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC'],
        'season': [2024, 2024],
        'week': [1, 2]
        # Missing off_epa_per_play and off_success_rate
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Should issue a UserWarning but not crash
        with pytest.warns(UserWarning, match="Missing offensive metrics"):
            result = compute_offense_form(df)

    # Should return NaN for form features
    assert pd.isna(result['offense_form_epa'].iloc[0])
    assert pd.isna(result['offense_form_success'].iloc[0])


def test_missing_defensive_columns():
    """
    Test graceful handling of missing defensive columns.
    """
    from ball_knower.features.form import compute_defense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC'],
        'season': [2024, 2024],
        'week': [1, 2]
        # Missing def_epa_per_play and def_success_rate
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Should issue a UserWarning but not crash
        with pytest.warns(UserWarning, match="Missing defensive metrics"):
            result = compute_defense_form(df)

    # Should return NaN for form features
    assert pd.isna(result['defense_form_epa'].iloc[0])
    assert pd.isna(result['defense_form_success'].iloc[0])


def test_unsorted_input():
    """
    Test that function correctly sorts input data.
    """
    from ball_knower.features.form import compute_offense_form

    # Create unsorted data
    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024],
        'week': [3, 1, 2],  # Out of order
        'off_epa_per_play': [0.3, 0.1, 0.2],
        'off_success_rate': [0.6, 0.4, 0.5]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_offense_form(df, window=4)

    # After sorting, week 2 should use week 1 (0.1)
    week2_row = result[result['week'] == 2]
    assert week2_row['offense_form_epa'].iloc[0] == pytest.approx(0.1)


def test_cross_season_boundaries():
    """
    Test that rolling calculations respect team boundaries across seasons.
    """
    from ball_knower.features.form import compute_offense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC', 'KC'],
        'season': [2023, 2023, 2024, 2024],
        'week': [17, 18, 1, 2],
        'off_epa_per_play': [0.1, 0.2, 0.3, 0.4],
        'off_success_rate': [0.4, 0.5, 0.6, 0.7]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_offense_form(df, window=4)

    # 2024 week 2 should average 2023 week 17-18 and 2024 week 1
    # (0.1 + 0.2 + 0.3) / 3 = 0.2
    week2_2024 = result[(result['season'] == 2024) & (result['week'] == 2)]
    assert week2_2024['offense_form_epa'].iloc[0] == pytest.approx(0.2)


def test_nan_values_in_input():
    """
    Test handling of NaN values in input data.
    """
    from ball_knower.features.form import compute_offense_form

    df = pd.DataFrame({
        'team': ['KC', 'KC', 'KC', 'KC'],
        'season': [2024, 2024, 2024, 2024],
        'week': [1, 2, 3, 4],
        'off_epa_per_play': [0.1, np.nan, 0.3, 0.4],
        'off_success_rate': [0.4, 0.5, 0.6, 0.7]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_offense_form(df, window=4)

    # Week 3 should average weeks 1-2, but week 2 is NaN
    # So it should use only week 1: 0.1
    assert result.loc[2, 'offense_form_epa'] == pytest.approx(0.1, nan_ok=True) or pd.isna(result.loc[2, 'offense_form_epa'])


def test_output_column_names():
    """
    Test that output column names match specification.
    """
    from ball_knower.features.form import compute_team_form

    df = pd.DataFrame({
        'team': ['KC'],
        'season': [2024],
        'week': [1],
        'off_epa_per_play': [0.1],
        'off_success_rate': [0.4],
        'def_epa_per_play': [-0.1],
        'def_success_rate': [0.35]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = compute_team_form(df)

    expected_columns = {
        'team', 'season', 'week',
        'offense_form_epa', 'offense_form_success',
        'defense_form_epa', 'defense_form_success'
    }

    assert set(result.columns) == expected_columns
