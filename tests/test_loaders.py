"""
Test ball_knower.io.loaders module

Sanity checks for:
- Unified loader functions
- File resolution and normalization
- Merge behavior and audit warnings
"""

import pytest
import pandas as pd
import warnings
from pathlib import Path

from ball_knower.io import loaders


# ============================================================================
# TEST DATA AVAILABILITY
# ============================================================================

def check_current_season_data_exists(season: int = 2025, week: int = 11) -> bool:
    """
    Check if current-season data files exist for the specified season/week.

    Returns:
        True if all expected files exist, False otherwise
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
# TEST LOAD_ALL_SOURCES
# ============================================================================

def test_load_all_sources():
    """
    Test that load_all_sources returns expected structure when data exists.

    Uses fixture data from tests/fixtures/current_season/.
    """
    season = 2025
    week = 1

    # Load all sources
    data = loaders.load_all_sources(season=season, week=week)

    # Assert result is a dict
    assert isinstance(data, dict), "load_all_sources should return a dict"

    # Assert expected keys exist
    expected_keys = [
        "power_ratings_nfelo",
        "epa_tiers_nfelo",
        "strength_of_schedule_nfelo",
        "power_ratings_substack",
        "qb_epa_substack",
        "weekly_projections_ppg_substack",
        "merged_ratings",
    ]

    for key in expected_keys:
        assert key in data, f"Expected key '{key}' in load_all_sources result"

    # Assert each value is a non-empty DataFrame
    for key in ["power_ratings_nfelo", "epa_tiers_nfelo", "qb_epa_substack"]:
        assert isinstance(data[key], pd.DataFrame), \
            f"{key} should be a DataFrame"
        assert len(data[key]) > 0, \
            f"{key} should be non-empty"
        assert "team" in data[key].columns, \
            f"{key} should have a 'team' column"


def test_load_all_sources_merged_ratings():
    """
    Test that merged_ratings key contains properly merged data.

    Uses fixture data from tests/fixtures/current_season/.
    """
    season = 2025
    week = 1

    # Load all sources
    data = loaders.load_all_sources(season=season, week=week)

    # Assert merged_ratings exists
    assert "merged_ratings" in data, "Expected 'merged_ratings' key"

    merged = data["merged_ratings"]

    # Assert it's a non-empty DataFrame
    assert isinstance(merged, pd.DataFrame), "merged_ratings should be a DataFrame"
    assert len(merged) > 0, "merged_ratings should be non-empty"

    # Assert it has a team column
    assert "team" in merged.columns, "merged_ratings should have a 'team' column"

    # Assert it has more columns than just 'team' (merged from multiple sources)
    assert len(merged.columns) > 5, \
        "merged_ratings should have multiple columns from different sources"


# ============================================================================
# TEST MERGE AUDIT BEHAVIOR
# ============================================================================

def test_merge_team_ratings_no_warning_when_teams_match():
    """
    Test that merge_team_ratings does not emit warnings when all teams match.
    """
    # Create dummy DataFrames with matching teams
    base_df = pd.DataFrame({
        'team': ['BUF', 'KC', 'PHI'],
        'nfelo': [1600, 1650, 1580]
    })

    epa_df = pd.DataFrame({
        'team': ['BUF', 'KC', 'PHI'],
        'epa_margin': [0.1, 0.2, 0.05]
    })

    sources = {
        "power_ratings_nfelo": base_df,
        "epa_tiers_nfelo": epa_df,
    }

    # Merge should not emit warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = loaders.merge_team_ratings(sources)

        # Filter to UserWarnings about merge audit
        merge_warnings = [
            warning for warning in w
            if issubclass(warning.category, UserWarning)
            and "Merge audit" in str(warning.message)
        ]

        assert len(merge_warnings) == 0, \
            "No merge audit warnings should be raised when all teams match"

    # Assert merge succeeded
    assert len(result) == 3, "Merge should have 3 rows"
    assert "team" in result.columns, "Result should have 'team' column"
    assert "nfelo" in result.columns, "Result should have 'nfelo' column"
    assert "epa_margin" in result.columns, "Result should have 'epa_margin' column"


def test_merge_team_ratings_warning_when_teams_differ():
    """
    Test that merge_team_ratings emits a warning when right frame has unmatched teams.
    """
    # Create dummy DataFrames with non-matching teams
    base_df = pd.DataFrame({
        'team': ['BUF', 'KC'],
        'nfelo': [1600, 1650]
    })

    epa_df = pd.DataFrame({
        'team': ['BUF', 'KC', 'PHI'],  # PHI is not in base
        'epa_margin': [0.1, 0.2, 0.05]
    })

    sources = {
        "power_ratings_nfelo": base_df,
        "epa_tiers_nfelo": epa_df,
    }

    # Merge should emit a warning about unmatched team
    with pytest.warns(UserWarning, match="Merge audit.*PHI"):
        result = loaders.merge_team_ratings(sources)

    # Assert merge succeeded (left join keeps base teams only)
    assert len(result) == 2, "Merge should have 2 rows (left join)"
    assert "PHI" not in result['team'].values, \
        "PHI should not be in result (left join behavior)"


# ============================================================================
# TEST FILE RESOLUTION
# ============================================================================

def test_resolve_file_missing_raises_error():
    """
    Test that _resolve_file raises FileNotFoundError when no file exists.
    """
    with pytest.raises(FileNotFoundError):
        loaders._resolve_file(
            category="power_ratings",
            provider="nonexistent_provider",
            season=9999,
            week=99,
            data_dir=Path("/tmp/nonexistent_dir")
        )
