"""
Test ball_knower.datasets.v1_2_bk module.

Tests for v1.2+BK dataset builder with team power ratings.
"""

import pytest
import pandas as pd

from ball_knower.datasets import v1_2_bk


# Use historical data for testing since 2025 games haven't been played yet
TEST_SEASON = 2024
TEST_WEEK = 12


def test_v1_2_bk_requires_week():
    """
    Test that week parameter is required.
    """
    season = TEST_SEASON

    with pytest.raises(ValueError) as exc_info:
        v1_2_bk.build_training_frame(season=season, week=None)

    assert "requires week parameter" in str(exc_info.value)


@pytest.mark.skip(reason="2025 Week 12 data not available in nfelo yet - will enable when live")
def test_v1_2_bk_build_training_frame_week12_2025():
    """
    Test v1.2+BK dataset builder for Week 12 2025 (sandbox target).

    This test is skipped until 2025 Week 12 data is available.
    """
    season = 2025
    week = 12

    # Build dataset
    df = v1_2_bk.build_training_frame(season=season, week=week)

    # Should have data
    assert len(df) > 0, "Should have games for Week 12 2025"

    # Should have BK rating columns
    bk_cols = ['bk_rating_home', 'bk_rating_away', 'bk_rating_diff']
    for col in bk_cols:
        assert col in df.columns, f"Missing BK rating column: {col}"

