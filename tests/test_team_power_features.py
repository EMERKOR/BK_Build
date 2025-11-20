"""
Test ball_knower.features.team_power module.

Tests for attaching team power ratings to games DataFrames.
"""

import pandas as pd
import pytest

from ball_knower.features.team_power import attach_team_power_ratings
from ball_knower.io.loaders import load_team_power_ratings


def test_attach_team_power_ratings_basic():
    """
    Test that attach_team_power_ratings correctly merges ratings into a games frame.
    """
    season = 2025
    week = 12

    ratings = load_team_power_ratings(season=season, week=week)
    assert ratings["team_code"].nunique() == 32

    # Build a tiny fake games frame for this week
    sample_games = pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "home_team": "LAR",
                "away_team": "KC",
                "game_id": "2025_12_LAR_KC",
            },
            {
                "season": season,
                "week": week,
                "home_team": "DET",
                "away_team": "PHI",
                "game_id": "2025_12_DET_PHI",
            },
        ]
    )

    enriched = attach_team_power_ratings(
        games=sample_games,
        season=season,
        week=week,
        rating_column="bk_blended_rating",
    )

    # Columns exist
    for col in ["bk_rating_home", "bk_rating_away", "bk_rating_diff"]:
        assert col in enriched.columns

    # Ratings are not null
    assert not enriched["bk_rating_home"].isna().any()
    assert not enriched["bk_rating_away"].isna().any()

    # Check that diff is consistent for at least one game
    row = enriched.loc[enriched["game_id"] == "2025_12_LAR_KC"].iloc[0]
    assert pytest.approx(row["bk_rating_diff"], abs=1e-6) == (
        row["bk_rating_home"] - row["bk_rating_away"]
    )


def test_attach_team_power_ratings_missing_columns():
    """
    Test that attach_team_power_ratings raises ValueError for missing columns.
    """
    season = 2025
    week = 12

    # Missing 'away_team' column
    bad_games = pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "home_team": "LAR",
                "game_id": "2025_12_LAR_KC",
            }
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        attach_team_power_ratings(
            games=bad_games,
            season=season,
            week=week,
        )

    assert "missing required columns" in str(exc_info.value)


def test_attach_team_power_ratings_wrong_season():
    """
    Test that attach_team_power_ratings raises ValueError for wrong season.
    """
    season = 2025
    week = 12

    # Games from wrong season
    wrong_season_games = pd.DataFrame(
        [
            {
                "season": 2024,  # Wrong season
                "week": week,
                "home_team": "LAR",
                "away_team": "KC",
                "game_id": "2024_12_LAR_KC",
            }
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        attach_team_power_ratings(
            games=wrong_season_games,
            season=season,
            week=week,
        )

    assert "must contain exactly season" in str(exc_info.value)


def test_attach_team_power_ratings_diff_calculation():
    """
    Test that bk_rating_diff is calculated correctly (home - away).
    """
    season = 2025
    week = 12

    # Load ratings to get actual values
    ratings = load_team_power_ratings(season=season, week=week)
    lar_rating = ratings.loc[ratings["team_code"] == "LAR", "bk_blended_rating"].iloc[0]
    kc_rating = ratings.loc[ratings["team_code"] == "KC", "bk_blended_rating"].iloc[0]

    # Create a single game
    sample_games = pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "home_team": "LAR",
                "away_team": "KC",
                "game_id": "2025_12_LAR_KC",
            }
        ]
    )

    enriched = attach_team_power_ratings(
        games=sample_games,
        season=season,
        week=week,
    )

    # Check values
    row = enriched.iloc[0]
    assert row["bk_rating_home"] == pytest.approx(lar_rating, abs=1e-6)
    assert row["bk_rating_away"] == pytest.approx(kc_rating, abs=1e-6)
    assert row["bk_rating_diff"] == pytest.approx(lar_rating - kc_rating, abs=1e-6)
