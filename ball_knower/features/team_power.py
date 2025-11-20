"""
Team power rating features for Ball Knower.

Utilities to attach blended team power ratings to games DataFrames.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ball_knower.io.loaders import load_team_power_ratings


def attach_team_power_ratings(
    games: pd.DataFrame,
    season: int,
    week: int,
    rating_column: str = "bk_blended_rating",
    how: Literal["left", "inner"] = "left",
) -> pd.DataFrame:
    """
    Attach BK team power ratings to a single-week games frame.

    Parameters
    ----------
    games : pd.DataFrame
        Must contain at least ['season', 'week', 'home_team', 'away_team'].
        Assumed to be filtered to a single (season, week) pair that matches
        the ratings file.
    season : int
        Season year, e.g. 2025.
    week : int
        Week number, e.g. 12.
    rating_column : str, default "bk_blended_rating"
        Which numeric rating column from the ratings file to attach as
        home/away ratings. For now this will be 'bk_blended_rating',
        but the function is generic.
    how : {"left", "inner"}, default "left"
        How to merge ratings in. Typically 'left'.

    Returns
    -------
    pd.DataFrame
        Copy of `games` with three new columns:
            - bk_rating_home
            - bk_rating_away
            - bk_rating_diff (home - away)
    """
    required_cols = {"season", "week", "home_team", "away_team"}
    missing = required_cols.difference(games.columns)
    if missing:
        raise ValueError(f"games is missing required columns: {sorted(missing)}")

    # Sanity: ensure we are really on the expected slice
    unique_seasons = games["season"].unique()
    unique_weeks = games["week"].unique()
    if len(unique_seasons) != 1 or unique_seasons[0] != season:
        raise ValueError(
            f"games must contain exactly season={season}, "
            f"found {unique_seasons.tolist()}"
        )
    if len(unique_weeks) != 1 or unique_weeks[0] != week:
        raise ValueError(
            f"games must contain exactly week={week}, "
            f"found {unique_weeks.tolist()}"
        )

    ratings = load_team_power_ratings(season=season, week=week)

    if rating_column not in ratings.columns:
        raise ValueError(
            f"rating_column '{rating_column}' not found in ratings columns: "
            f"{sorted(ratings.columns)}"
        )

    # Minimal frame: team_code + chosen rating column
    ratings_small = ratings[["team_code", rating_column]].rename(
        columns={rating_column: "bk_rating"}
    )

    # Merge for home team
    df = games.copy()
    df = df.merge(
        ratings_small.rename(
            columns={
                "team_code": "home_team",
                "bk_rating": "bk_rating_home",
            }
        ),
        on="home_team",
        how=how,
    )

    # Merge for away team
    df = df.merge(
        ratings_small.rename(
            columns={
                "team_code": "away_team",
                "bk_rating": "bk_rating_away",
            }
        ),
        on="away_team",
        how=how,
    )

    # Compute differential (home minus away)
    df["bk_rating_diff"] = df["bk_rating_home"] - df["bk_rating_away"]

    return df
