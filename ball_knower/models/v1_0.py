"""
Ball Knower v1.0 - Deterministic Spread Model

A simple, interpretable model combining nfelo and Substack power ratings.
No machine learning - just weighted components based on pre-game features.

Public API:
    - load_week_data(season, week, data_dir): Load and preprocess all source data
    - build_week_lines(season, week, data_dir, hfa): Generate game-level spread predictions
"""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np

from ball_knower.io import loaders


def load_week_data(
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Wrapper around loaders.load_all_sources() with additional preprocessing.

    Args:
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory (defaults to repo/data/current_season)

    Returns:
        Dictionary containing:
            - All original sources from load_all_sources()
            - 'merged_ratings': Team-level ratings merged into single DataFrame
            - 'schedule': Game schedule for the season
    """
    # Load all team-level ratings
    data = loaders.load_all_sources(season, week, data_dir)

    # Load schedule
    project_root = Path(__file__).resolve().parents[2]
    schedule_path = project_root / "data" / "cache" / f"schedules_{season}.csv"

    if schedule_path.exists():
        schedule = pd.read_csv(schedule_path)
        data["schedule"] = schedule
    else:
        raise FileNotFoundError(
            f"Schedule file not found: {schedule_path}. "
            f"Cannot build game-level predictions without schedule."
        )

    return data


def _calculate_spread_components(
    games: pd.DataFrame,
    hfa: float = 1.5,
) -> pd.DataFrame:
    """
    Calculate individual spread components from team ratings.

    Components:
        - nfelo_diff: (nfelo_home - nfelo_away) - hfa
        - substack_power_diff: substack_ovr_home - substack_ovr_away
        - epa_off_diff: EPA offensive advantage for home team
        - epa_def_diff: EPA defensive advantage for home team

    Args:
        games: DataFrame with home/away team ratings already joined
        hfa: Home field advantage in points (default: 1.5)

    Returns:
        DataFrame with component columns added
    """
    games = games.copy()

    # nfelo component (primary rating system)
    # Convention: positive spread = home team favored, negative = away team favored
    # nfelo_diff = (home_nfelo - away_nfelo) - hfa
    # Subtracting HFA accounts for home field advantage
    if "nfelo_away" in games.columns and "nfelo_home" in games.columns:
        games["nfelo_diff"] = games["nfelo_home"] - games["nfelo_away"] - hfa
    else:
        games["nfelo_diff"] = np.nan

    # Substack overall power rating component
    # Higher = stronger, so home - away gives home advantage
    # Substack columns: Off., Def., Ovr. (overall = off + def)
    if "Ovr._away" in games.columns and "Ovr._home" in games.columns:
        games["substack_power_diff"] = games["Ovr._home"] - games["Ovr._away"]
    else:
        games["substack_power_diff"] = np.nan

    # EPA per play component (offensive EPA - defensive EPA allowed)
    # nfelo columns: Play (overall EPA/play), Pass, Rush (offensive)
    # Play.1 (defensive EPA/play allowed)
    if "Play_away" in games.columns and "Play_home" in games.columns:
        # Offensive EPA advantage for home team
        games["epa_off_diff"] = games["Play_home"] - games["Play_away"]
    else:
        games["epa_off_diff"] = np.nan

    if "Play.1_away" in games.columns and "Play.1_home" in games.columns:
        # Defensive EPA advantage for home team (lower defensive EPA is better, so flip)
        games["epa_def_diff"] = games["Play.1_away"] - games["Play.1_home"]
    else:
        games["epa_def_diff"] = np.nan

    return games


def _combine_components_to_line(
    games: pd.DataFrame,
    weight_nfelo: float = 0.015,
    weight_substack: float = 1.0,
    weight_epa_off: float = 15.0,
    weight_epa_def: float = 15.0,
) -> pd.DataFrame:
    """
    Combine spread components into final bk_line prediction.

    Formula (v1.0):
        bk_line = (nfelo_diff * w_nfelo) +
                  (substack_power_diff * w_substack) +
                  (epa_off_diff * w_epa_off) +
                  (epa_def_diff * w_epa_def)

    Weights are deterministic (no training). Chosen based on:
        - nfelo: ~0.015 converts elo points to spread points (~100 elo ≈ 1.5 pts)
        - substack: 1.0 (already in spread-like scale)
        - epa: ~15.0 converts EPA/play to points (~0.1 EPA/play ≈ 1.5 pts)

    Args:
        games: DataFrame with component columns
        weight_nfelo: Weight for nfelo_diff component
        weight_substack: Weight for substack_power_diff component
        weight_epa_off: Weight for offensive EPA component
        weight_epa_def: Weight for defensive EPA component

    Returns:
        DataFrame with bk_line column added
    """
    games = games.copy()

    # Initialize bk_line to zero
    games["bk_line"] = 0.0

    # Add each component if available
    if "nfelo_diff" in games.columns:
        games["bk_line"] += games["nfelo_diff"].fillna(0) * weight_nfelo

    if "substack_power_diff" in games.columns:
        games["bk_line"] += games["substack_power_diff"].fillna(0) * weight_substack

    if "epa_off_diff" in games.columns:
        games["bk_line"] += games["epa_off_diff"].fillna(0) * weight_epa_off

    if "epa_def_diff" in games.columns:
        games["bk_line"] += games["epa_def_diff"].fillna(0) * weight_epa_def

    return games


def build_week_lines(
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None,
    hfa: float = 1.5,
    weight_nfelo: float = 0.015,
    weight_substack: float = 1.0,
    weight_epa_off: float = 15.0,
    weight_epa_def: float = 15.0,
) -> pd.DataFrame:
    """
    Generate game-level spread predictions for a specific week.

    This is the main public API for Ball Knower v1.0.

    Process:
        1. Load team ratings and schedule
        2. Filter schedule to target week
        3. Join team ratings for home and away teams
        4. Calculate spread components
        5. Combine into final bk_line prediction

    Args:
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory
        hfa: Home field advantage in points (default: 1.5)
        weight_nfelo: Weight for nfelo component (default: 0.015)
        weight_substack: Weight for Substack power rating (default: 1.0)
        weight_epa_off: Weight for offensive EPA (default: 15.0)
        weight_epa_def: Weight for defensive EPA (default: 15.0)

    Returns:
        DataFrame with columns:
            - game_id: Unique game identifier
            - season, week: Season and week numbers
            - gameday: Date of game
            - away_team, home_team: Team abbreviations
            - bk_line: Ball Knower predicted spread (positive = home favored, negative = away favored)
            - vegas_line: Vegas spread line (same convention, if available)
            - nfelo_diff: nfelo component (home - away - hfa)
            - substack_power_diff: Substack power component (home - away)
            - epa_off_diff, epa_def_diff: EPA components for home team (if available)
            - nfelo_away, nfelo_home: Individual team nfelo ratings
            - Other team rating columns

    Example:
        >>> from ball_knower.models.v1_0 import build_week_lines
        >>> predictions = build_week_lines(season=2025, week=11)
        >>> print(predictions[["game_id", "away_team", "home_team", "bk_line", "vegas_line"]])
    """
    # Load data
    data = load_week_data(season, week, data_dir)

    if "merged_ratings" not in data:
        raise ValueError(
            "merged_ratings not found in loaded data. "
            "Cannot build predictions without team ratings."
        )

    ratings = data["merged_ratings"]
    schedule = data["schedule"]

    # Deduplicate ratings (keep first occurrence of each team)
    # This handles cases where merge_team_ratings creates duplicate rows
    if "team" in ratings.columns:
        ratings = ratings.drop_duplicates(subset=["team"], keep="first")

    # Filter schedule to target week and regular season only
    week_schedule = schedule[
        (schedule["season"] == season) &
        (schedule["week"] == week) &
        (schedule["game_type"] == "REG")
    ].copy()

    if len(week_schedule) == 0:
        raise ValueError(f"No games found for season {season}, week {week}")

    # Prepare away team ratings with _away suffix
    away_ratings = ratings.copy()
    away_ratings = away_ratings.rename(
        columns={c: f"{c}_away" for c in away_ratings.columns if c != "team"}
    )
    away_ratings = away_ratings.rename(columns={"team": "away_team"})

    # Join away team ratings
    week_schedule = week_schedule.merge(
        away_ratings,
        on="away_team",
        how="left"
    )

    # Prepare home team ratings with _home suffix
    home_ratings = ratings.copy()
    home_ratings = home_ratings.rename(
        columns={c: f"{c}_home" for c in home_ratings.columns if c != "team"}
    )
    home_ratings = home_ratings.rename(columns={"team": "home_team"})

    # Join home team ratings
    week_schedule = week_schedule.merge(
        home_ratings,
        on="home_team",
        how="left"
    )

    # Calculate spread components
    week_schedule = _calculate_spread_components(week_schedule, hfa=hfa)

    # Combine into final bk_line
    week_schedule = _combine_components_to_line(
        week_schedule,
        weight_nfelo=weight_nfelo,
        weight_substack=weight_substack,
        weight_epa_off=weight_epa_off,
        weight_epa_def=weight_epa_def,
    )

    # Add vegas_line column (spread_line from schedule)
    week_schedule["vegas_line"] = week_schedule["spread_line"]

    # Select and order output columns
    output_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "home_team",
        "bk_line",
        "vegas_line",
        "nfelo_diff",
        "substack_power_diff",
    ]

    # Add optional columns if they exist
    optional_cols = [
        "epa_off_diff",
        "epa_def_diff",
        "nfelo_away",
        "nfelo_home",
        "Ovr._away",
        "Ovr._home",
    ]

    for col in optional_cols:
        if col in week_schedule.columns:
            output_cols.append(col)

    # Filter to output columns that exist
    output_cols = [c for c in output_cols if c in week_schedule.columns]

    return week_schedule[output_cols].reset_index(drop=True)
