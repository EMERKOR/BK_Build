"""
Subjective Data Loaders

Utilities for loading and merging subjective/narrative inputs with
Ball Knower canonical game datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def load_subjective_team_week(
    path: str = "subjective/subjective_team_week.csv"
) -> pd.DataFrame:
    """
    Load team-week subjective assessments.

    This includes health/depth sliders, QB functionality, coaching stability,
    motivation, and scheme family categorizations.

    Args:
        path: Path to subjective_team_week.csv (relative to project root or absolute)

    Returns:
        DataFrame with columns:
            - season (int): NFL season year
            - week (int): NFL week number
            - team (str): Team code
            - ol_health (int): Offensive line health slider
            - skill_health (int): Skill position health slider
            - front7_health (int): Front seven health slider
            - secondary_health (int): Secondary health slider
            - ol_depth_quality (int): OL depth quality slider
            - skill_depth_quality (int): Skill depth quality slider
            - front7_depth_quality (int): Front 7 depth quality slider
            - secondary_depth_quality (int): Secondary depth quality slider
            - qb_functionality (int): QB functionality slider
            - coaching_stability (int): Coaching stability slider
            - motivation (int): Team motivation slider
            - team_stability_index (int): Overall team stability index
            - off_scheme_family (str): Offensive scheme family
            - def_scheme_family (str): Defensive scheme family
            - notes (str): Free-text notes

    If the file doesn't exist or is empty (header only), returns an empty
    DataFrame with the correct schema.

    Example:
        >>> df = load_subjective_team_week()
        >>> df.head()
    """
    # Resolve path (support both relative and absolute)
    csv_path = Path(path)
    if not csv_path.is_absolute():
        # Assume relative to project root
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / path

    # Define schema
    schema = {
        'season': 'Int64',
        'week': 'Int64',
        'team': 'str',
        'ol_health': 'Int64',
        'skill_health': 'Int64',
        'front7_health': 'Int64',
        'secondary_health': 'Int64',
        'ol_depth_quality': 'Int64',
        'skill_depth_quality': 'Int64',
        'front7_depth_quality': 'Int64',
        'secondary_depth_quality': 'Int64',
        'qb_functionality': 'Int64',
        'coaching_stability': 'Int64',
        'motivation': 'Int64',
        'team_stability_index': 'Int64',
        'off_scheme_family': 'str',
        'def_scheme_family': 'str',
        'notes': 'str',
    }

    # If file doesn't exist, create empty DataFrame with schema
    if not csv_path.exists():
        return pd.DataFrame(columns=list(schema.keys())).astype(schema)

    # Load CSV
    df = pd.read_csv(csv_path, dtype=schema)

    # If empty (only headers), return empty DataFrame
    if len(df) == 0:
        return pd.DataFrame(columns=list(schema.keys())).astype(schema)

    return df


def load_subjective_game_week(
    path: str = "subjective/subjective_game_week.csv"
) -> pd.DataFrame:
    """
    Load game-week subjective assessments.

    This includes mismatch flags and matchup-specific narrative notes.

    Args:
        path: Path to subjective_game_week.csv (relative to project root or absolute)

    Returns:
        DataFrame with columns:
            - season (int): NFL season year
            - week (int): NFL week number
            - game_id (str): Unique game identifier
            - home_team (str): Home team code
            - away_team (str): Away team code
            - home_trench_mismatch_flag (int): Home trench advantage flag (-1, 0, +1)
            - away_trench_mismatch_flag (int): Away trench advantage flag (-1, 0, +1)
            - scheme_mismatch_flag (int): Scheme mismatch flag
            - coverage_mismatch_flag (int): Coverage mismatch flag
            - home_total_edge_flag (int): Home overall edge flag
            - away_total_edge_flag (int): Away overall edge flag
            - notes (str): Free-text matchup notes

    If the file doesn't exist or is empty (header only), returns an empty
    DataFrame with the correct schema.

    Example:
        >>> df = load_subjective_game_week()
        >>> df.head()
    """
    # Resolve path (support both relative and absolute)
    csv_path = Path(path)
    if not csv_path.is_absolute():
        # Assume relative to project root
        project_root = Path(__file__).resolve().parents[2]
        csv_path = project_root / path

    # Define schema
    schema = {
        'season': 'Int64',
        'week': 'Int64',
        'game_id': 'str',
        'home_team': 'str',
        'away_team': 'str',
        'home_trench_mismatch_flag': 'Int64',
        'away_trench_mismatch_flag': 'Int64',
        'scheme_mismatch_flag': 'Int64',
        'coverage_mismatch_flag': 'Int64',
        'home_total_edge_flag': 'Int64',
        'away_total_edge_flag': 'Int64',
        'notes': 'str',
    }

    # If file doesn't exist, create empty DataFrame with schema
    if not csv_path.exists():
        return pd.DataFrame(columns=list(schema.keys())).astype(schema)

    # Load CSV
    df = pd.read_csv(csv_path, dtype=schema)

    # If empty (only headers), return empty DataFrame
    if len(df) == 0:
        return pd.DataFrame(columns=list(schema.keys())).astype(schema)

    return df


def merge_subjective_with_games(
    games: pd.DataFrame,
    subjective_team_week: Optional[pd.DataFrame] = None,
    subjective_game_week: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge subjective team-week and game-week inputs into the canonical games DataFrame.

    Team-week data is merged twice (once for home team, once for away team) with
    columns prefixed as `home_` and `away_` for sliders.

    Game-week data is merged on (season, week, game_id).

    Args:
        games: Canonical Ball Knower games DataFrame
            Must include: season, week, game_id, home_team, away_team
        subjective_team_week: Optional team-week subjective data
            If None, will attempt to load from default path
        subjective_game_week: Optional game-week subjective data
            If None, will attempt to load from default path

    Returns:
        Merged DataFrame with subjective columns added

    Example:
        >>> from ball_knower.datasets import v1_2
        >>> games = v1_2.build_training_frame(start_year=2024, end_year=2025)
        >>> merged = merge_subjective_with_games(games)
        >>> # Check for new subjective columns
        >>> print([c for c in merged.columns if 'health' in c or 'mismatch' in c])

    Notes:
        - Team-week sliders are prefixed with 'home_' or 'away_'
        - Game-week flags are not prefixed (already home/away specific)
        - Missing subjective data results in NaN values for those columns
        - The merge is conservative: existing columns in games are not overwritten
    """
    # Load subjective data if not provided
    if subjective_team_week is None:
        subjective_team_week = load_subjective_team_week()

    if subjective_game_week is None:
        subjective_game_week = load_subjective_game_week()

    # Start with a copy of games
    result = games.copy()

    # Merge team-week data for home team
    if len(subjective_team_week) > 0:
        # Prepare home team merge
        home_tw = subjective_team_week.copy()
        home_tw_cols = [c for c in home_tw.columns if c not in ['season', 'week', 'team']]

        # Prefix all non-key columns with 'home_'
        home_tw = home_tw.rename(
            columns={c: f'home_{c}' for c in home_tw_cols}
        )

        # Merge on (season, week, home_team=team)
        result = result.merge(
            home_tw,
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_home_dup')
        )

        # Drop the duplicate 'team' column from merge
        if 'team' in result.columns:
            result = result.drop(columns=['team'])

        # Prepare away team merge
        away_tw = subjective_team_week.copy()
        away_tw_cols = [c for c in away_tw.columns if c not in ['season', 'week', 'team']]

        # Prefix all non-key columns with 'away_'
        away_tw = away_tw.rename(
            columns={c: f'away_{c}' for c in away_tw_cols}
        )

        # Merge on (season, week, away_team=team)
        result = result.merge(
            away_tw,
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_away_dup')
        )

        # Drop the duplicate 'team' column from merge
        if 'team' in result.columns:
            result = result.drop(columns=['team'])

    # Merge game-week data
    if len(subjective_game_week) > 0:
        # Game-week columns already have home/away prefixes built-in
        # Merge on (season, week, game_id)
        gw_cols = [c for c in subjective_game_week.columns
                   if c not in ['season', 'week', 'game_id', 'home_team', 'away_team']]

        result = result.merge(
            subjective_game_week[['season', 'week', 'game_id'] + gw_cols],
            on=['season', 'week', 'game_id'],
            how='left',
            suffixes=('', '_gw_dup')
        )

    return result
