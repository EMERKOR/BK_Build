"""
PredictionTracker Benchmarks for Ball Knower v1.2

This module:
- Loads PredictionTracker NFL prediction CSVs
- Normalizes team names using Ball Knower's existing alias mapping
- Merges PT consensus predictions onto the canonical v1.2 games frame using
  (season, week, home_team, away_team) for precise 1:1 matching
- Computes MAE metrics vs actual game results and Vegas lines

Improvements over v1.0:
- Uses canonical v1.2 dataset WITH actual scores and Vegas lines
- Merges on (season, week, home_team, away_team) instead of just teams
- Computes MAE vs actual game margin (not just vs Vegas)
- Reports matched/unmatched game counts

Usage:
    from ball_knower.benchmarks import predictiontracker as pt_bench

    # Load PredictionTracker CSV
    pt_df = pt_bench.load_predictiontracker_csv("path/to/pt_nfl_2024.csv")

    # Merge with BK canonical games and compute metrics
    merged = pt_bench.merge_with_bk_games(pt_df, outlier_threshold=4.0)

    # Compute summary
    summary = pt_bench.compute_summary_metrics(merged)
    print(summary)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import pandas as pd
import numpy as np

# Import team normalization from the existing mapping
import sys
import importlib.util

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Direct import of team_mapping to avoid circular dependencies
_team_mapping_path = _PROJECT_ROOT / "src" / "team_mapping.py"
_spec = importlib.util.spec_from_file_location("team_mapping", _team_mapping_path)
_team_mapping = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_team_mapping)
normalize_team = _team_mapping.normalize_team_name

# Import v1.2 dataset builder
from ball_knower.datasets import v1_2


def load_predictiontracker_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a PredictionTracker NFL CSV and return a cleaned DataFrame
    with canonical column names and normalized team codes.

    Expected minimal columns (names may vary slightly between seasons):
        - date: game date (REQUIRED for season/week extraction)
        - home / hometeam / home_team: home team name/abbreviation
        - visitor / away / awayteam: away team name/abbreviation
        - line / vegasline / spread: Vegas line (home referenced)
        - predictionavg / predictavg: PT consensus predicted spread (home referenced)
        - predictionstd / predstd: (optional) standard deviation of model predictions

    Parameters
    ----------
    path : str or Path
        Path to PredictionTracker CSV file

    Returns
    -------
    df : DataFrame
        Cleaned frame with columns:
            - season: NFL season year (extracted from date)
            - week: NFL week number (extracted from date)
            - home_team: normalized home team code
            - away_team: normalized away team code
            - game_date: parsed game date
            - pt_spread: PredictionTracker consensus spread
            - pt_spread_std: standard deviation of predictions (if available)
            - vegas_spread_pt: Vegas line from PT CSV (if available)

    Raises
    ------
    ValueError
        If required columns cannot be detected or mapped

    Notes
    -----
    Team names are normalized using Ball Knower's canonical team mapping
    from src/team_mapping.py. Unknown team names will raise an error.

    Season and week are extracted from the game date using NFL calendar logic:
    - NFL season starts in September
    - Games in Jan/Feb belong to previous year's season
    - Week numbers are approximate based on date
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PredictionTracker CSV not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names for robust detection
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}

    # Detect basic columns
    for col in df.columns:
        col_clean = col.replace('_', '').replace(' ', '')

        # Home team
        if col_clean in {'home', 'hometeam'}:
            col_map['home_team_raw'] = col
        # Away team
        elif col_clean in {'visitor', 'away', 'awayteam', 'road', 'visitorteam'}:
            col_map['away_team_raw'] = col
        # Date
        elif col_clean in {'date', 'gamedate'}:
            col_map['date'] = col
        # Vegas line
        elif col_clean in {'line', 'vegasline', 'spread', 'closingline'}:
            col_map['vegas_line'] = col
        # PT consensus average
        elif col_clean in {'predictionavg', 'predictavg', 'predavg', 'consensus', 'average'}:
            col_map['pt_pred_avg'] = col
        # PT standard deviation
        elif col_clean in {'predictionstd', 'predstd', 'std', 'stdev'}:
            col_map['pt_pred_std'] = col

    # Validate required columns
    required_keys = {'home_team_raw', 'away_team_raw', 'date'}
    missing = required_keys - set(col_map)
    if missing:
        raise ValueError(
            f"Could not detect required columns in PredictionTracker CSV. "
            f"Missing: {missing}. Columns present: {list(df.columns)}"
        )

    # Build cleaned frame
    out = pd.DataFrame()

    # Parse date FIRST so we can extract season/week
    out['game_date'] = pd.to_datetime(df[col_map['date']], errors='coerce')

    # Extract season and week from game date
    # NFL season logic: games in Jan/Feb belong to previous year's season
    out['season'] = out['game_date'].apply(lambda d: d.year if d.month >= 9 else d.year - 1)

    # Approximate week number based on date within season
    # NFL regular season typically starts first week of September
    # Week 1 = early Sept, Week 18 = early Jan
    def get_nfl_week(date):
        """Approximate NFL week from game date."""
        if pd.isna(date):
            return np.nan

        # Determine season start (first Thursday of September)
        season = date.year if date.month >= 9 else date.year - 1
        sept_1 = pd.Timestamp(year=season, month=9, day=1)

        # Find first Thursday (typically Week 1 kickoff)
        days_to_thursday = (3 - sept_1.weekday()) % 7
        season_start = sept_1 + pd.Timedelta(days=days_to_thursday)

        # Calculate weeks since season start
        days_since_start = (date - season_start).days
        week = (days_since_start // 7) + 1

        # Clamp to reasonable range (1-18)
        return max(1, min(18, week))

    out['week'] = out['game_date'].apply(get_nfl_week)

    # Normalize team names using canonical mapping
    try:
        out['home_team'] = df[col_map['home_team_raw']].astype(str).str.strip().map(normalize_team)
        out['away_team'] = df[col_map['away_team_raw']].astype(str).str.strip().map(normalize_team)
    except Exception as e:
        raise ValueError(
            f"Error normalizing team names. Ensure all teams in the CSV are valid NFL teams. "
            f"Original error: {e}"
        )

    # Check for any unmapped teams
    unmapped_home = out['home_team'].isna().sum()
    unmapped_away = out['away_team'].isna().sum()
    if unmapped_home > 0 or unmapped_away > 0:
        # Show which teams failed to map
        bad_homes = df.loc[out['home_team'].isna(), col_map['home_team_raw']].unique()
        bad_aways = df.loc[out['away_team'].isna(), col_map['away_team_raw']].unique()
        raise ValueError(
            f"Could not normalize {unmapped_home + unmapped_away} team names. "
            f"Unknown home teams: {list(bad_homes)}. "
            f"Unknown away teams: {list(bad_aways)}. "
            f"Update src/team_mapping.py if needed."
        )

    # PT consensus average (rename to pt_spread for clarity)
    if 'pt_pred_avg' in col_map:
        out['pt_spread'] = pd.to_numeric(df[col_map['pt_pred_avg']], errors='coerce')
    else:
        # No explicit average column - try to compute from numeric model columns
        # Exclude known non-prediction columns
        exclude_cols = set(col_map.values())
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        if numeric_cols:
            warnings.warn(
                f"No explicit prediction average column found. "
                f"Computing consensus from {len(numeric_cols)} numeric columns: {numeric_cols[:5]}...",
                UserWarning
            )
            out['pt_spread'] = df[numeric_cols].mean(axis=1)
            # Also compute std while we're at it
            out['pt_spread_std'] = df[numeric_cols].std(axis=1)
        else:
            raise ValueError(
                "No explicit prediction average column and no numeric model columns "
                "found to compute a consensus."
            )

    # PT standard deviation (if not already computed above)
    if 'pt_pred_std' in col_map and 'pt_spread_std' not in out.columns:
        out['pt_spread_std'] = pd.to_numeric(df[col_map['pt_pred_std']], errors='coerce')

    # Vegas line from PT CSV (if available)
    if 'vegas_line' in col_map:
        out['vegas_spread_pt'] = pd.to_numeric(df[col_map['vegas_line']], errors='coerce')

    return out


def merge_with_bk_games(
    pt_df: pd.DataFrame,
    bk_games: Optional[pd.DataFrame] = None,
    outlier_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Merge PredictionTracker consensus predictions with Ball Knower's
    canonical v1.2 game-level frame and compute benchmark metrics.

    Uses (season, week, home_team, away_team) as merge key for precise 1:1 matching.

    Parameters
    ----------
    pt_df : DataFrame
        Output of load_predictiontracker_csv()
    bk_games : DataFrame, optional
        If None, build the v1.2 canonical training frame via
        ball_knower.datasets.v1_2.build_training_frame().
        If provided, must have columns: season, week, home_team, away_team, home_margin
    outlier_threshold : float, default=4.0
        Absolute difference (in points) between BK line and PT spread
        beyond which a game is flagged as a BK outlier.

    Returns
    -------
    merged : DataFrame
        Combined frame with columns from both BK games and PT data, plus:
            - pt_spread: PredictionTracker consensus spread
            - mae_pt_vs_actual: MAE of PT spread vs actual game margin
            - mae_vegas_vs_actual: MAE of Vegas closing spread vs actual margin
            - mae_bk_vs_actual: MAE of BK line vs actual margin (if bk_line present)
            - bk_vs_pt_diff: difference between BK and PT predictions (if both present)
            - bk_outlier_flag: boolean flag for BK outliers (if bk_line present)

    Notes
    -----
    The merge is performed on (season, week, home_team, away_team) to ensure
    each PT game matches at most one BK game.

    Games in pt_df that don't match any BK game will be dropped.
    Games in bk_games that don't have PT predictions will have NaN for PT columns.
    """
    # Load canonical BK games if not provided
    if bk_games is None:
        print("\nLoading canonical v1.2 games dataset...")
        bk_games = v1_2.build_training_frame()

    # Validate required columns in BK games
    required_bk_cols = {'season', 'week', 'home_team', 'away_team', 'home_margin', 'vegas_closing_spread'}
    missing = required_bk_cols - set(bk_games.columns)
    if missing:
        raise ValueError(
            f"Canonical BK games frame missing required columns {missing}. "
            f"Available columns: {list(bk_games.columns)}"
        )

    # Validate required columns in PT data
    required_pt_cols = {'season', 'week', 'home_team', 'away_team', 'pt_spread'}
    missing_pt = required_pt_cols - set(pt_df.columns)
    if missing_pt:
        raise ValueError(
            f"PredictionTracker frame missing required columns {missing_pt}. "
            f"Available columns: {list(pt_df.columns)}"
        )

    # Merge on (season, week, home_team, away_team) for precise matching
    join_keys = ['season', 'week', 'home_team', 'away_team']

    print(f"\nMerging PT predictions with BK games on {join_keys}...")
    print(f"  PT games: {len(pt_df)}")
    print(f"  BK games: {len(bk_games)}")

    # Perform inner merge to only keep matched games
    merged = pt_df.merge(
        bk_games,
        on=join_keys,
        how='inner',
        suffixes=('_pt', '_bk'),
        validate='1:1',  # Ensure 1:1 matching
    )

    print(f"  Matched games: {len(merged)}")
    print(f"  PT unmatched: {len(pt_df) - len(merged)}")
    print(f"  BK unmatched: {len(bk_games) - len(merged)}")

    # --- Compute MAE metrics vs actual game results ---

    # PT MAE vs actual margin
    if 'home_margin' in merged.columns and 'pt_spread' in merged.columns:
        # Note: spread predicts margin, so MAE = |actual_margin - predicted_spread|
        merged['mae_pt_vs_actual'] = (merged['home_margin'] - merged['pt_spread']).abs()

    # Vegas MAE vs actual margin
    if 'home_margin' in merged.columns and 'vegas_closing_spread' in merged.columns:
        merged['mae_vegas_vs_actual'] = (merged['home_margin'] - merged['vegas_closing_spread']).abs()

    # BK MAE vs actual margin (if BK predictions available)
    if 'bk_line' in merged.columns and 'home_margin' in merged.columns:
        merged['mae_bk_vs_actual'] = (merged['home_margin'] - merged['bk_line']).abs()

    # --- Compute MAE metrics comparing predictions to each other ---

    # BK vs Vegas
    if 'bk_line' in merged.columns and 'vegas_closing_spread' in merged.columns:
        merged['mae_bk_vs_vegas'] = (merged['bk_line'] - merged['vegas_closing_spread']).abs()

    # PT vs Vegas
    if 'pt_spread' in merged.columns and 'vegas_closing_spread' in merged.columns:
        merged['mae_pt_vs_vegas'] = (merged['pt_spread'] - merged['vegas_closing_spread']).abs()

    # BK vs PT difference and outlier flagging
    if 'bk_line' in merged.columns and 'pt_spread' in merged.columns:
        merged['bk_vs_pt_diff'] = merged['bk_line'] - merged['pt_spread']
        merged['bk_outlier_flag'] = merged['bk_vs_pt_diff'].abs() > outlier_threshold

    return merged


def compute_summary_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary benchmark metrics from a merged BK + PT frame.

    Parameters
    ----------
    merged : DataFrame
        Output of merge_with_bk_games()

    Returns
    -------
    summary : DataFrame
        Single-row DataFrame with summary metrics:
            - n_games: total number of matched games
            - mae_pt_vs_actual: mean absolute error of PT vs actual margin
            - mae_vegas_vs_actual: MAE of Vegas vs actual
            - mae_bk_vs_actual: MAE of BK vs actual (if available)
            - mae_bk_vs_vegas: MAE of BK vs Vegas (if available)
            - mae_pt_vs_vegas: MAE of PT vs Vegas
            - bk_vs_pt_mean_diff: mean difference BK - PT
            - bk_vs_pt_mae_diff: mean absolute difference BK vs PT
            - bk_outlier_count: number of BK outlier games
            - bk_outlier_pct: percentage of games flagged as outliers
    """
    summary = {}

    summary['n_games'] = len(merged)

    # MAE vs actual game results
    if 'mae_pt_vs_actual' in merged.columns:
        summary['mae_pt_vs_actual'] = merged['mae_pt_vs_actual'].mean()

    if 'mae_vegas_vs_actual' in merged.columns:
        summary['mae_vegas_vs_actual'] = merged['mae_vegas_vs_actual'].mean()

    if 'mae_bk_vs_actual' in merged.columns:
        summary['mae_bk_vs_actual'] = merged['mae_bk_vs_actual'].mean()

    # MAE comparing predictions to each other
    if 'mae_bk_vs_vegas' in merged.columns:
        summary['mae_bk_vs_vegas'] = merged['mae_bk_vs_vegas'].mean()

    if 'mae_pt_vs_vegas' in merged.columns:
        summary['mae_pt_vs_vegas'] = merged['mae_pt_vs_vegas'].mean()

    # BK vs PT comparison
    if 'bk_vs_pt_diff' in merged.columns:
        summary['bk_vs_pt_mean_diff'] = merged['bk_vs_pt_diff'].mean()
        summary['bk_vs_pt_mae_diff'] = merged['bk_vs_pt_diff'].abs().mean()

    if 'bk_outlier_flag' in merged.columns:
        summary['bk_outlier_count'] = int(merged['bk_outlier_flag'].sum())
        summary['bk_outlier_pct'] = (
            summary['bk_outlier_count'] / len(merged) * 100
            if len(merged) > 0 else 0.0
        )

    return pd.DataFrame([summary])
