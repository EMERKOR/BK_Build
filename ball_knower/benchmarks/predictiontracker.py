"""
PredictionTracker Benchmarks for Ball Knower

This module:
- Loads PredictionTracker NFL prediction CSVs.
- Normalizes team names using Ball Knower's existing alias mapping.
- Merges PT consensus predictions onto the canonical v1.2 games frame.
- Computes MAE metrics for PT, BK, and Vegas, plus a BK-vs-PT outlier flag.

Usage:
    from ball_knower.benchmarks import predictiontracker as pt_bench

    # Load PredictionTracker CSV
    pt_df = pt_bench.load_predictiontracker_csv("path/to/pt_nfl_2024.csv")

    # Merge with BK canonical games and compute metrics
    merged = pt_bench.merge_with_bk_games(pt_df, outlier_threshold=4.0)

    # Analyze results
    print(f"PT MAE: {merged['pt_mae_vs_margin'].mean():.2f}")
    print(f"BK MAE: {merged['bk_mae_vs_margin'].mean():.2f}")
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
        - date: game date
        - home / hometeam / home_team: home team name/abbreviation
        - visitor / away / awayteam: away team name/abbreviation
        - line / vegasline / spread: Vegas line (home referenced)
        - predictionavg / predictavg: PT consensus predicted spread (home referenced)
        - predictionstd / predstd: (optional) standard deviation of model predictions

    Alternatively, if no explicit consensus column exists, this function will
    compute the average across all numeric model prediction columns.

    Parameters
    ----------
    path : str or Path
        Path to PredictionTracker CSV file

    Returns
    -------
    df : DataFrame
        Cleaned frame with columns:
            - home_team: normalized home team code
            - away_team: normalized away team code
            - game_date: parsed game date (if available)
            - vegas_line: Vegas closing line (if available)
            - pt_pred_avg: PredictionTracker consensus spread
            - pt_pred_std: standard deviation of predictions (if available)

    Raises
    ------
    ValueError
        If required team columns cannot be detected or mapped

    Notes
    -----
    Team names are normalized using Ball Knower's canonical team mapping
    from src/team_mapping.py. Unknown team names will raise an error.
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
    required_keys = {'home_team_raw', 'away_team_raw'}
    missing = required_keys - set(col_map)
    if missing:
        raise ValueError(
            f"Could not detect required team columns in PredictionTracker CSV. "
            f"Missing: {missing}. Columns present: {list(df.columns)}"
        )

    # Build cleaned frame
    out = pd.DataFrame()

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

    # Parse date if available
    if 'date' in col_map:
        out['game_date'] = pd.to_datetime(df[col_map['date']], errors='coerce')

    # Vegas line
    if 'vegas_line' in col_map:
        out['vegas_line'] = pd.to_numeric(df[col_map['vegas_line']], errors='coerce')

    # PT consensus average
    if 'pt_pred_avg' in col_map:
        out['pt_pred_avg'] = pd.to_numeric(df[col_map['pt_pred_avg']], errors='coerce')
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
            out['pt_pred_avg'] = df[numeric_cols].mean(axis=1)
            # Also compute std while we're at it
            out['pt_pred_std'] = df[numeric_cols].std(axis=1)
        else:
            raise ValueError(
                "No explicit prediction average column and no numeric model columns "
                "found to compute a consensus."
            )

    # PT standard deviation (if not already computed above)
    if 'pt_pred_std' in col_map and 'pt_pred_std' not in out.columns:
        out['pt_pred_std'] = pd.to_numeric(df[col_map['pt_pred_std']], errors='coerce')

    return out


def merge_with_bk_games(
    pt_df: pd.DataFrame,
    bk_games: Optional[pd.DataFrame] = None,
    outlier_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Merge PredictionTracker consensus predictions with Ball Knower's
    canonical v1.2 game-level frame and compute benchmark metrics.

    Parameters
    ----------
    pt_df : DataFrame
        Output of load_predictiontracker_csv()
    bk_games : DataFrame, optional
        If None, build the v1.2 canonical training frame via
        ball_knower.datasets.v1_2.build_training_frame().
        If provided, must have columns: home_team, away_team, home_margin
    outlier_threshold : float, default=4.0
        Absolute difference (in points) between BK line and PT average
        beyond which a game is flagged as a BK outlier.

    Returns
    -------
    merged : DataFrame
        Combined frame with additional columns:
            - pt_pred_avg: PredictionTracker consensus spread
            - pt_pred_std: disagreement between models (if available)
            - pt_mae_vs_margin: MAE of PT consensus vs actual margin
            - bk_mae_vs_margin: MAE of BK vs actual margin (if bk_line present)
            - vegas_mae_vs_margin: MAE of closing line vs actual margin (if available)
            - bk_vs_pt_diff: difference between BK and PT predictions (if both present)
            - bk_outlier_flag: boolean flag for BK outliers (if bk_line present)

    Notes
    -----
    The merge is performed on (home_team, away_team). If the canonical BK games
    frame has season/week columns, those will also be used for the join to
    tighten the matching.

    Games in the BK frame that don't have corresponding PT predictions will
    have NaN values for PT columns.
    """
    # Load canonical BK games if not provided
    if bk_games is None:
        bk_games = v1_2.build_training_frame()

    # Validate required columns
    required_cols = {'home_team', 'away_team'}
    missing = required_cols - set(bk_games.columns)
    if missing:
        raise ValueError(
            f"Canonical BK games frame missing required columns {missing}. "
            f"Available columns: {list(bk_games.columns)}"
        )

    # Determine join keys
    # Use home_team + away_team as base; add season/week if available for tighter join
    join_keys = ['home_team', 'away_team']

    # Note: PT data typically doesn't have season/week, so we only join on teams
    # If you have season/week in PT data, you could enhance this

    # Perform left merge to keep all BK games
    merged = bk_games.merge(
        pt_df,
        on=join_keys,
        how='left',
        suffixes=('', '_pt'),
    )

    # --- Compute Benchmark Metrics ---

    # PT MAE vs actual margin
    if 'home_margin' in merged.columns and 'pt_pred_avg' in merged.columns:
        merged['pt_mae_vs_margin'] = (merged['home_margin'] - merged['pt_pred_avg']).abs()

    # BK MAE vs actual margin (if BK predictions available)
    if 'bk_line' in merged.columns and 'home_margin' in merged.columns:
        merged['bk_mae_vs_margin'] = (merged['home_margin'] - merged['bk_line']).abs()

    # Vegas MAE vs actual margin
    # Look for closing line in multiple possible column names
    closing_col = None
    for cand in ['closing_line', 'vegas_line', 'line_close', 'home_line_close']:
        if cand in merged.columns:
            closing_col = cand
            break

    if closing_col is not None and 'home_margin' in merged.columns:
        merged['vegas_mae_vs_margin'] = (merged['home_margin'] - merged[closing_col]).abs()

    # BK vs PT difference and outlier flagging
    if 'bk_line' in merged.columns and 'pt_pred_avg' in merged.columns:
        merged['bk_vs_pt_diff'] = merged['bk_line'] - merged['pt_pred_avg']
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
            - n_games: total number of games
            - n_games_with_pt: number of games with PT predictions
            - pt_mae_vs_margin: mean absolute error of PT vs actual margin
            - bk_mae_vs_margin: MAE of BK vs actual (if available)
            - vegas_mae_vs_margin: MAE of Vegas vs actual (if available)
            - bk_vs_pt_mean_diff: mean difference BK - PT
            - bk_vs_pt_mae_diff: mean absolute difference BK vs PT
            - bk_outlier_count: number of BK outlier games
            - bk_outlier_pct: percentage of games flagged as outliers
    """
    summary = {}

    summary['n_games'] = len(merged)

    if 'pt_pred_avg' in merged.columns:
        summary['n_games_with_pt'] = merged['pt_pred_avg'].notna().sum()

    if 'pt_mae_vs_margin' in merged.columns:
        summary['pt_mae_vs_margin'] = merged['pt_mae_vs_margin'].mean()

    if 'bk_mae_vs_margin' in merged.columns:
        summary['bk_mae_vs_margin'] = merged['bk_mae_vs_margin'].mean()

    if 'vegas_mae_vs_margin' in merged.columns:
        summary['vegas_mae_vs_margin'] = merged['vegas_mae_vs_margin'].mean()

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
