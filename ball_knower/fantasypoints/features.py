"""
FantasyPoints Feature Engineering

This module aggregates raw FantasyPoints data to team-week and matchup-week levels.

Key functions:
    - build_fpd_team_week: Aggregate to team-week level (~60-80 features)
    - build_fpd_matchup_week: Build matchup features with differentials
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from ball_knower.fantasypoints import loaders

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "fantasypoints"


def _aggregate_qb_metrics(qb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate QB coverage matchup data to team level.

    Args:
        qb_df: QB coverage matchup DataFrame

    Returns:
        Team-level QB metrics
    """
    # Group by team and aggregate (taking the primary QB's stats)
    agg_dict = {}

    # Build aggregation dictionary dynamically based on available columns
    for col in qb_df.columns:
        if col in ['season', 'team', 'name', 'pos', 'rank', 'opp']:
            continue

        # For percentage columns, take mean
        if 'pct' in col or '%' in col:
            agg_dict[col] = 'mean'
        # For rate/ratio columns, take mean weighted by dropbacks
        elif 'fp_db' in col or 'fp/db' in col or 'db' in col:
            agg_dict[col] = 'mean'
        # For other numeric columns, sum or mean
        elif col in ['g', 'db', 'db_g']:
            agg_dict[col] = 'sum' if col == 'db' else 'mean'
        else:
            agg_dict[col] = 'mean'

    # Group by team and aggregate
    team_qb = qb_df.groupby('team', as_index=False).agg(agg_dict)

    # Prefix columns with 'qb_'
    rename_map = {col: f'qb_{col}' for col in team_qb.columns if col not in ['season', 'team']}
    team_qb = team_qb.rename(columns=rename_map)

    return team_qb


def _aggregate_receiver_metrics(rec_df: pd.DataFrame, metric_prefix: str) -> pd.DataFrame:
    """
    Aggregate receiver metrics to team level (top 2-3 receivers weighted).

    Args:
        rec_df: Receiver DataFrame (routes run, separation, etc.)
        metric_prefix: Prefix for output columns (e.g., 'wr_sep')

    Returns:
        Team-level receiver room metrics
    """
    if 'team' not in rec_df.columns:
        return pd.DataFrame()

    # Filter to WR/TE only
    if 'pos' in rec_df.columns:
        rec_df = rec_df[rec_df['pos'].isin(['WR', 'TE'])].copy()

    # Determine weighting column (prefer routes, then targets, then receptions)
    weight_col = None
    for col in ['rte', 'routes', 'tgt', 'targets', 'rec', 'receptions']:
        if col in rec_df.columns:
            weight_col = col
            break

    if weight_col is None:
        # No weighting available, just take mean
        team_rec = rec_df.groupby('team', as_index=False).mean(numeric_only=True)
    else:
        # Weight by routes/targets and take top performers
        rec_df = rec_df.sort_values(['team', weight_col], ascending=[True, False])

        # Take top 3 receivers per team
        top_receivers = rec_df.groupby('team').head(3)

        # Weighted average by routes/targets
        def weighted_agg(group):
            weights = group[weight_col]
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            result = {}
            for col in numeric_cols:
                if col == weight_col or col in ['season', 'rank', 'g']:
                    continue
                if weights.sum() > 0:
                    result[col] = (group[col] * weights).sum() / weights.sum()
                else:
                    result[col] = group[col].mean()
            return pd.Series(result)

        team_rec = top_receivers.groupby('team').apply(weighted_agg).reset_index()

    # Prefix columns
    rename_map = {col: f'{metric_prefix}_{col}' for col in team_rec.columns if col not in ['season', 'team']}
    team_rec = team_rec.rename(columns=rename_map)

    return team_rec


def _aggregate_rushing_metrics(rush_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rushing metrics to team level.

    Args:
        rush_df: Rushing advanced DataFrame

    Returns:
        Team-level rushing metrics
    """
    if 'team' not in rush_df.columns:
        return pd.DataFrame()

    # Filter to RB only
    if 'pos' in rush_df.columns:
        rush_df = rush_df[rush_df['pos'] == 'RB'].copy()

    # Aggregate by team (weighted by attempts)
    if 'att' in rush_df.columns:
        def weighted_rush_agg(group):
            weights = group['att']
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            result = {}
            for col in numeric_cols:
                if col in ['season', 'rank', 'att']:
                    if col == 'att':
                        result[col] = weights.sum()
                    continue
                if weights.sum() > 0:
                    result[col] = (group[col] * weights).sum() / weights.sum()
                else:
                    result[col] = group[col].mean()
            return pd.Series(result)

        team_rush = rush_df.groupby('team').apply(weighted_rush_agg).reset_index()
    else:
        team_rush = rush_df.groupby('team', as_index=False).mean(numeric_only=True)

    # Prefix columns
    rename_map = {col: f'rush_{col}' for col in team_rush.columns if col not in ['season', 'team']}
    team_rush = team_rush.rename(columns=rename_map)

    return team_rush


def build_fpd_team_week(season: int = 2025) -> pd.DataFrame:
    """
    Build team-week level features from FantasyPoints data.

    This function aggregates all FantasyPoints tables down to one row per team,
    creating ~60-80 hybrid features including:
    - Defensive coverage percentages
    - WR/TE room quality (separation, win rates)
    - QB environment metrics
    - Offensive identity (run/pass tendencies)
    - Run concept effectiveness
    - OL/DL functional metrics

    Args:
        season: NFL season year

    Returns:
        DataFrame with columns:
            - season, team
            - ~60-80 feature columns (prefixed by category)

    Example:
        >>> team_week = build_fpd_team_week(season=2025)
        >>> print(team_week.shape)
        (32, 82)
    """
    print(f"Loading FantasyPoints data for {season} season...")

    # Load all tables
    tables = loaders.load_all_tables(season)

    # Start with coverage matrix (defensive coverage percentages)
    team_features = tables['coverage_matrix'].copy()

    # Prefix defensive coverage columns with 'def_'
    def_cols = [col for col in team_features.columns if col not in ['season', 'team', 'g']]
    rename_def = {col: f'def_{col}' for col in def_cols}
    team_features = team_features.rename(columns=rename_def)

    print(f"✓ Loaded defensive coverage matrix: {len(team_features)} teams")

    # Add QB metrics
    qb_metrics = _aggregate_qb_metrics(tables['qb_coverage_matchup'])
    if not qb_metrics.empty:
        team_features = team_features.merge(qb_metrics, on='team', how='left')
        print(f"✓ Aggregated QB coverage metrics")

    # Add WR/TE receiving metrics
    # Routes run
    if not tables['receiving_routes_run'].empty:
        wr_routes = _aggregate_receiver_metrics(tables['receiving_routes_run'], 'wr_routes')
        if not wr_routes.empty:
            team_features = team_features.merge(wr_routes, on='team', how='left')
            print(f"✓ Aggregated WR route metrics")

    # Separation by coverage
    if not tables['receiving_separation_by_coverage'].empty:
        wr_sep_cov = _aggregate_receiver_metrics(tables['receiving_separation_by_coverage'], 'wr_sep_cov')
        if not wr_sep_cov.empty:
            team_features = team_features.merge(wr_sep_cov, on='team', how='left')
            print(f"✓ Aggregated WR separation by coverage")

    # Advanced receiving
    if not tables['receiving_advanced'].empty:
        wr_adv = _aggregate_receiver_metrics(tables['receiving_advanced'], 'wr_adv')
        if not wr_adv.empty:
            team_features = team_features.merge(wr_adv, on='team', how='left')
            print(f"✓ Aggregated advanced receiving metrics")

    # Add rushing metrics
    rush_metrics = _aggregate_rushing_metrics(tables['rushing_advanced'])
    if not rush_metrics.empty:
        team_features = team_features.merge(rush_metrics, on='team', how='left')
        print(f"✓ Aggregated rushing metrics")

    # Add run-pass report (already team-level)
    run_pass = tables['run_pass_report'].copy()
    if not run_pass.empty and 'team' in run_pass.columns:
        # Select key situational columns
        run_pass_cols = ['team', 'season']
        for col in run_pass.columns:
            if col in ['team', 'season', 'rank', 'name', 'g', 'location']:
                continue
            if 'pass' in col or 'rush' in col or 'snaps' in col:
                run_pass_cols.append(col)

        run_pass = run_pass[[c for c in run_pass_cols if c in run_pass.columns]]

        # Prefix with 'off_'
        rename_rp = {col: f'off_{col}' for col in run_pass.columns if col not in ['season', 'team']}
        run_pass = run_pass.rename(columns=rename_rp)

        team_features = team_features.merge(run_pass, on='team', how='left')
        print(f"✓ Added run-pass tendency report")

    # Add line matchups (already team-level)
    line_matchups = tables['line_matchups'].copy()
    if not line_matchups.empty and 'team' in line_matchups.columns:
        # Prefix offensive line stats with 'ol_' and defensive with 'dl_'
        ol_cols = ['team', 'season']
        dl_cols = ['team', 'season']

        for col in line_matchups.columns:
            if col in ['team', 'season', 'rank', 'name', 'g', 'location']:
                continue
            # Offensive line columns (first set)
            if any(x in col for x in ['rush_grade', 'pass_grade', 'adj_ybc', 'press', 'roe']):
                if 'tm' not in col and 'att' not in col and 'name' not in col and 'ybco' not in col:
                    ol_cols.append(col)

        # For now, just take offensive stats (defensive line stats would need team mapping)
        ol_data = line_matchups[[c for c in ol_cols if c in line_matchups.columns]]
        rename_ol = {col: f'ol_{col}' for col in ol_data.columns if col not in ['season', 'team']}
        ol_data = ol_data.rename(columns=rename_ol)

        team_features = team_features.merge(ol_data, on='team', how='left')
        print(f"✓ Added offensive line metrics")

    print(f"\n✓ Built team-week features: {team_features.shape[0]} teams × {team_features.shape[1]} features")

    return team_features


def build_fpd_matchup_week(
    team_week_df: pd.DataFrame,
    games_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build matchup-week level features with team differentials.

    Args:
        team_week_df: Team-week features from build_fpd_team_week()
        games_df: Games DataFrame with columns:
            - game_id, season, week, home_team, away_team

    Returns:
        DataFrame with columns:
            - game_id, season, week, home_team, away_team
            - home_* features (all team_week features)
            - away_* features (all team_week features)
            - diff_* features (key differentials)

    Example:
        >>> games = load_games_dataset()
        >>> team_week = build_fpd_team_week(season=2025)
        >>> matchups = build_fpd_matchup_week(team_week, games)
    """
    print(f"Building matchup-week features from {len(games_df)} games...")

    # Merge home team features
    home_features = team_week_df.copy()
    home_features.columns = [f'home_{col}' if col not in ['season', 'team'] else col
                              for col in home_features.columns]
    home_features = home_features.rename(columns={'team': 'home_team'})

    matchups = games_df.merge(
        home_features,
        on=['season', 'home_team'],
        how='left'
    )

    # Merge away team features
    away_features = team_week_df.copy()
    away_features.columns = [f'away_{col}' if col not in ['season', 'team'] else col
                              for col in away_features.columns]
    away_features = away_features.rename(columns={'team': 'away_team'})

    matchups = matchups.merge(
        away_features,
        on=['season', 'away_team'],
        how='left'
    )

    # Compute key differentials (home - away)
    diff_features = []

    # Trench differentials
    for metric in ['ol_rush_grade', 'ol_pass_grade', 'ol_adj_ybc_att', 'ol_press_pct', 'ol_proe']:
        home_col = f'home_{metric}'
        away_col = f'away_{metric}'
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[f'diff_{metric}'] = matchups[home_col] - matchups[away_col]
            diff_features.append(f'diff_{metric}')

    # Coverage differentials (home offense vs away defense)
    for metric in ['def_man_pct', 'def_zone_pct', 'def_cover_0_pct', 'def_cover_1_pct',
                   'def_cover_2_pct', 'def_cover_3_pct', 'def_cover_4_pct', 'def_cover_6_pct']:
        home_col = f'away_{metric}'  # Away defense
        if home_col in matchups.columns:
            matchups[f'home_faces_{metric}'] = matchups[home_col]

        away_col = f'home_{metric}'  # Home defense
        if away_col in matchups.columns:
            matchups[f'away_faces_{metric}'] = matchups[away_col]

    # Run concept differentials
    for metric in ['rush_ypc', 'rush_success_pct', 'rush_stuff_pct', 'rush_mtf_att']:
        home_col = f'home_{metric}'
        away_col = f'away_{metric}'
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[f'diff_{metric}'] = matchups[home_col] - matchups[away_col]
            diff_features.append(f'diff_{metric}')

    # WR room quality differentials
    for metric in ['wr_adv_yprr', 'wr_routes_fp_rr', 'wr_sep_cov_sep']:
        home_col = f'home_{metric}'
        away_col = f'away_{metric}'
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[f'diff_{metric}'] = matchups[home_col] - matchups[away_col]
            diff_features.append(f'diff_{metric}')

    # Offensive identity differentials
    for metric in ['off_pass_pct', 'off_rush_pct']:
        home_col = f'home_{metric}'
        away_col = f'away_{metric}'
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[f'diff_{metric}'] = matchups[home_col] - matchups[away_col]
            diff_features.append(f'diff_{metric}')

    print(f"✓ Created {len(diff_features)} differential features")
    print(f"✓ Built matchup features: {matchups.shape[0]} games × {matchups.shape[1]} features")

    return matchups


def save_features(
    team_week_df: pd.DataFrame,
    matchup_week_df: Optional[pd.DataFrame] = None,
    season: int = 2025
) -> None:
    """
    Save team-week and matchup-week features to parquet files.

    Args:
        team_week_df: Team-week features
        matchup_week_df: Optional matchup-week features
        season: NFL season year
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save team-week
    team_week_path = OUTPUT_DIR / f"fpd_team_week_{season}.parquet"
    team_week_df.to_parquet(team_week_path, index=False)
    print(f"\n✓ Saved team-week features to: {team_week_path}")

    # Save matchup-week
    if matchup_week_df is not None:
        matchup_week_path = OUTPUT_DIR / f"fpd_matchup_week_{season}.parquet"
        matchup_week_df.to_parquet(matchup_week_path, index=False)
        print(f"✓ Saved matchup-week features to: {matchup_week_path}")
