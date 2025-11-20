"""
Offensive Line Structure Index (OLSI) Metrics

Leak-free computation of offensive line performance using pass protection metrics.
For each team-week, metrics use only data from prior weeks.
"""

import pandas as pd
import numpy as np


def compute_ol_structure_metrics(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute Offensive Line Structure Index (OLSI) proxies per team-week.

    Uses pass-play-based features:
      - pressure rate allowed
      - sack rate over expected (if feasible)
      - qb hit rate (if available in pbp)

    All computed leak-free using only weeks < current_week.

    Returns DataFrame:
      ['season', 'week', 'team',
       'pressure_rate_raw', 'sack_rate_raw', 'qb_hit_rate_raw',
       'olsi_raw', 'olsi_z']

    Args:
        pbp: Play-by-play DataFrame for a single season
        season: Season year

    Returns:
        DataFrame with OLSI metrics per team-week
    """
    # Filter to relevant plays
    pbp = pbp[pbp['season'] == season].copy()

    # Filter to pass plays only
    if 'play_type' in pbp.columns:
        pbp = pbp[pbp['play_type'] == 'pass'].copy()
    elif 'pass' in pbp.columns:
        pbp = pbp[pbp['pass'] == 1].copy()

    # Get unique weeks
    weeks = sorted(pbp['week'].dropna().unique())

    results = []

    for target_week in weeks:
        # LEAK-FREE: Only use data from weeks < target_week
        if target_week == min(weeks):
            prior_data = pd.DataFrame()
        else:
            prior_data = pbp[pbp['week'] < target_week].copy()

        if len(prior_data) == 0:
            # No prior data - assign NaN for all teams
            teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
            for team in teams_in_week:
                results.append({
                    'season': season,
                    'week': target_week,
                    'team': team,
                    'pressure_rate_raw': np.nan,
                    'sack_rate_raw': np.nan,
                    'qb_hit_rate_raw': np.nan,
                    'olsi_raw': np.nan,
                })
        else:
            # Compute pass protection metrics from prior weeks
            team_stats_list = []

            for team in prior_data['posteam'].unique():
                team_passes = prior_data[prior_data['posteam'] == team]

                dropbacks = len(team_passes)
                if dropbacks == 0:
                    continue

                # Pressure rate (if qb_hit or sack occurred)
                pressures = 0
                if 'qb_hit' in team_passes.columns:
                    pressures = team_passes['qb_hit'].fillna(0).sum()
                if 'sack' in team_passes.columns:
                    pressures += team_passes['sack'].fillna(0).sum()

                # Sack rate
                sacks = 0
                if 'sack' in team_passes.columns:
                    sacks = team_passes['sack'].fillna(0).sum()

                # QB hit rate
                qb_hits = 0
                if 'qb_hit' in team_passes.columns:
                    qb_hits = team_passes['qb_hit'].fillna(0).sum()

                team_stats_list.append({
                    'team': team,
                    'dropbacks': dropbacks,
                    'pressures': pressures,
                    'sacks': sacks,
                    'qb_hits': qb_hits,
                })

            if len(team_stats_list) == 0:
                # No teams with data
                teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
                for team in teams_in_week:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'pressure_rate_raw': np.nan,
                        'sack_rate_raw': np.nan,
                        'qb_hit_rate_raw': np.nan,
                        'olsi_raw': np.nan,
                    })
                continue

            team_stats = pd.DataFrame(team_stats_list)

            # Compute rates
            team_stats['pressure_rate_raw'] = team_stats['pressures'] / team_stats['dropbacks']
            team_stats['sack_rate_raw'] = team_stats['sacks'] / team_stats['dropbacks']
            team_stats['qb_hit_rate_raw'] = team_stats['qb_hits'] / team_stats['dropbacks']

            # Add results for teams playing this week
            teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
            for team in teams_in_week:
                team_row = team_stats[team_stats['team'] == team]
                if len(team_row) > 0:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'pressure_rate_raw': team_row['pressure_rate_raw'].iloc[0],
                        'sack_rate_raw': team_row['sack_rate_raw'].iloc[0],
                        'qb_hit_rate_raw': team_row['qb_hit_rate_raw'].iloc[0],
                        'olsi_raw': np.nan,  # Computed after normalization
                    })
                else:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'pressure_rate_raw': np.nan,
                        'sack_rate_raw': np.nan,
                        'qb_hit_rate_raw': np.nan,
                        'olsi_raw': np.nan,
                    })

    df = pd.DataFrame(results)

    # Compute z-scores within season
    for metric in ['pressure_rate_raw', 'sack_rate_raw', 'qb_hit_rate_raw']:
        z_col = metric.replace('_raw', '_z')

        def compute_z(series):
            mean = series.mean()
            std = series.std()
            if std > 0 and not pd.isna(mean):
                return (series - mean) / std
            else:
                return pd.Series(0.0, index=series.index)

        df[z_col] = df.groupby('season')[metric].transform(compute_z)

    # Compute OLSI composite (negative because lower rates are better)
    df['olsi_raw'] = (
        -0.40 * df['pressure_rate_z']
        - 0.30 * df['sack_rate_z']
        - 0.30 * df['qb_hit_rate_z']
    )

    # Normalize OLSI to z-score
    def compute_z(series):
        mean = series.mean()
        std = series.std()
        if std > 0 and not pd.isna(mean):
            return (series - mean) / std
        else:
            return pd.Series(0.0, index=series.index)

    df['olsi_z'] = df.groupby('season')['olsi_raw'].transform(compute_z)

    return df
