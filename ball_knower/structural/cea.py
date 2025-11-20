"""
Coaching Edge / Fourth-Down Aggression (CEA) Metrics

Leak-free computation of coaching decision quality using 4th down situations.
For each team-week, metrics use only data from prior weeks.
"""

import pandas as pd
import numpy as np


def _identify_go_situations(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Identify 4th down "go situations" where going for it is a reasonable option.

    Criteria:
    - 4th down
    - 5 yards or less to go
    - Between own 40 and opponent 40 yard line
    - Not obvious garbage time (score diff < 14, quarter <= 3 or early 4th)
    """
    # Return empty if required columns missing
    if 'down' not in pbp.columns or 'ydstogo' not in pbp.columns:
        return pd.DataFrame()

    go_sits = pbp[
        (pbp['down'] == 4) &
        (pbp['ydstogo'] <= 5)
    ].copy()

    # Filter by field position if yardline_100 available
    if 'yardline_100' in go_sits.columns:
        go_sits = go_sits[
            (go_sits['yardline_100'] >= 40) &  # Between own 40
            (go_sits['yardline_100'] <= 60)     # and opponent 40
        ]

    # Filter out garbage time if score_differential and quarter available
    if 'score_differential' in go_sits.columns and 'quarter' in go_sits.columns:
        go_sits = go_sits[
            (go_sits['score_differential'].abs() < 14) |
            (go_sits['quarter'] <= 3)
        ]

    return go_sits


def _classify_decision(row: pd.Series) -> str:
    """
    Classify a 4th down decision as GO or NO_GO.

    GO: pass or run play
    NO_GO: punt or field goal attempt
    """
    if 'play_type' in row and pd.notna(row['play_type']):
        if row['play_type'] in ['pass', 'run']:
            return 'GO'
        elif row['play_type'] in ['punt', 'field_goal']:
            return 'NO_GO'

    # Fallback: check individual flags
    if 'pass' in row and row.get('pass', 0) == 1:
        return 'GO'
    if 'rush' in row and row.get('rush', 0) == 1:
        return 'GO'
    if 'punt_attempt' in row and row.get('punt_attempt', 0) == 1:
        return 'NO_GO'
    if 'field_goal_attempt' in row and row.get('field_goal_attempt', 0) == 1:
        return 'NO_GO'

    return 'UNKNOWN'


def compute_coaching_edge_metrics(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute coaching edge / 4th down aggression metrics.

    Approximation:
      - 4th down go-rate over expected (GOOE proxy)
      - Win probability added lost from conservative decisions (WPA_Lost proxy)

    Returns:
      ['season', 'week', 'team',
       'go_rate_over_expected_raw',
       'wpa_lost_raw',
       'cea_raw',
       'cea_z']

    Must be leak-free: week w uses only weeks < w.

    Args:
        pbp: Play-by-play DataFrame for a single season
        season: Season year

    Returns:
        DataFrame with CEA metrics per team-week
    """
    # Filter to relevant season
    pbp = pbp[pbp['season'] == season].copy()

    # Identify go situations
    go_situations = _identify_go_situations(pbp)

    if len(go_situations) == 0:
        # No go situations found - return empty DataFrame
        return pd.DataFrame(columns=[
            'season', 'week', 'team',
            'go_rate_over_expected_raw',
            'wpa_lost_raw',
            'cea_raw',
            'cea_z'
        ])

    # Classify each decision
    go_situations['decision'] = go_situations.apply(_classify_decision, axis=1)

    # Get unique weeks
    weeks = sorted(pbp['week'].dropna().unique())

    results = []

    for target_week in weeks:
        # LEAK-FREE: Only use data from weeks < target_week
        if target_week == min(weeks):
            prior_data = pd.DataFrame()
        else:
            prior_data = go_situations[go_situations['week'] < target_week].copy()

        if len(prior_data) == 0:
            # No prior data - assign NaN for all teams
            teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
            for team in teams_in_week:
                results.append({
                    'season': season,
                    'week': target_week,
                    'team': team,
                    'go_rate_over_expected_raw': np.nan,
                    'wpa_lost_raw': np.nan,
                    'cea_raw': np.nan,
                })
        else:
            # Compute league-wide expected go rate from prior weeks
            total_decisions = prior_data[prior_data['decision'].isin(['GO', 'NO_GO'])]
            if len(total_decisions) > 0:
                expected_go_rate = (total_decisions['decision'] == 'GO').mean()
            else:
                expected_go_rate = 0.5  # Default if no data

            # Compute per-team metrics
            team_stats_list = []

            for team in prior_data['posteam'].unique():
                team_decisions = prior_data[
                    (prior_data['posteam'] == team) &
                    (prior_data['decision'].isin(['GO', 'NO_GO']))
                ]

                if len(team_decisions) == 0:
                    continue

                # Actual go rate
                go_count = (team_decisions['decision'] == 'GO').sum()
                total_count = len(team_decisions)
                actual_go_rate = go_count / total_count if total_count > 0 else 0.0

                # Go rate over expected
                go_rate_over_expected = actual_go_rate - expected_go_rate

                # WPA lost proxy: count "egregious" conservative decisions
                # 4th and 2 or less, close game
                egregious = team_decisions[
                    (team_decisions['ydstogo'] <= 2) &
                    (team_decisions['decision'] == 'NO_GO')
                ]
                if 'score_differential' in egregious.columns:
                    egregious = egregious[egregious['score_differential'].abs() <= 7]

                wpa_lost = -len(egregious)  # Negative because it's a penalty

                team_stats_list.append({
                    'team': team,
                    'go_rate_over_expected_raw': go_rate_over_expected,
                    'wpa_lost_raw': wpa_lost,
                })

            if len(team_stats_list) == 0:
                # No teams with data
                teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
                for team in teams_in_week:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'go_rate_over_expected_raw': np.nan,
                        'wpa_lost_raw': np.nan,
                        'cea_raw': np.nan,
                    })
                continue

            team_stats = pd.DataFrame(team_stats_list)

            # Add results for teams playing this week
            teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
            for team in teams_in_week:
                team_row = team_stats[team_stats['team'] == team]
                if len(team_row) > 0:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'go_rate_over_expected_raw': team_row['go_rate_over_expected_raw'].iloc[0],
                        'wpa_lost_raw': team_row['wpa_lost_raw'].iloc[0],
                        'cea_raw': np.nan,  # Computed after normalization
                    })
                else:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'go_rate_over_expected_raw': np.nan,
                        'wpa_lost_raw': np.nan,
                        'cea_raw': np.nan,
                    })

    df = pd.DataFrame(results)

    # Compute z-scores within season
    for metric in ['go_rate_over_expected_raw', 'wpa_lost_raw']:
        z_col = metric.replace('_raw', '_z')

        def compute_z(series):
            mean = series.mean()
            std = series.std()
            if std > 0 and not pd.isna(mean):
                return (series - mean) / std
            else:
                return pd.Series(0.0, index=series.index)

        df[z_col] = df.groupby('season')[metric].transform(compute_z)

    # Compute CEA composite
    df['cea_raw'] = (
        0.60 * df['go_rate_over_expected_z'] +
        0.40 * df['wpa_lost_z']
    )

    # Normalize CEA to z-score
    def compute_z(series):
        mean = series.mean()
        std = series.std()
        if std > 0 and not pd.isna(mean):
            return (series - mean) / std
        else:
            return pd.Series(0.0, index=series.index)

    df['cea_z'] = df.groupby('season')['cea_raw'].transform(compute_z)

    return df
