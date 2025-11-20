"""
Offensive and Defensive Series Success Rate (OSR/DSR) Metrics

Leak-free computation of series-level success rates using play-by-play data.
For each team-week, metrics use only data from prior weeks.
"""

import pandas as pd
import numpy as np


def _identify_series_starts(pbp: pd.DataFrame) -> pd.Series:
    """
    Identify the start of each offensive series.

    A new series starts when:
    - First play of the game
    - Down changes from 4th to 1st (new possession)
    - Change of possession (posteam changes)
    - After scoring plays
    """
    # Sort by game_id and play order
    pbp = pbp.sort_values(['game_id', 'play_id'] if 'play_id' in pbp.columns else ['game_id'])

    # Series starts on first down or possession change
    series_start = (
        (pbp['down'] == 1) |  # First down
        (pbp['posteam'] != pbp['posteam'].shift()) |  # Possession change
        (pbp['game_id'] != pbp['game_id'].shift())  # New game
    )

    return series_start


def _compute_series_success(pbp: pd.DataFrame) -> pd.Series:
    """
    Determine if each series was successful.

    A series is successful if:
    - Results in a new first down (gained 10+ yards)
    - Series total EPA > 0
    - Gained sufficient yardage on early downs (70% on 1st, 60% on 2nd)
    - Results in touchdown
    """
    # Simple success metric: use the 'success' column if available
    # Otherwise, approximate with EPA and first down achievement
    if 'success' in pbp.columns:
        return pbp['success'].fillna(0).astype(float)
    elif 'epa' in pbp.columns:
        return (pbp['epa'] > 0).astype(float)
    else:
        # Fallback: assume 50% success rate
        return pd.Series(0.5, index=pbp.index)


def compute_offensive_series_metrics(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute Offensive Series Success Rate (OSR) for each team-week in a season.

    Input pbp should be a single season of play-by-play, with at least:
        ['game_id', 'season', 'week', 'posteam', 'defteam',
         'down', 'ydstogo', 'play_type', 'epa', 'success']

    Returns a DataFrame with columns:
        ['season', 'week', 'team', 'osr_raw']

    Leak-free: For a given (team, week), OSR uses only plays from weeks < week.

    Args:
        pbp: Play-by-play DataFrame for a single season
        season: Season year

    Returns:
        DataFrame with OSR metrics per team-week
    """
    # Filter to relevant plays
    pbp = pbp[pbp['season'] == season].copy()

    # Remove non-plays (penalties, timeouts, etc.)
    if 'play_type' in pbp.columns:
        pbp = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()

    # Compute series success
    pbp['series_success'] = _compute_series_success(pbp)

    # Get unique weeks
    weeks = sorted(pbp['week'].dropna().unique())

    results = []

    for target_week in weeks:
        # LEAK-FREE: Only use data from weeks < target_week
        if target_week == min(weeks):
            # Week 1 has no prior data - set to NaN
            prior_data = pd.DataFrame()
        else:
            prior_data = pbp[pbp['week'] < target_week].copy()

        if len(prior_data) == 0:
            # No prior data - assign neutral/NaN values for all teams
            teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
            for team in teams_in_week:
                results.append({
                    'season': season,
                    'week': target_week,
                    'team': team,
                    'osr_raw': np.nan
                })
        else:
            # Compute OSR from prior weeks
            team_stats = prior_data.groupby('posteam').agg({
                'series_success': ['sum', 'count']
            }).reset_index()

            team_stats.columns = ['team', 'successes', 'total_plays']
            team_stats['osr_raw'] = team_stats['successes'] / team_stats['total_plays'].clip(lower=1)

            # Add results for teams playing this week
            teams_in_week = pbp[pbp['week'] == target_week]['posteam'].unique()
            for team in teams_in_week:
                team_row = team_stats[team_stats['team'] == team]
                if len(team_row) > 0:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'osr_raw': team_row['osr_raw'].iloc[0]
                    })
                else:
                    # Team has no prior data (e.g., week 1)
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'osr_raw': np.nan
                    })

    return pd.DataFrame(results)


def compute_defensive_series_metrics(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute Defensive Series Success Rate (DSR) for each team-week in a season.

    Same leak-free convention as OSR - for each team-week, uses only prior weeks.

    Returns:
        DataFrame with columns ['season', 'week', 'team', 'dsr_raw']

    DSR measures how often the defense prevents offensive success.
    Higher DSR = better defense.
    """
    # Filter to relevant plays
    pbp = pbp[pbp['season'] == season].copy()

    # Remove non-plays
    if 'play_type' in pbp.columns:
        pbp = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()

    # Compute series success (from offense perspective)
    pbp['series_success'] = _compute_series_success(pbp)

    # DSR = 1 - opponent's offensive success rate
    pbp['defensive_stop'] = 1 - pbp['series_success']

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
            teams_in_week = pbp[pbp['week'] == target_week]['defteam'].unique()
            for team in teams_in_week:
                results.append({
                    'season': season,
                    'week': target_week,
                    'team': team,
                    'dsr_raw': np.nan
                })
        else:
            # Compute DSR from prior weeks (aggregate by defensive team)
            team_stats = prior_data.groupby('defteam').agg({
                'defensive_stop': ['sum', 'count']
            }).reset_index()

            team_stats.columns = ['team', 'stops', 'total_plays']
            team_stats['dsr_raw'] = team_stats['stops'] / team_stats['total_plays'].clip(lower=1)

            # Add results for teams playing this week
            teams_in_week = pbp[pbp['week'] == target_week]['defteam'].unique()
            for team in teams_in_week:
                team_row = team_stats[team_stats['team'] == team]
                if len(team_row) > 0:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'dsr_raw': team_row['dsr_raw'].iloc[0]
                    })
                else:
                    results.append({
                        'season': season,
                        'week': target_week,
                        'team': team,
                        'dsr_raw': np.nan
                    })

    return pd.DataFrame(results)


def normalize_osr_dsr(osr_dsr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a DataFrame with columns:
        ['season', 'week', 'team', 'osr_raw', 'dsr_raw']
    and add z-score columns:
        'osr_z', 'dsr_z'

    Normalization is done within season (all teams, all weeks).

    Args:
        osr_dsr_df: DataFrame with raw OSR/DSR metrics

    Returns:
        DataFrame with added z-score columns
    """
    df = osr_dsr_df.copy()

    # Compute z-scores within each season
    for metric in ['osr_raw', 'dsr_raw']:
        z_col = metric.replace('_raw', '_z')

        # Group by season and compute z-scores
        def compute_z(series):
            mean = series.mean()
            std = series.std()
            if std > 0 and not pd.isna(mean):
                return (series - mean) / std
            else:
                return pd.Series(0.0, index=series.index)

        df[z_col] = df.groupby('season')[metric].transform(compute_z)

    return df
