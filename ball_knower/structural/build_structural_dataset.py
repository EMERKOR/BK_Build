"""
Structural Metrics Dataset Builder

Orchestrates OSR, DSR, OLSI, and CEA metrics into a unified structural dataset.
All metrics are computed leak-free using only prior week data.
"""

import pandas as pd
import numpy as np

from ball_knower.structural.osr_dsr import (
    compute_offensive_series_metrics,
    compute_defensive_series_metrics,
    normalize_osr_dsr,
)
from ball_knower.structural.olsi import compute_ol_structure_metrics
from ball_knower.structural.cea import compute_coaching_edge_metrics


def build_structural_metrics_for_season(
    pbp: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Orchestrate OSR, DSR, OLSI, CEA for a single season.

    Input pbp: full-season play-by-play for that season.

    Output columns:
      ['season', 'week', 'team',
       'osr_raw', 'dsr_raw',
       'osr_z', 'dsr_z',
       'olsi_raw', 'olsi_z',
       'go_rate_over_expected_raw',
       'wpa_lost_raw',
       'cea_raw', 'cea_z',
       'structural_edge']

    Where:
      structural_edge =
          0.35 * osr_z
        + 0.35 * dsr_z
        + 0.20 * olsi_z
        + 0.10 * cea_z

    Args:
        pbp: Play-by-play DataFrame for a single season
        season: Season year

    Returns:
        DataFrame with all structural metrics per team-week
    """
    print(f"Building structural metrics for {season}...")

    # Compute OSR
    print("  Computing OSR...")
    osr_df = compute_offensive_series_metrics(pbp, season)

    # Compute DSR
    print("  Computing DSR...")
    dsr_df = compute_defensive_series_metrics(pbp, season)

    # Merge OSR and DSR
    osr_dsr = osr_df.merge(
        dsr_df,
        on=['season', 'week', 'team'],
        how='outer'
    )

    # Normalize OSR/DSR
    osr_dsr = normalize_osr_dsr(osr_dsr)

    # Compute OLSI
    print("  Computing OLSI...")
    olsi_df = compute_ol_structure_metrics(pbp, season)

    # Compute CEA
    print("  Computing CEA...")
    cea_df = compute_coaching_edge_metrics(pbp, season)

    # Merge all metrics
    print("  Merging all metrics...")
    result = osr_dsr.merge(
        olsi_df,
        on=['season', 'week', 'team'],
        how='outer'
    )

    result = result.merge(
        cea_df,
        on=['season', 'week', 'team'],
        how='outer'
    )

    # Fill NaN z-scores with 0 (neutral)
    z_cols = ['osr_z', 'dsr_z', 'olsi_z', 'cea_z']
    for col in z_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    # Compute composite structural_edge
    result['structural_edge'] = (
        0.35 * result['osr_z'] +
        0.35 * result['dsr_z'] +
        0.20 * result['olsi_z'] +
        0.10 * result['cea_z']
    )

    # Sort by week and team
    result = result.sort_values(['week', 'team']).reset_index(drop=True)

    print(f"  ✓ Completed {season}: {len(result)} team-week rows")

    return result


def build_structural_metrics_all_seasons(
    pbp_all: pd.DataFrame,
    seasons: list,
) -> pd.DataFrame:
    """
    Loop over seasons, call build_structural_metrics_for_season,
    and concatenate results.

    Args:
        pbp_all: Play-by-play DataFrame for multiple seasons
        seasons: List of season years to process

    Returns:
        Single DataFrame with all seasons' structural metrics
    """
    print("="*70)
    print("Building Structural Metrics for Multiple Seasons")
    print("="*70)
    print()

    all_results = []

    for season in seasons:
        pbp_season = pbp_all[pbp_all['season'] == season].copy()

        if len(pbp_season) == 0:
            print(f"WARNING: No data found for season {season}, skipping...")
            continue

        try:
            season_metrics = build_structural_metrics_for_season(pbp_season, season)
            all_results.append(season_metrics)
        except Exception as e:
            print(f"ERROR: Failed to process season {season}: {e}")
            continue

    if len(all_results) == 0:
        print("ERROR: No seasons were successfully processed")
        return pd.DataFrame()

    # Concatenate all seasons
    print()
    print("="*70)
    print("Concatenating all seasons...")
    final_df = pd.concat(all_results, ignore_index=True)

    print(f"✓ Total rows: {len(final_df)}")
    print(f"✓ Seasons: {sorted(final_df['season'].unique())}")
    print(f"✓ Teams: {final_df['team'].nunique()}")
    print()

    return final_df
