"""
Ball Knower I/O Module

Unified data loading interfaces for NFL analytics data.
Supports both category-first (new) and provider-first (legacy) filename patterns.
"""

from .loaders import (
    load_power_ratings,
    load_epa_tiers,
    load_strength_of_schedule,
    load_qb_rankings,
    load_qb_epa,
    load_weekly_projections_ppg,
    load_weekly_projections_elo,
    load_win_totals,
    load_receiving_leaders,
    load_all_sources,
    merge_team_ratings,
    VALID_CATEGORIES,
    VALID_PROVIDERS,
)

__all__ = [
    "load_power_ratings",
    "load_epa_tiers",
    "load_strength_of_schedule",
    "load_qb_rankings",
    "load_qb_epa",
    "load_weekly_projections_ppg",
    "load_weekly_projections_elo",
    "load_win_totals",
    "load_receiving_leaders",
    "load_all_sources",
    "merge_team_ratings",
    "VALID_CATEGORIES",
    "VALID_PROVIDERS",
]
