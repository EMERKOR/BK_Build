"""
Ball Knower Structural Metrics

Historical, leak-free structural factors derived from play-by-play data:
- Offensive Series Success Rate (OSR)
- Defensive Series Success Rate (DSR)
- Offensive Line Structure Index (OLSI)
- Coaching Edge/Aggression (CEA)

All metrics are computed using only prior weeks to ensure leak-free modeling.
"""

from ball_knower.structural.osr_dsr import (
    compute_offensive_series_metrics,
    compute_defensive_series_metrics,
    normalize_osr_dsr,
)
from ball_knower.structural.olsi import compute_ol_structure_metrics
from ball_knower.structural.cea import compute_coaching_edge_metrics
from ball_knower.structural.build_structural_dataset import (
    build_structural_metrics_for_season,
    build_structural_metrics_all_seasons,
)

__all__ = [
    "compute_offensive_series_metrics",
    "compute_defensive_series_metrics",
    "normalize_osr_dsr",
    "compute_ol_structure_metrics",
    "compute_coaching_edge_metrics",
    "build_structural_metrics_for_season",
    "build_structural_metrics_all_seasons",
]
