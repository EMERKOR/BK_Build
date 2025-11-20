"""
Spread prediction models for Ball Knower.

This module contains all spread prediction models:
- DeterministicSpreadModel (v1.0)
- EnhancedSpreadModel (v1.1)
- MLCorrectionModel (v1.2)
"""

from ball_knower.modeling.models import (
    DeterministicSpreadModel,
    EnhancedSpreadModel,
    MLCorrectionModel,
    evaluate_model,
    backtest_model,
    calculate_roi_by_edge,
)

__all__ = [
    "DeterministicSpreadModel",
    "EnhancedSpreadModel",
    "MLCorrectionModel",
    "evaluate_model",
    "backtest_model",
    "calculate_roi_by_edge",
]
