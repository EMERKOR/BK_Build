"""
Ball Knower Modeling Module

Contains spread prediction models:
- DeterministicSpreadModel (v1.0)
- EnhancedSpreadModel (v1.1)
- MLCorrectionModel / SpreadCorrectionModel (v1.2)
"""

from .models import (
    DeterministicSpreadModel,
    EnhancedSpreadModel,
    MLCorrectionModel,
    load_calibrated_weights,
    evaluate_model,
    backtest_model,
    calculate_roi_by_edge,
)

__all__ = [
    'DeterministicSpreadModel',
    'EnhancedSpreadModel',
    'MLCorrectionModel',
    'load_calibrated_weights',
    'evaluate_model',
    'backtest_model',
    'calculate_roi_by_edge',
]
