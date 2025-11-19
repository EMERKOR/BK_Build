"""
Standalone validation utilities for detecting data leakage.

This module provides tools for validating that feature engineering pipelines
do not introduce data leakage through:
- Target leakage (features depending on targets at the same timestamp)
- Future information leakage (features using future data)
- Rolling window inconsistencies (incorrect window calculations)

Dependencies: pandas, numpy only.
No imports from other ball_knower modules.
"""

from .leakage import (
    check_no_target_leakage,
    check_no_future_info,
    check_rolling_features,
)

__all__ = [
    "check_no_target_leakage",
    "check_no_future_info",
    "check_rolling_features",
]
