"""
Ball Knower Benchmarking Module

This module provides utilities for comparing and evaluating different
versions of the Ball Knower prediction models.
"""

from ball_knower.benchmarks.v1_comparison import (
    compare_v1_models,
    run_v1_0_backtest_on_frame,
    run_v1_2_backtest_on_frame,
    run_v1_3_backtest_on_frame,
)

__all__ = [
    'compare_v1_models',
    'run_v1_0_backtest_on_frame',
    'run_v1_2_backtest_on_frame',
    'run_v1_3_backtest_on_frame',
]
