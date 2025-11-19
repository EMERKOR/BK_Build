"""
Ball Knower Dataset Builders

Provides canonical dataset builders for training Ball Knower models.

Available dataset versions:
- v1_0: Baseline actual margin prediction (ELO-based)
- v1_2: Enhanced Vegas spread prediction (situational + QB features)

Usage:
    from ball_knower.datasets import v1_0, v1_2

    df_v1_0 = v1_0.build_training_frame()
    df_v1_2 = v1_2.build_training_frame()
"""

from . import v1_0, v1_2

__all__ = ['v1_0', 'v1_2']
