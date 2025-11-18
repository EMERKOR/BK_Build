"""
Ball Knower v1.2 Model

Residual correction model using Ridge regression.
Predicts: residual = vegas_closing_spread - bk_base_spread
"""

from .model import V1_2ResidualModel
from .features import get_feature_matrix, get_target_vector, FEATURE_COLUMNS
from .train import train_v1_2

__all__ = [
    'V1_2ResidualModel',
    'get_feature_matrix',
    'get_target_vector',
    'FEATURE_COLUMNS',
    'train_v1_2'
]
