"""
Ball Knower - NFL Betting Analytics

A leak-free, modular NFL spread prediction system.
"""

__version__ = '1.0.0'

# Import from ball_knower canonical modules (source of truth)
from ball_knower import config  # type: ignore
from ball_knower.features import engineering as features  # type: ignore
from ball_knower.modeling import models  # type: ignore

# Import from src (still housed here)
from . import team_mapping
from . import data_loader

__all__ = [
    'config',
    'team_mapping',
    'data_loader',
    'features',
    'models'
]
