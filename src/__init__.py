"""
Ball Knower - NFL Betting Analytics

A leak-free, modular NFL spread prediction system.
"""

__version__ = '1.0.0'

from . import config
from . import team_mapping
from . import data_loader  # LEGACY: Use ball_knower.io.loaders for new code
from . import features
from . import models

__all__ = [
    'config',
    'team_mapping',
    'data_loader',  # LEGACY: Use ball_knower.io.loaders for new code
    'features',
    'models'
]
