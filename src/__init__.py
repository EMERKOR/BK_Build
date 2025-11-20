"""
Ball Knower - NFL Betting Analytics

A leak-free, modular NFL spread prediction system.

IMPORTANT: This module is legacy. Most functionality has been migrated to the
ball_knower package. Only team_mapping and utility scripts remain here.
"""

__version__ = '1.0.0'

from . import team_mapping

__all__ = [
    'team_mapping',
]
