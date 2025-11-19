"""
Ball Knower Feature Engineering Module

Centralized feature engineering utilities for:
- Rest advantage calculations (NFElo bye modifiers and schedule-based)
- Rolling statistics (EPA, form, etc.)
- Matchup features

CRITICAL: All features must be LEAK-FREE.
Rolling statistics ONLY use past games, never include current game.
"""

from ball_knower.features import engineering

__all__ = ['engineering']
