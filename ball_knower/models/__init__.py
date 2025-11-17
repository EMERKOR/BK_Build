"""
Ball Knower Models Module

ML-based spread prediction models.

Available models:
- v1_2_correction: Residual correction layer on top of deterministic model
"""

from . import v1_2_correction

__all__ = ["v1_2_correction"]
