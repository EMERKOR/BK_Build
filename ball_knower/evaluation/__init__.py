"""
Ball Knower Evaluation Utilities

Provides unified evaluation metrics and reporting functions for model backtesting.

Available metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- ATS (Against The Spread) record
- Edge and EV calculations
- Summary backtest reports
"""

from . import metrics

__all__ = ['metrics']
