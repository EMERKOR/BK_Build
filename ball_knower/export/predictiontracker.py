"""
PredictionTracker Export Formatter

Converts Ball Knower backtest results into PredictionTracker-compatible CSV format.

PredictionTracker is a platform for tracking and comparing sports betting models.
This module transforms Ball Knower backtest outputs into the required schema.

Required PredictionTracker columns:
- date: Game date (YYYY-MM-DD)
- home_team: Home team name
- away_team: Away team name
- line: Model's predicted line
- vegas_line: Closing market line (optional)
- model_version: Model identifier

Usage:
    from ball_knower.export import predictiontracker
    import pandas as pd

    # Load backtest results
    df = pd.read_csv('output/backtests/v1.2/backtest_v1.2_2019_2024.csv')

    # Convert to PredictionTracker format
    pt_df = predictiontracker.export_predictiontracker_format(df, 'v1.2')

    # Save
    pt_df.to_csv('output/predictiontracker/v1_2_2019_2024.csv', index=False)
"""

import pandas as pd
from typing import Optional
from datetime import datetime


def export_predictiontracker_format(
    backtest_df: pd.DataFrame,
    model_version: str,
    include_actuals: bool = True
) -> pd.DataFrame:
    """
    Convert a Ball Knower backtest DataFrame into PredictionTracker-compatible format.

    Args:
        backtest_df: DataFrame from run_backtests.py with standardized schema
        model_version: Model version identifier (e.g., 'v1.0', 'v1.2')
        include_actuals: Whether to include actual_margin column (default: True)

    Returns:
        DataFrame with PredictionTracker-compatible schema

    Required input columns:
        - season, week: For date derivation
        - home_team, away_team: Team identifiers
        - model_line or bk_line: Model's predicted line
        - closing_spread: Vegas closing line (optional but recommended)
        - actual_margin: Actual game result (optional)

    Output columns:
        - date: Estimated game date (YYYY-MM-DD)
        - home_team: Home team name
        - away_team: Away team name
        - line: Model's predicted line (home team perspective)
        - vegas_line: Market closing line
        - actual_margin: Actual result (if include_actuals=True)
        - model_version: Model identifier

    Example:
        >>> df = pd.DataFrame({
        ...     'season': [2023, 2023],
        ...     'week': [1, 1],
        ...     'home_team': ['BUF', 'KC'],
        ...     'away_team': ['NYJ', 'DET'],
        ...     'model_line': [-7.5, -10.2],
        ...     'closing_spread': [-7.0, -9.5],
        ...     'actual_margin': [-6, -11]
        ... })
        >>> result = export_predictiontracker_format(df, 'v1.2')
        >>> 'date' in result.columns
        True
    """
    if backtest_df is None or len(backtest_df) == 0:
        # Return empty DataFrame with correct schema
        cols = ['date', 'home_team', 'away_team', 'line', 'vegas_line', 'model_version']
        if include_actuals:
            cols.append('actual_margin')
        return pd.DataFrame(columns=cols)

    df = backtest_df.copy()

    # Validate required columns
    required_cols = ['season', 'week', 'home_team', 'away_team']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # =========================================================================
    # DATE DERIVATION
    # =========================================================================
    # PredictionTracker requires dates, but Ball Knower backtests use season/week
    # We'll estimate dates using NFL calendar conventions
    #
    # NFL season typically starts first Thursday after Labor Day (first Mon in Sept)
    # For simplicity, we'll use: Season start = Sept 10 + (week - 1) * 7 days
    # This is approximate but sufficient for tracking purposes

    def estimate_game_date(season: int, week: int) -> str:
        """
        Estimate game date from season and week.

        NFL regular season weeks 1-18 run roughly Sept-Jan.
        Week 1 typically starts around Sept 10.
        """
        # Base date: approximate first week of season
        # Most years start around Sept 5-12, so we'll use Sept 10
        base_date = datetime(season, 9, 10)

        # Add weeks (each week = 7 days)
        # Week 1 = base_date, Week 2 = base_date + 7, etc.
        estimated_date = base_date + pd.Timedelta(days=(week - 1) * 7)

        return estimated_date.strftime('%Y-%m-%d')

    df['date'] = df.apply(
        lambda row: estimate_game_date(int(row['season']), int(row['week'])),
        axis=1
    )

    # =========================================================================
    # LINE COLUMNS
    # =========================================================================
    # Model line: prefer 'model_line', fallback to 'bk_line'
    if 'model_line' in df.columns:
        df['line'] = df['model_line']
    elif 'bk_line' in df.columns:
        df['line'] = df['bk_line']
    else:
        raise ValueError("No model prediction column found (expected 'model_line' or 'bk_line')")

    # Vegas line: use closing_spread if available
    if 'closing_spread' in df.columns:
        df['vegas_line'] = df['closing_spread']
    else:
        df['vegas_line'] = None

    # =========================================================================
    # MODEL VERSION
    # =========================================================================
    df['model_version'] = model_version

    # =========================================================================
    # SELECT AND ORDER COLUMNS
    # =========================================================================
    output_cols = [
        'date',
        'home_team',
        'away_team',
        'line',
        'vegas_line',
        'model_version'
    ]

    # Optionally include actual results
    if include_actuals and 'actual_margin' in df.columns:
        output_cols.append('actual_margin')

    # Filter to only columns that exist
    output_cols = [col for col in output_cols if col in df.columns]

    return df[output_cols].copy()


def validate_predictiontracker_format(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame conforms to PredictionTracker format.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If DataFrame doesn't conform to expected schema
    """
    required_cols = ['date', 'home_team', 'away_team', 'line', 'model_version']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required PredictionTracker columns: {missing_cols}")

    # Validate date format (should be parseable as dates)
    try:
        pd.to_datetime(df['date'])
    except Exception as e:
        raise ValueError(f"Invalid date format in 'date' column: {e}")

    # Validate line is numeric
    if not pd.api.types.is_numeric_dtype(df['line']):
        raise ValueError("'line' column must be numeric")

    return True
