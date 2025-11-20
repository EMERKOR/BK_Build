"""
Ball Knower v1.2+BK Dataset Builder

Extends v1.2 dataset with Ball Knower blended team power ratings.

Features:
- All v1.2 features (nfelo_diff, rest_advantage, div_game, etc.)
- BK blended team ratings (bk_rating_home, bk_rating_away, bk_rating_diff)

Target:
- vegas_closing_spread (market consensus)

Use Case:
- Evaluate incremental value of BK ratings over baseline v1.2 model
- Compare BK ratings to market-implied team strength
"""

import pandas as pd
from typing import Optional

from ball_knower.datasets import v1_2
from ball_knower.features.team_power import attach_team_power_ratings


def build_training_frame(
    season: int,
    week: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Build v1.2+BK training dataset with blended team power ratings.

    Args:
        season: NFL season year (for single-week mode: 2025)
                For historical mode: use as start_year
        week: NFL week number (required for BK ratings sandbox)
              If None, falls back to historical v1.2 behavior
        **kwargs: Additional arguments passed to v1_2.build_training_frame()

    Returns:
        DataFrame with all v1.2 columns plus:
            - bk_rating_home: BK blended rating for home team
            - bk_rating_away: BK blended rating for away team
            - bk_rating_diff: Differential (home - away)

    Raises:
        ValueError: If week is None (BK ratings require single-week mode)
    """
    # For now, BK ratings only work in single-week mode
    if week is None:
        raise ValueError(
            "v1.2+BK dataset requires week parameter for BK ratings attachment.\n"
            "Use v1_2.build_training_frame() for historical backtesting."
        )

    # Build base v1.2 dataset
    # For single-week sandbox, filter to specific season/week
    base = v1_2.build_training_frame(
        start_year=season,
        end_year=season,
        **kwargs
    )

    # Filter to specific week
    base = base[base['week'] == week].copy()

    if len(base) == 0:
        raise ValueError(
            f"No games found for season={season}, week={week} in v1.2 dataset"
        )

    # Attach BK team power ratings
    # This adds: bk_rating_home, bk_rating_away, bk_rating_diff
    enriched = attach_team_power_ratings(
        games=base,
        season=season,
        week=week,
        rating_column="bk_blended_rating",
        how="left",
    )

    # Sanity checks
    # 1. Should have same number of rows (no games dropped)
    if len(enriched) != len(base):
        raise RuntimeError(
            f"BK ratings merge changed row count: {len(base)} -> {len(enriched)}"
        )

    # 2. Should have BK rating columns
    required_bk_cols = ['bk_rating_home', 'bk_rating_away', 'bk_rating_diff']
    missing = [col for col in required_bk_cols if col not in enriched.columns]
    if missing:
        raise RuntimeError(f"Missing BK rating columns after merge: {missing}")

    # 3. No NaN values in BK ratings (all teams should have ratings)
    for col in required_bk_cols:
        if enriched[col].isna().any():
            raise RuntimeError(
                f"Found NaN values in {col} - not all teams have BK ratings"
            )

    return enriched
