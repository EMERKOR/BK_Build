"""
Ball Knower v1.3 Dataset Builder

Builds training dataset with team form features on top of v1.2.

New Features (v1.3):
- offense_form_epa_diff: Home vs away rolling offensive EPA
- offense_form_success_diff: Home vs away rolling offensive success rate
- defense_form_epa_diff: Home vs away rolling defensive EPA
- defense_form_success_diff: Home vs away rolling defensive success rate

All form features are leak-free (computed from prior games only).

Use Case:
- Capture recent team performance trends
- Identify momentum and form advantages
- Enhance v1.2 predictions with rolling efficiency metrics
"""

import pandas as pd
import numpy as np
import warnings

from ball_knower.datasets import v1_2
from ball_knower.io import loaders
from ball_knower.features import form


def build_training_frame(
    start_year: int = 2013,
    end_year: int = 2024,
    data_url: str = None
) -> pd.DataFrame:
    """
    Build v1.3 training dataset (v1.2 + team form features).

    Args:
        start_year: Start season year (default: 2013, when team-week EPA data starts)
        end_year: End season year (default: 2024)
        data_url: Optional custom nfelo data URL (passed to v1.2 builder)

    Returns:
        DataFrame with all v1.2 columns plus:
            - offense_form_epa_diff: Home - away rolling offensive EPA
            - offense_form_success_diff: Home - away rolling offensive success rate
            - defense_form_epa_diff: Home - away rolling defensive EPA
            - defense_form_success_diff: Home - away rolling defensive success rate

    Expected shape:
        - Rows: 1500-3500 games (depending on year range, fewer than v1.2 due to 2013 start)
        - Columns: v1.2 columns (18) + 4 new form features = 22

    Note:
        - Form features require team-week EPA data, which starts in 2013
        - Games with missing form data will have NaN for form features
        - Start year must be >= 2013 for form features to be available
    """
    # Suppress v1.3 warning from form module import
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        # Build v1.2 base dataset
        df_v1_2 = v1_2.build_training_frame(
            start_year=start_year,
            end_year=end_year,
            data_url=data_url
        )

        # Load team-week EPA data
        df_team_week = loaders.load_team_week_epa(
            start_season=start_year,
            end_season=end_year
        )

        # Compute team form features
        df_form = form.compute_team_form(df_team_week, window=4)

        # Prepare form data for joining
        # Home team form
        df_home_form = df_form.rename(columns={
            'team': 'home_team',
            'offense_form_epa': 'home_offense_form_epa',
            'offense_form_success': 'home_offense_form_success',
            'defense_form_epa': 'home_defense_form_epa',
            'defense_form_success': 'home_defense_form_success'
        })

        # Away team form
        df_away_form = df_form.rename(columns={
            'team': 'away_team',
            'offense_form_epa': 'away_offense_form_epa',
            'offense_form_success': 'away_offense_form_success',
            'defense_form_epa': 'away_defense_form_epa',
            'defense_form_success': 'away_defense_form_success'
        })

        # Join home team form
        df_v1_3 = df_v1_2.merge(
            df_home_form[['home_team', 'season', 'week',
                         'home_offense_form_epa', 'home_offense_form_success',
                         'home_defense_form_epa', 'home_defense_form_success']],
            on=['home_team', 'season', 'week'],
            how='left'
        )

        # Join away team form
        df_v1_3 = df_v1_3.merge(
            df_away_form[['away_team', 'season', 'week',
                         'away_offense_form_epa', 'away_offense_form_success',
                         'away_defense_form_epa', 'away_defense_form_success']],
            on=['away_team', 'season', 'week'],
            how='left'
        )

        # Compute form differentials (home - away)
        df_v1_3['offense_form_epa_diff'] = (
            df_v1_3['home_offense_form_epa'] - df_v1_3['away_offense_form_epa']
        )
        df_v1_3['offense_form_success_diff'] = (
            df_v1_3['home_offense_form_success'] - df_v1_3['away_offense_form_success']
        )
        df_v1_3['defense_form_epa_diff'] = (
            df_v1_3['home_defense_form_epa'] - df_v1_3['away_defense_form_epa']
        )
        df_v1_3['defense_form_success_diff'] = (
            df_v1_3['home_defense_form_success'] - df_v1_3['away_defense_form_success']
        )

        # Drop intermediate home/away form columns (keep only differentials)
        cols_to_drop = [
            'home_offense_form_epa', 'home_offense_form_success',
            'home_defense_form_epa', 'home_defense_form_success',
            'away_offense_form_epa', 'away_offense_form_success',
            'away_defense_form_epa', 'away_defense_form_success'
        ]
        df_v1_3 = df_v1_3.drop(columns=cols_to_drop)

        # Log missing form data
        form_cols = [
            'offense_form_epa_diff', 'offense_form_success_diff',
            'defense_form_epa_diff', 'defense_form_success_diff'
        ]
        n_missing = df_v1_3[form_cols].isna().any(axis=1).sum()
        if n_missing > 0:
            warnings.warn(
                f"{n_missing} games have missing form features (likely early-season games). "
                f"These games can still be used for training, but form features will be NaN.",
                UserWarning
            )

        return df_v1_3.reset_index(drop=True)
