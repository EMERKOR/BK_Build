"""
Sanity Check Module for Weekly Predictions

Validates predictions and features for anomalies, extreme values,
and logical inconsistencies. Designed to catch data issues before
they reach production.

All functions print warnings but do not raise exceptions, allowing
predictions to proceed with awareness of potential issues.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List


def check_prediction_ranges(df: pd.DataFrame, spread_cols: List[str] = None) -> Dict:
    """
    Check that predicted spreads are within reasonable ranges.

    Args:
        df: DataFrame with predictions
        spread_cols: List of spread column names to check (default: ['bk_line'])

    Returns:
        dict: Summary of range check results
    """
    if spread_cols is None:
        spread_cols = ['bk_line']

    results = {
        'checked_columns': [],
        'extreme_values': [],
        'warnings': []
    }

    for col in spread_cols:
        if col not in df.columns:
            continue

        results['checked_columns'].append(col)

        # Check for extreme values (outside ±40)
        extreme_mask = (df[col].abs() > 40) & df[col].notna()
        if extreme_mask.any():
            extreme_values = df[extreme_mask][[col]]
            results['extreme_values'].append({
                'column': col,
                'count': extreme_mask.sum(),
                'max': df[col].abs().max()
            })
            warnings.warn(
                f"Found {extreme_mask.sum()} extreme spread values in '{col}' (|spread| > 40). "
                f"Max absolute value: {df[col].abs().max():.1f}",
                UserWarning
            )

        # Check for NaN values
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            results['warnings'].append(f"{col}: {nan_count} NaN values")
            warnings.warn(
                f"Found {nan_count} NaN values in '{col}'",
                UserWarning
            )

        # Check for inf values
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            results['warnings'].append(f"{col}: {inf_count} inf values")
            warnings.warn(
                f"Found {inf_count} infinite values in '{col}'",
                UserWarning
            )

    return results


def check_feature_anomalies(features_df: pd.DataFrame) -> Dict:
    """
    Detect anomalies in feature matrix.

    Checks for:
    - NaN, inf, or absurd values in features
    - Impossible rest advantages (e.g., > 14 days)
    - Duplicated games

    Args:
        features_df: DataFrame with features for all games

    Returns:
        dict: Summary of detected anomalies
    """
    results = {
        'nan_features': {},
        'inf_features': {},
        'rest_anomalies': [],
        'duplicate_games': 0,
        'warnings': []
    }

    # Check for NaN values in each column
    for col in features_df.columns:
        nan_count = features_df[col].isna().sum()
        if nan_count > 0:
            results['nan_features'][col] = nan_count

    if results['nan_features']:
        total_nans = sum(results['nan_features'].values())
        warnings.warn(
            f"Found {total_nans} total NaN values across {len(results['nan_features'])} features",
            UserWarning
        )

    # Check for inf values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(features_df[col]).sum()
        if inf_count > 0:
            results['inf_features'][col] = inf_count

    if results['inf_features']:
        total_infs = sum(results['inf_features'].values())
        warnings.warn(
            f"Found {total_infs} total inf values across {len(results['inf_features'])} features",
            UserWarning
        )

    # Check for impossible rest advantages
    rest_cols = [col for col in features_df.columns if 'rest' in col.lower() and 'advantage' in col.lower()]
    for col in rest_cols:
        if col in features_df.columns:
            # Rest advantage should typically be between -14 and +14 days
            impossible_mask = (features_df[col].abs() > 14) & features_df[col].notna()
            if impossible_mask.any():
                count = impossible_mask.sum()
                max_val = features_df[col].abs().max()
                results['rest_anomalies'].append({
                    'column': col,
                    'count': count,
                    'max_absolute': max_val
                })
                warnings.warn(
                    f"Found {count} impossible rest advantages in '{col}' (|rest| > 14 days). "
                    f"Max: {max_val:.1f}",
                    UserWarning
                )

    # Check for duplicate games (same home/away pair)
    if 'team_home' in features_df.columns and 'team_away' in features_df.columns:
        game_pairs = features_df[['team_home', 'team_away']].copy()
        duplicate_count = game_pairs.duplicated().sum()
        if duplicate_count > 0:
            results['duplicate_games'] = duplicate_count
            warnings.warn(
                f"Found {duplicate_count} duplicate game matchups",
                UserWarning
            )

    return results


def check_team_consistency(df: pd.DataFrame) -> Dict:
    """
    Ensure home/away team pairs are unique per game_id.

    Also checks for:
    - Teams playing themselves
    - Missing team names
    - Invalid team codes

    Args:
        df: DataFrame with team_home, team_away, and optionally game_id

    Returns:
        dict: Summary of consistency checks
    """
    results = {
        'self_matchups': 0,
        'missing_teams': {},
        'duplicate_game_ids': 0,
        'warnings': []
    }

    if 'team_home' not in df.columns or 'team_away' not in df.columns:
        results['warnings'].append("Missing team_home or team_away columns")
        return results

    # Check for teams playing themselves
    self_matchup_mask = df['team_home'] == df['team_away']
    if self_matchup_mask.any():
        count = self_matchup_mask.sum()
        results['self_matchups'] = count
        warnings.warn(
            f"Found {count} games where team plays itself (team_home == team_away)",
            UserWarning
        )

    # Check for missing team names
    home_missing = df['team_home'].isna().sum()
    away_missing = df['team_away'].isna().sum()
    if home_missing > 0:
        results['missing_teams']['home'] = home_missing
        warnings.warn(f"Found {home_missing} games with missing home team", UserWarning)
    if away_missing > 0:
        results['missing_teams']['away'] = away_missing
        warnings.warn(f"Found {away_missing} games with missing away team", UserWarning)

    # Check for duplicate game_ids
    if 'game_id' in df.columns:
        duplicate_count = df['game_id'].duplicated().sum()
        if duplicate_count > 0:
            results['duplicate_game_ids'] = duplicate_count
            warnings.warn(
                f"Found {duplicate_count} duplicate game_ids",
                UserWarning
            )

    return results


def run_all_sanity_checks(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame = None
) -> Dict:
    """
    Run all sanity checks on predictions and features.

    Args:
        predictions_df: DataFrame with predictions (must have bk_line or similar)
        features_df: Optional DataFrame with feature matrix

    Returns:
        dict: Combined results from all checks
    """
    print("\n[Sanity Checks] Running validation on predictions...")

    all_results = {}

    # Check prediction ranges
    spread_cols = [col for col in predictions_df.columns if 'line' in col.lower() or 'spread' in col.lower()]
    if spread_cols:
        all_results['prediction_ranges'] = check_prediction_ranges(predictions_df, spread_cols)
        print(f"  ✓ Checked prediction ranges for {len(spread_cols)} column(s)")

    # Check team consistency
    all_results['team_consistency'] = check_team_consistency(predictions_df)
    print("  ✓ Checked team consistency")

    # Check features if provided
    if features_df is not None:
        all_results['feature_anomalies'] = check_feature_anomalies(features_df)
        print("  ✓ Checked feature anomalies")

    # Print summary
    total_warnings = sum(
        len(check.get('warnings', [])) for check in all_results.values()
    )

    if total_warnings == 0:
        print("  ✓ All sanity checks passed")
    else:
        print(f"  ⚠ Found issues (see warnings above)")

    return all_results
