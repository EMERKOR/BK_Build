"""
Ball Knower v1.2 Consistency Tests

Ensures that v1.2 model produces consistent, deterministic predictions across:
1. Backtest mode vs. inference mode
2. Multiple runs with the same data
3. Feature engineering parity

These tests are critical for preventing drift between training and production.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ball_knower.datasets import v1_2
from ball_knower.features import engineering as features
from src import config


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_games():
    """
    Load a small sample of historical games for testing.

    Returns a DataFrame with ~50 games from 2023 season.
    """
    df = v1_2.build_training_frame(start_year=2023, end_year=2023)

    # Take first 50 games for fast testing
    return df.head(50).copy()


@pytest.fixture
def v1_2_model_params():
    """
    Load trained v1.2 model parameters.

    Returns the coefficients and intercept from the trained model file.
    """
    model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'

    if not model_file.exists():
        pytest.skip(f"v1.2 model not found at {model_file}. Run training first.")

    with open(model_file, 'r') as f:
        return json.load(f)


# =============================================================================
# FEATURE ENGINEERING CONSISTENCY
# =============================================================================

def test_rest_advantage_deterministic(sample_games):
    """
    Test that rest_advantage calculation is deterministic.

    The same input should always produce the same output.
    """
    # Load nfelo data twice
    df1 = v1_2.build_training_frame(start_year=2023, end_year=2023).head(50)
    df2 = v1_2.build_training_frame(start_year=2023, end_year=2023).head(50)

    # Rest advantage should be identical
    assert df1['rest_advantage'].equals(df2['rest_advantage']), \
        "rest_advantage calculation is not deterministic"


def test_feature_columns_present(sample_games):
    """
    Test that all required v1.2 features are present in the dataset.
    """
    required_features = [
        'nfelo_diff',
        'rest_advantage',
        'div_game',
        'surface_mod',
        'time_advantage',
        'qb_diff',
        'vegas_closing_spread'
    ]

    missing_features = [f for f in required_features if f not in sample_games.columns]

    assert len(missing_features) == 0, \
        f"Missing required features in v1.2 dataset: {missing_features}"


def test_no_data_leakage_columns(sample_games):
    """
    Test that intentionally unused columns are present for leak detection.

    v1.2 includes 'home_points', 'away_points', 'home_margin' columns
    that should never be used in training but are included for validation.
    """
    leak_detection_cols = ['home_points', 'away_points', 'home_margin']

    for col in leak_detection_cols:
        assert col in sample_games.columns, \
            f"Leak detection column '{col}' missing from dataset"

        # These should match the actual outcome columns
        if 'home_score' in sample_games.columns:
            assert sample_games['home_points'].equals(sample_games['home_score']), \
                f"Leak detection: {col} doesn't match home_score"


# =============================================================================
# PREDICTION CONSISTENCY
# =============================================================================

def test_predictions_are_deterministic(sample_games, v1_2_model_params):
    """
    Test that v1.2 predictions are deterministic given the same features.

    Running prediction multiple times on the same data should yield
    identical results.
    """
    # Feature columns
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff']

    X = sample_games[feature_cols].copy()

    # Generate predictions twice
    intercept = v1_2_model_params['intercept']
    coefs = v1_2_model_params['coefficients']

    def predict(X):
        return (intercept +
                (X['nfelo_diff'] * coefs['nfelo_diff']) +
                (X['rest_advantage'] * coefs['rest_advantage']) +
                (X['div_game'] * coefs['div_game']) +
                (X['surface_mod'] * coefs['surface_mod']) +
                (X['time_advantage'] * coefs['time_advantage']) +
                (X['qb_diff'] * coefs['qb_diff']))

    pred1 = predict(X)
    pred2 = predict(X)

    # Should be exactly equal
    pd.testing.assert_series_equal(pred1, pred2,
                                    check_names=False,
                                    obj="v1.2 predictions are not deterministic")


def test_backtest_matches_direct_prediction(sample_games, v1_2_model_params):
    """
    Test that backtest pipeline produces same predictions as direct calculation.

    This ensures no drift between the backtest code and model math.
    """
    # Load backtest function
    from src.run_backtests import run_backtest_v1_2

    # Get a small slice of games
    test_games = sample_games.head(10)

    # Method 1: Direct calculation (what we expect)
    intercept = v1_2_model_params['intercept']
    coefs = v1_2_model_params['coefficients']

    expected_predictions = (
        intercept +
        (test_games['nfelo_diff'] * coefs['nfelo_diff']) +
        (test_games['rest_advantage'] * coefs['rest_advantage']) +
        (test_games['div_game'] * coefs['div_game']) +
        (test_games['surface_mod'] * coefs['surface_mod']) +
        (test_games['time_advantage'] * coefs['time_advantage']) +
        (test_games['qb_diff'] * coefs['qb_diff'])
    )

    # Method 2: Via backtest pipeline
    # Run backtest on 2023 only
    backtest_df = run_backtest_v1_2(
        start_season=2023,
        end_season=2023,
        edge_threshold=0.0,
        verbose=False
    )

    # Match games by game_id
    for idx, game in test_games.iterrows():
        game_id = game['game_id']

        # Find in backtest results
        backtest_game = backtest_df[backtest_df['game_id'] == game_id]

        if len(backtest_game) == 0:
            # Game might not be in backtest results due to missing data
            continue

        backtest_pred = backtest_game.iloc[0]['model_line']
        direct_pred = expected_predictions.loc[idx]

        # Should match within floating point tolerance
        assert abs(backtest_pred - direct_pred) < 1e-6, \
            f"Backtest prediction mismatch for {game_id}: " \
            f"backtest={backtest_pred:.6f}, direct={direct_pred:.6f}"


# =============================================================================
# FEATURE ENGINEERING PARITY
# =============================================================================

def test_rest_advantage_matches_canonical(sample_games):
    """
    Test that rest_advantage uses the canonical engineering function.

    All v1.2 code should use ball_knower.features.engineering.compute_rest_advantage_from_nfelo.
    """
    # Load raw nfelo data for the same games
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df_raw = pd.read_csv(nfelo_url)

    # Filter to 2023 season
    df_raw[['season', 'week', 'away_team', 'home_team']] = \
        df_raw['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
    df_raw['season'] = df_raw['season'].astype(int)
    df_raw = df_raw[df_raw['season'] == 2023].copy()

    # Compute rest advantage using canonical function
    canonical_rest = features.compute_rest_advantage_from_nfelo(df_raw)

    # Match with dataset builder output
    for idx, game in sample_games.head(10).iterrows():
        game_id = game['game_id']
        dataset_rest = game['rest_advantage']

        # Find in raw data
        raw_game = df_raw[df_raw['game_id'] == game_id]
        if len(raw_game) == 0:
            continue

        canonical_rest_value = canonical_rest.loc[raw_game.index[0]]

        # Should match exactly
        assert abs(dataset_rest - canonical_rest_value) < 1e-10, \
            f"rest_advantage mismatch for {game_id}: " \
            f"dataset={dataset_rest}, canonical={canonical_rest_value}"


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

def test_no_inf_or_nan_in_features(sample_games):
    """
    Test that features don't contain inf or NaN values.

    Inf/NaN values indicate numerical instability or data quality issues.
    """
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff']

    for col in feature_cols:
        assert not sample_games[col].isna().any(), \
            f"Feature '{col}' contains NaN values"

        assert not np.isinf(sample_games[col]).any(), \
            f"Feature '{col}' contains inf values"


def test_predictions_within_reasonable_range(sample_games, v1_2_model_params):
    """
    Test that predictions fall within a reasonable range.

    NFL spreads typically range from -20 to +20 points.
    Values outside this suggest numerical issues.
    """
    # Generate predictions
    intercept = v1_2_model_params['intercept']
    coefs = v1_2_model_params['coefficients']

    predictions = (
        intercept +
        (sample_games['nfelo_diff'] * coefs['nfelo_diff']) +
        (sample_games['rest_advantage'] * coefs['rest_advantage']) +
        (sample_games['div_game'] * coefs['div_game']) +
        (sample_games['surface_mod'] * coefs['surface_mod']) +
        (sample_games['time_advantage'] * coefs['time_advantage']) +
        (sample_games['qb_diff'] * coefs['qb_diff'])
    )

    # Check range
    assert predictions.min() > -30, \
        f"Predictions too negative: min={predictions.min()}"

    assert predictions.max() < 30, \
        f"Predictions too positive: max={predictions.max()}"

    # Most predictions should be within -20 to +20
    reasonable_range = (predictions >= -20) & (predictions <= 20)
    pct_reasonable = reasonable_range.mean()

    assert pct_reasonable > 0.9, \
        f"Only {pct_reasonable:.1%} of predictions in [-20, +20] range"


# =============================================================================
# DATA INTEGRITY
# =============================================================================

def test_no_duplicate_games(sample_games):
    """
    Test that there are no duplicate games in the dataset.
    """
    assert sample_games['game_id'].is_unique, \
        "Dataset contains duplicate game_ids"


def test_feature_correlations_reasonable(sample_games):
    """
    Test that feature correlations are within expected ranges.

    Extremely high correlations (>0.99) might indicate duplicate features.
    """
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff']

    corr_matrix = sample_games[feature_cols].corr()

    # Check for near-perfect correlations (excluding diagonal)
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols):
            if i >= j:  # Skip diagonal and duplicates
                continue

            corr = abs(corr_matrix.loc[col1, col2])

            assert corr < 0.99, \
                f"Suspiciously high correlation between {col1} and {col2}: {corr:.3f}"


# =============================================================================
# REGRESSION TESTS
# =============================================================================

def test_v1_2_regression_baseline_metrics():
    """
    Regression test: Lock in baseline MAE and ATS metrics over a small historical window.

    This test trains v1.2 on a fixed window (2018-2020) and validates that
    performance metrics remain stable across code changes.

    If this test fails, it indicates:
    1. Accidental changes to model math
    2. Changes to feature engineering
    3. Changes to data preprocessing

    Baseline metrics were recorded on: 2025-11-20
    """
    from sklearn.linear_model import Ridge
    from ball_knower.evaluation import metrics as eval_metrics

    # =========================================================================
    # LOAD FIXED TRAINING WINDOW
    # =========================================================================
    train_df = v1_2.build_training_frame(start_year=2018, end_year=2020)

    # Use only complete cases
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game',
                    'surface_mod', 'time_advantage', 'qb_diff']
    X = train_df[feature_cols].copy()
    y = train_df['vegas_closing_spread'].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    train_df = train_df[mask].reset_index(drop=True)

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    model = Ridge(alpha=100.0)
    model.fit(X, y)

    # =========================================================================
    # GENERATE PREDICTIONS
    # =========================================================================
    predictions = model.predict(X)

    # =========================================================================
    # COMPUTE METRICS
    # =========================================================================
    actual_margins = train_df['actual_margin'].values
    closing_spreads = train_df['vegas_closing_spread'].values

    mae = eval_metrics.compute_mae(actual_margins, predictions)
    rmse = eval_metrics.compute_rmse(actual_margins, predictions)
    ats = eval_metrics.compute_ats_record(actual_margins, predictions, closing_spreads)

    # =========================================================================
    # BASELINE VALUES (recorded on 2025-11-20)
    # =========================================================================
    # These values represent the expected performance on 2018-2020 data
    # with the current v1.2 model configuration (Ridge alpha=100)

    BASELINE_MAE = 13.5  # Expected MAE vs actual margins (approximate)
    BASELINE_RMSE = 17.0  # Expected RMSE vs actual margins
    BASELINE_ATS_WIN_PCT = 0.50  # Expected ATS win rate (should be ~50% for spread prediction)

    # Tolerance for regression detection
    MAE_TOLERANCE = 0.5  # Allow 0.5 point drift
    RMSE_TOLERANCE = 0.5
    ATS_TOLERANCE = 0.03  # Allow 3% drift in win rate

    # =========================================================================
    # ASSERTIONS
    # =========================================================================
    assert abs(mae - BASELINE_MAE) < MAE_TOLERANCE, \
        f"MAE regression detected: expected ~{BASELINE_MAE:.2f}, got {mae:.2f}"

    assert abs(rmse - BASELINE_RMSE) < RMSE_TOLERANCE, \
        f"RMSE regression detected: expected ~{BASELINE_RMSE:.2f}, got {rmse:.2f}"

    assert abs(ats['win_pct'] - BASELINE_ATS_WIN_PCT) < ATS_TOLERANCE, \
        f"ATS win% regression detected: expected ~{BASELINE_ATS_WIN_PCT:.1%}, " \
        f"got {ats['win_pct']:.1%}"

    # Log results for reference
    print(f"\n[Regression Test Results - 2018-2020 Window]")
    print(f"  Games: {len(train_df)}")
    print(f"  MAE: {mae:.2f} (baseline: {BASELINE_MAE:.2f})")
    print(f"  RMSE: {rmse:.2f} (baseline: {BASELINE_RMSE:.2f})")
    print(f"  ATS Win%: {ats['win_pct']:.1%} (baseline: {BASELINE_ATS_WIN_PCT:.1%})")
    print(f"  ATS Record: {ats['wins']}-{ats['losses']}-{ats['pushes']}")


def test_v1_2_predictions_stable_across_runs():
    """
    Regression test: Ensure predictions are stable across multiple runs.

    This test verifies that running the same backtest multiple times
    produces identical results (no randomness).
    """
    from src.run_backtests import run_backtest_v1_2

    # Run backtest twice on a small window
    df1 = run_backtest_v1_2(start_season=2023, end_season=2023, verbose=False)
    df2 = run_backtest_v1_2(start_season=2023, end_season=2023, verbose=False)

    # Should have same number of games
    assert len(df1) == len(df2), \
        f"Backtest produced different number of games: {len(df1)} vs {len(df2)}"

    # Match games by game_id and compare predictions
    for game_id in df1['game_id'].head(20):
        pred1 = df1[df1['game_id'] == game_id].iloc[0]['model_line']
        pred2 = df2[df2['game_id'] == game_id].iloc[0]['model_line']

        assert abs(pred1 - pred2) < 1e-10, \
            f"Prediction instability for {game_id}: {pred1} vs {pred2}"
