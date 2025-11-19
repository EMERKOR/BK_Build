"""
Tests for BK v1.3 Score Prediction Model

This test module validates:
- Training pipeline functionality
- Model prediction functionality
- Backtest evaluation
- Data integrity and no leakage
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ball_knower.modeling.v1_3.training_template import (
    build_training_frame,
    split_train_val_test,
    train_v1_3,
    save_model_artifacts,
    load_model_artifacts
)
from ball_knower.modeling.v1_3.score_model_template import ScorePredictionModelV13
from ball_knower.modeling.v1_3.backtest_template import (
    backtest_v1_3,
    compute_score_metrics,
    compute_spread_total_metrics
)


class TestTrainingPipeline:
    """Test the training pipeline components."""

    def test_build_training_frame(self):
        """Test that build_training_frame loads data correctly."""
        # Load a small subset for testing
        df = build_training_frame(seasons=[2022])

        # Validate basic structure
        assert len(df) > 0, "Training frame should not be empty"
        assert 'home_score' in df.columns, "Missing home_score column"
        assert 'away_score' in df.columns, "Missing away_score column"
        assert 'season' in df.columns, "Missing season column"

        # Validate no leak columns
        leak_cols = ['home_points', 'away_points', 'home_margin']
        for col in leak_cols:
            assert col not in df.columns, f"Leak column {col} should be removed"

        # Validate all scores are non-null
        assert df['home_score'].notna().all(), "home_score has null values"
        assert df['away_score'].notna().all(), "away_score has null values"

    def test_split_train_val_test(self):
        """Test temporal data splitting."""
        df = build_training_frame(seasons=[2018, 2019, 2020, 2021, 2022])

        train_df, val_df, test_df = split_train_val_test(
            df,
            val_seasons=[2020, 2021],
            test_seasons=[2022]
        )

        # Check splits are non-empty
        assert len(train_df) > 0, "Train set should not be empty"
        assert len(val_df) > 0, "Val set should not be empty"
        assert len(test_df) > 0, "Test set should not be empty"

        # Check no overlap
        train_seasons = set(train_df['season'].unique())
        val_seasons = set(val_df['season'].unique())
        test_seasons = set(test_df['season'].unique())

        assert not (train_seasons & val_seasons), "Train and val should not overlap"
        assert not (train_seasons & test_seasons), "Train and test should not overlap"
        assert not (val_seasons & test_seasons), "Val and test should not overlap"

        # Check temporal ordering
        assert max(train_df['season']) < min(val_df['season']), "Train must be before val"

    def test_train_v1_3_basic(self):
        """Test that train_v1_3 runs without errors."""
        # Use small dataset for speed
        train_df = build_training_frame(seasons=[2018, 2019])
        val_df = build_training_frame(seasons=[2020])

        results = train_v1_3(
            train_df=train_df,
            val_df=val_df,
            model_type='ridge',
            hyperparams={'alpha': 1.0}
        )

        # Validate result structure
        assert 'home_model' in results, "Missing home_model"
        assert 'away_model' in results, "Missing away_model"
        assert 'feature_names' in results, "Missing feature_names"
        assert 'train_metrics' in results, "Missing train_metrics"
        assert 'val_metrics' in results, "Missing val_metrics"

        # Validate metrics are computed
        assert results['train_metrics']['mae_home_score'] > 0
        assert results['val_metrics']['mae_home_score'] > 0

    def test_save_load_model_artifacts(self):
        """Test saving and loading model artifacts."""
        # Train a simple model
        train_df = build_training_frame(seasons=[2018])
        results = train_v1_3(
            train_df=train_df,
            model_type='linear'
        )

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            save_model_artifacts(results, str(save_path))

            # Verify files exist
            assert (save_path / 'home_model.pkl').exists()
            assert (save_path / 'away_model.pkl').exists()
            assert (save_path / 'features.json').exists()
            assert (save_path / 'metadata.json').exists()

            # Load and verify
            loaded = load_model_artifacts(str(save_path))
            assert 'home_model' in loaded
            assert 'away_model' in loaded
            assert 'feature_names' in loaded
            assert len(loaded['feature_names']) > 0


class TestScorePredictionModel:
    """Test the ScorePredictionModelV13 class."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = ScorePredictionModelV13()
        assert model is not None
        assert not model.is_fitted

    def test_model_predict(self):
        """Test model prediction pipeline."""
        # Train a model
        train_df = build_training_frame(seasons=[2018, 2019])
        results = train_v1_3(train_df=train_df, model_type='ridge')

        # Create model instance
        model = ScorePredictionModelV13(
            home_model=results['home_model'],
            away_model=results['away_model'],
            feature_names=results['feature_names']
        )

        assert model.is_fitted

        # Make predictions
        test_df = build_training_frame(seasons=[2020])
        predictions = model.predict(test_df)

        # Validate prediction structure
        assert len(predictions) == len(test_df)
        assert 'pred_home_score' in predictions.columns
        assert 'pred_away_score' in predictions.columns
        assert 'pred_spread' in predictions.columns
        assert 'pred_total' in predictions.columns

        # Validate no NaN predictions
        assert predictions['pred_home_score'].notna().all()
        assert predictions['pred_away_score'].notna().all()

        # Validate predictions are reasonable (NFL scores typically 0-60)
        assert (predictions['pred_home_score'] >= 0).all()
        assert (predictions['pred_home_score'] <= 60).all()
        assert (predictions['pred_away_score'] >= 0).all()
        assert (predictions['pred_away_score'] <= 60).all()

    def test_model_save_load(self):
        """Test model save/load functionality."""
        # Train a model
        train_df = build_training_frame(seasons=[2018])
        results = train_v1_3(train_df=train_df, model_type='ridge')

        model = ScorePredictionModelV13(
            home_model=results['home_model'],
            away_model=results['away_model'],
            feature_names=results['feature_names']
        )

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model'
            model.save(str(save_path))

            loaded_model = ScorePredictionModelV13.load(str(save_path))

            # Verify loaded model works
            assert loaded_model.is_fitted
            test_df = build_training_frame(seasons=[2020])
            predictions = loaded_model.predict(test_df)
            assert len(predictions) > 0


class TestBacktesting:
    """Test the backtesting framework."""

    def test_compute_score_metrics(self):
        """Test score metrics computation."""
        # Create mock data
        actual_home = pd.Series([24, 27, 20, 31])
        actual_away = pd.Series([17, 24, 23, 28])
        pred_home = pd.Series([23, 28, 21, 30])
        pred_away = pd.Series([18, 23, 22, 27])

        metrics = compute_score_metrics(actual_home, actual_away, pred_home, pred_away)

        # Validate metric keys
        assert 'mae_home_score' in metrics
        assert 'mae_away_score' in metrics
        assert 'rmse_home_score' in metrics
        assert 'rmse_away_score' in metrics

        # Validate metrics are finite numbers
        assert np.isfinite(metrics['mae_home_score'])
        assert metrics['mae_home_score'] >= 0

    def test_compute_spread_total_metrics(self):
        """Test spread and total metrics computation."""
        actual_home = pd.Series([24, 27, 20, 31])
        actual_away = pd.Series([17, 24, 23, 28])
        pred_home = pd.Series([23, 28, 21, 30])
        pred_away = pd.Series([18, 23, 22, 27])

        metrics = compute_spread_total_metrics(actual_home, actual_away, pred_home, pred_away)

        # Validate metric keys
        assert 'mae_spread' in metrics
        assert 'mae_total' in metrics
        assert 'spread_accuracy_3pt' in metrics
        assert 'total_accuracy_7pt' in metrics

        # Validate accuracy percentages are in [0, 100]
        assert 0 <= metrics['spread_accuracy_3pt'] <= 100
        assert 0 <= metrics['total_accuracy_7pt'] <= 100

    def test_backtest_v1_3_integration(self):
        """Test full backtest pipeline."""
        # Train a model
        train_df = build_training_frame(seasons=[2018, 2019])
        results = train_v1_3(train_df=train_df, model_type='ridge')

        model = ScorePredictionModelV13(
            home_model=results['home_model'],
            away_model=results['away_model'],
            feature_names=results['feature_names']
        )

        # Run backtest
        test_df = build_training_frame(seasons=[2020])
        backtest_results = backtest_v1_3(
            model=model,
            test_df=test_df,
            compute_derived_metrics=True
        )

        # Validate result structure
        assert 'score_metrics' in backtest_results
        assert 'derived_metrics' in backtest_results
        assert 'predictions_df' in backtest_results
        assert 'summary' in backtest_results

        # Validate metrics are computed
        assert backtest_results['score_metrics']['mae_home_score'] > 0
        assert backtest_results['derived_metrics']['mae_spread'] > 0

        # Validate predictions DataFrame
        pred_df = backtest_results['predictions_df']
        assert len(pred_df) == len(test_df)
        assert 'pred_home_score' in pred_df.columns
        assert 'pred_spread' in pred_df.columns


def test_no_data_leakage():
    """Test that training data contains no leakage."""
    df = build_training_frame(seasons=[2022])

    # These columns should NOT exist (they leak actual outcomes)
    forbidden_cols = ['home_points', 'away_points', 'home_margin']
    for col in forbidden_cols:
        assert col not in df.columns, f"Leak column {col} found in training frame"


def test_end_to_end_workflow():
    """Test complete workflow from training to backtest."""
    # 1. Build data
    df = build_training_frame(seasons=[2018, 2019, 2020, 2021, 2022])

    # 2. Split
    train_df, val_df, test_df = split_train_val_test(
        df,
        val_seasons=[2020, 2021],
        test_seasons=[2022]
    )

    # 3. Train
    results = train_v1_3(
        train_df=train_df,
        val_df=val_df,
        model_type='ridge',
        hyperparams={'alpha': 1.0}
    )

    # 4. Create model
    model = ScorePredictionModelV13(
        home_model=results['home_model'],
        away_model=results['away_model'],
        feature_names=results['feature_names']
    )

    # 5. Backtest
    backtest_results = backtest_v1_3(
        model=model,
        test_df=test_df,
        compute_derived_metrics=True
    )

    # 6. Validate reasonable performance
    # (MAE should be less than 20 points for scores)
    assert backtest_results['score_metrics']['mae_home_score'] < 20
    assert backtest_results['score_metrics']['mae_away_score'] < 20

    print("\nEnd-to-end test passed!")
    print(f"Home Score MAE: {backtest_results['score_metrics']['mae_home_score']:.2f}")
    print(f"Spread MAE: {backtest_results['derived_metrics']['mae_spread']:.2f}")
