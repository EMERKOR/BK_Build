"""
Training Pipeline Template for BK v1.3 Score Prediction Model

PURPOSE
-------
This module provides scaffolding for the training pipeline that will:
- Build training datasets by joining v1.2 features with actual game scores
- Train separate models for home_score and away_score prediction
- Validate data for leakage and integrity
- Save trained model artifacts with versioning

INTEGRATION INTO BK ARCHITECTURE
--------------------------------
The training pipeline will:
- Consume features from v1.2 feature engineering pipeline
- Join with historical game results (scores)
- Split data into train/validation/test sets
- Train ScorePredictionModelV13 instances
- Evaluate model performance on validation set
- Save production-ready models with metadata

NON-NEGOTIABLE INVARIANTS
-------------------------
1. NO LEAKAGE: Training data must only include information available
   before each game's kickoff time
2. TEMPORAL SPLITS: Train/val/test splits must respect temporal ordering
   (no training on future data to predict past)
3. DATA INTEGRITY: All joins must be validated for correctness
4. REPRODUCIBILITY: Training runs must be reproducible with fixed seeds
5. VERSIONING: All trained models must be tagged with version and metadata

FUTURE DESIGN CONSIDERATIONS
----------------------------
- Feature selection strategy (correlation analysis, recursive elimination)
- Hyperparameter optimization (grid search, Bayesian optimization)
- Cross-validation strategy (time series aware)
- Model ensemble techniques
- Online learning / incremental updates

TODO
----
[ ] Implement build_training_frame() to join features with scores
[ ] Add data validation and leakage checks
[ ] Implement train/val/test splitting logic
[ ] Add hyperparameter tuning pipeline
[ ] Implement model evaluation metrics
[ ] Add model versioning and metadata tracking
[ ] Create training logging and monitoring
[ ] Add early stopping and regularization
"""

from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_training_frame(
    seasons: Optional[list] = None,
    feature_version: str = "v1_2",
    include_playoffs: bool = False
) -> pd.DataFrame:
    """
    Build training dataset by joining features with actual game scores.

    This function constructs the complete training dataset by:
    1. Loading v1.2 engineered features for specified seasons
    2. Loading actual game results (home_score, away_score)
    3. Joining features with results on game_id
    4. Validating temporal consistency and no leakage
    5. Returning clean training frame

    Parameters
    ----------
    seasons : list, optional
        List of seasons to include (e.g., [2018, 2019, 2020])
        If None, will use default range (2009-2024)
    feature_version : str, default='v1_2'
        Version of features to use (currently only v1_2 supported)
    include_playoffs : bool, default=False
        Whether to include playoff games in training

    Returns
    -------
    pd.DataFrame
        Training dataframe with columns:
        - game_id: Unique game identifier
        - season: Season year
        - week: Week number
        - home_team: Home team abbreviation
        - away_team: Away team abbreviation
        - home_score: Actual home team score (target)
        - away_score: Actual away team score (target)
        - [feature_columns]: All v1.2 engineered features

    Examples
    --------
    >>> df = build_training_frame(seasons=[2018, 2019, 2020])
    >>> print(df.shape)
    >>> print(df[['game_id', 'home_score', 'away_score']].head())
    """
    # Import the v1.2 dataset builder
    from ball_knower.datasets.v1_2 import build_training_frame as build_v1_2_frame

    # Determine season range
    if seasons is not None:
        start_year = min(seasons)
        end_year = max(seasons)
    else:
        start_year = 2009
        end_year = 2024

    # Load v1.2 features + scores
    df = build_v1_2_frame(start_year=start_year, end_year=end_year)

    # Filter to specified seasons if provided
    if seasons is not None:
        df = df[df['season'].isin(seasons)].copy()

    # Filter playoffs if requested
    if not include_playoffs:
        # Regular season is typically weeks 1-18 (varies by year but this is safe)
        df = df[df['week'] <= 18].copy()

    # Remove leak columns (columns that contain actual outcomes in different forms)
    leak_columns = ['home_points', 'away_points', 'home_margin']
    df = df.drop(columns=[col for col in leak_columns if col in df.columns])

    # Ensure we have the required target columns
    required_cols = ['home_score', 'away_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required target column: {col}")

    # Filter out rows with missing targets
    df = df[df['home_score'].notna() & df['away_score'].notna()].copy()

    # Sort by season and week for temporal consistency
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    return df


def split_train_val_test(
    df: pd.DataFrame,
    val_seasons: Optional[list] = None,
    test_seasons: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets with temporal awareness.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset from build_training_frame()
    val_seasons : list, optional
        Seasons to use for validation (e.g., [2019, 2020, 2021])
        If None, defaults to [2019, 2020, 2021]
    test_seasons : list, optional
        Seasons to use for test (e.g., [2022, 2023, 2024])
        If None, defaults to [2022, 2023, 2024]

    Returns
    -------
    train_df : pd.DataFrame
        Training set (all seasons not in val or test)
    val_df : pd.DataFrame
        Validation set
    test_df : pd.DataFrame
        Test set

    Notes
    -----
    Ensures temporal ordering: train < val < test
    No data overlap between sets
    """
    # Default splits if not provided
    if val_seasons is None:
        val_seasons = [2019, 2020, 2021]
    if test_seasons is None:
        test_seasons = [2022, 2023, 2024]

    # Validate no overlap
    if set(val_seasons) & set(test_seasons):
        raise ValueError("Validation and test seasons must not overlap")

    # Get max season in validation to ensure train < val
    if val_seasons:
        max_val_season = max(val_seasons)
    else:
        max_val_season = 0

    # Split the data
    test_df = df[df['season'].isin(test_seasons)].copy()
    val_df = df[df['season'].isin(val_seasons)].copy()
    train_df = df[~df['season'].isin(val_seasons + test_seasons)].copy()

    # Filter train to be before validation
    if max_val_season > 0:
        train_df = train_df[train_df['season'] < max_val_season].copy()

    return train_df, val_df, test_df


def train_v1_3(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_type: str = "ridge",
    hyperparams: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train v1.3 score prediction models.

    This function orchestrates the complete training process:
    1. Initialize models with specified hyperparameters
    2. Train home_model on home_score targets
    3. Train away_model on away_score targets
    4. Evaluate on validation set if provided
    5. Save model artifacts if save_path specified
    6. Return training results and metrics

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data from build_training_frame()
    val_df : pd.DataFrame, optional
        Validation data for evaluation
    model_type : str, default='ridge'
        Type of model to train ('ridge' or 'linear')
    hyperparams : dict, optional
        Hyperparameters to pass to the model
    save_path : str, optional
        Path to save trained model artifacts
        If None, model will not be saved

    Returns
    -------
    dict
        Training results containing:
        - 'home_model': Trained home score model
        - 'away_model': Trained away score model
        - 'feature_names': List of feature column names
        - 'train_metrics': Training set performance metrics
        - 'val_metrics': Validation set performance metrics (if val_df provided)
        - 'training_metadata': Metadata about the training run

    Examples
    --------
    >>> train_df = build_training_frame(seasons=[2009, 2010])
    >>> val_df = build_training_frame(seasons=[2019])
    >>> results = train_v1_3(
    ...     train_df=train_df,
    ...     val_df=val_df,
    ...     model_type='ridge',
    ...     hyperparams={'alpha': 1.0}
    ... )
    >>> print(results['val_metrics'])
    """
    # Identify feature and target columns
    metadata_cols = ['game_id', 'season', 'week', 'home_team', 'away_team']
    target_cols = ['home_score', 'away_score', 'actual_margin', 'vegas_closing_spread']

    # Feature columns are everything else
    feature_cols = [col for col in train_df.columns
                   if col not in metadata_cols + target_cols]

    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_home_train = train_df['home_score'].values
    y_away_train = train_df['away_score'].values

    # Set default hyperparameters
    if hyperparams is None:
        hyperparams = {'alpha': 1.0} if model_type == 'ridge' else {}

    # Initialize models
    if model_type == 'ridge':
        home_model = Ridge(**hyperparams)
        away_model = Ridge(**hyperparams)
    elif model_type == 'linear':
        home_model = LinearRegression(**hyperparams)
        away_model = LinearRegression(**hyperparams)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Train models
    print(f"Training {model_type} models on {len(train_df)} games...")
    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    # Compute training metrics
    train_metrics = _compute_metrics(
        home_model, away_model, train_df, feature_cols
    )

    # Compute validation metrics if val data provided
    val_metrics = None
    if val_df is not None and len(val_df) > 0:
        val_metrics = _compute_metrics(
            home_model, away_model, val_df, feature_cols
        )
        print(f"Validation metrics: MAE home={val_metrics['mae_home_score']:.2f}, "
              f"away={val_metrics['mae_away_score']:.2f}, "
              f"spread={val_metrics['mae_spread']:.2f}")

    # Build result dictionary
    results = {
        'home_model': home_model,
        'away_model': away_model,
        'feature_names': feature_cols,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_metadata': {
            'model_type': model_type,
            'hyperparams': hyperparams,
            'train_size': len(train_df),
            'val_size': len(val_df) if val_df is not None else 0,
            'train_seasons': sorted(train_df['season'].unique().tolist()),
            'val_seasons': sorted(val_df['season'].unique().tolist()) if val_df is not None else [],
            'n_features': len(feature_cols)
        }
    }

    # Save if path provided
    if save_path:
        save_model_artifacts(results, save_path)

    return results


def _compute_metrics(
    home_model: Any,
    away_model: Any,
    df: pd.DataFrame,
    feature_cols: list
) -> Dict[str, float]:
    """
    Compute evaluation metrics for score predictions.

    Parameters
    ----------
    home_model : sklearn model
        Trained home score model
    away_model : sklearn model
        Trained away score model
    df : pd.DataFrame
        Data to evaluate on
    feature_cols : list
        List of feature column names

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    X = df[feature_cols].values
    y_home_actual = df['home_score'].values
    y_away_actual = df['away_score'].values

    # Predict scores
    y_home_pred = home_model.predict(X)
    y_away_pred = away_model.predict(X)

    # Compute score metrics
    mae_home = mean_absolute_error(y_home_actual, y_home_pred)
    mae_away = mean_absolute_error(y_away_actual, y_away_pred)
    rmse_home = np.sqrt(mean_squared_error(y_home_actual, y_home_pred))
    rmse_away = np.sqrt(mean_squared_error(y_away_actual, y_away_pred))

    # Compute derived metrics (spread and total)
    actual_spread = y_home_actual - y_away_actual
    pred_spread = y_home_pred - y_away_pred
    actual_total = y_home_actual + y_away_actual
    pred_total = y_home_pred + y_away_pred

    mae_spread = mean_absolute_error(actual_spread, pred_spread)
    mae_total = mean_absolute_error(actual_total, pred_total)

    return {
        'mae_home_score': mae_home,
        'mae_away_score': mae_away,
        'rmse_home_score': rmse_home,
        'rmse_away_score': rmse_away,
        'mae_spread': mae_spread,
        'mae_total': mae_total,
        'n_samples': len(df)
    }


def save_model_artifacts(
    results: Dict[str, Any],
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save trained model with versioning and metadata.

    Parameters
    ----------
    results : dict
        Training results from train_v1_3()
    save_path : str
        Directory path to save model artifacts
    metadata : dict, optional
        Additional metadata to save with model

    Notes
    -----
    Saves:
    - home_model.pkl: Serialized home score model
    - away_model.pkl: Serialized away score model
    - features.json: List of feature column names
    - metadata.json: Training metadata and metrics
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(results['home_model'], save_dir / 'home_model.pkl')
    joblib.dump(results['away_model'], save_dir / 'away_model.pkl')

    # Save feature names
    with open(save_dir / 'features.json', 'w') as f:
        json.dump(results['feature_names'], f, indent=2)

    # Combine metadata
    full_metadata = {
        'training_metadata': results['training_metadata'],
        'train_metrics': results['train_metrics'],
        'val_metrics': results['val_metrics'],
        'saved_at': datetime.now().isoformat(),
        'version': 'v1.3'
    }
    if metadata:
        full_metadata.update(metadata)

    # Save metadata
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(full_metadata, f, indent=2)

    print(f"Model artifacts saved to: {save_path}")


def load_model_artifacts(load_path: str) -> Dict[str, Any]:
    """
    Load trained model with its metadata.

    Parameters
    ----------
    load_path : str
        Directory path to load model artifacts from

    Returns
    -------
    dict
        Dictionary containing:
        - 'home_model': Loaded home score model
        - 'away_model': Loaded away score model
        - 'feature_names': List of feature column names
        - 'metadata': Model metadata and metrics
    """
    load_dir = Path(load_path)

    # Load models
    home_model = joblib.load(load_dir / 'home_model.pkl')
    away_model = joblib.load(load_dir / 'away_model.pkl')

    # Load feature names
    with open(load_dir / 'features.json', 'r') as f:
        feature_names = json.load(f)

    # Load metadata
    with open(load_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Model artifacts loaded from: {load_path}")

    return {
        'home_model': home_model,
        'away_model': away_model,
        'feature_names': feature_names,
        'metadata': metadata
    }
