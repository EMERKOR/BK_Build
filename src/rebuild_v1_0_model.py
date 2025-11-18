"""
Ball Knower v1.0 - Actual Margin Model

A clean, simple model that predicts actual game margins (not Vegas lines).

Philosophy:
- Model the game outcomes first
- Then compare to Vegas to find edges
- Avoid training on Vegas lines (which creates artificial "edges" from model errors)

Model:
- Features: nfelo_diff (starting_nfelo_home - starting_nfelo_away)
- Target: actual_margin (home_score - away_score)
- Method: Simple linear regression with intercept

Training period: 2009-2023 (holdout 2024+ for validation)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ball_knower.datasets import v1_0 as ds_v1_0

print("\n" + "="*80)
print("BALL KNOWER v1.0 - ACTUAL MARGIN MODEL")
print("="*80)

# ============================================================================
# LOAD TRAINING DATA
# ============================================================================

print("\n[1/4] Loading training data...")

# Train on 2009-2023, save 2024+ for potential validation
df = ds_v1_0.build_v1_0_training_frame(min_season=2009, max_season=2023)

print(f"✓ Loaded {len(df):,} games from {df['season'].min()}-{df['season'].max()}")

# ============================================================================
# PREPARE FEATURES AND TARGET
# ============================================================================

print("\n[2/4] Preparing features and target...")

# Features: Start with just nfelo_diff + intercept
# (We can add structural features later if needed)
X = df[['nfelo_diff']].copy()
y = df['actual_margin'].copy()

# Verify no missing data
assert X.notna().all().all(), "Missing values in features"
assert y.notna().all(), "Missing values in target"

print(f"✓ Features: {list(X.columns)}")
print(f"✓ Target: actual_margin")
print(f"✓ Training samples: {len(X):,}")

# ============================================================================
# FIT LINEAR REGRESSION
# ============================================================================

print("\n[3/4] Fitting linear regression (actual_margin ~ nfelo_diff)...")

# Simple linear regression using sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X, y)

# Extract coefficients
intercept = model.intercept_
coef_nfelo_diff = model.coef_[0]

print(f"\n✓ Model fitted successfully")
print(f"\nCoefficients:")
print(f"  Intercept:       {intercept:>8.4f}")
print(f"  nfelo_diff:      {coef_nfelo_diff:>8.4f}")

# ============================================================================
# EVALUATE TRAINING PERFORMANCE
# ============================================================================

print("\n[4/4] Evaluating training performance...")

# Predictions on training set
y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"\nTraining Set Performance (vs actual margins):")
print(f"  MAE:  {mae:.2f} points")
print(f"  RMSE: {rmse:.2f} points")
print(f"  R²:   {r2:.3f}")

# Quick sanity checks
residuals = y - y_pred
print(f"\nResidual Analysis:")
print(f"  Mean residual:   {residuals.mean():>8.4f} (should be ~0)")
print(f"  Median residual: {residuals.median():>8.4f}")
print(f"  Std residual:    {residuals.std():>8.4f}")

# ============================================================================
# SAVE MODEL PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

output_dir = Path(project_root) / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

model_params = {
    'version': 'v1.0',
    'model_type': 'linear_regression',
    'target': 'actual_margin',
    'features': ['nfelo_diff'],
    'intercept': float(intercept),
    'coef_nfelo_diff': float(coef_nfelo_diff),
    'training_period': {
        'min_season': int(df['season'].min()),
        'max_season': int(df['season'].max()),
        'n_games': int(len(df))
    },
    'training_metrics': {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    },
    'notes': 'Predicts actual game margin, NOT Vegas line'
}

params_file = output_dir / 'v1_0_model_params.json'
with open(params_file, 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"\n✓ Model parameters saved to: {params_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)

print(f"""
Ball Knower v1.0 successfully trained!

Key Characteristics:
- Simple linear model: actual_margin ~ nfelo_diff
- Trained on {len(df):,} games ({df['season'].min()}-{df['season'].max()})
- Target: actual game outcomes (NOT Vegas lines)
- Philosophy: Model the game first, then find edges vs Vegas

Model Equation:
  predicted_margin = {intercept:.4f} + {coef_nfelo_diff:.4f} * nfelo_diff

Performance (vs actual game margins):
- MAE:  {mae:.2f} points
- RMSE: {rmse:.2f} points
- R²:   {r2:.3f}

Interpretation:
- Intercept ~{intercept:.1f}: Home field advantage (points)
- Coefficient ~{coef_nfelo_diff:.4f}: Each 100 ELO points = {coef_nfelo_diff*100:.1f} points on spread

Next Steps:
- Run backtest_v1_0_actual_margin.py to evaluate vs Vegas
- Compare v1.0 (actual margin) to v1.2 (Vegas line) predictions
- Identify where v1.0 finds edges vs the market
""")

print("="*80 + "\n")
