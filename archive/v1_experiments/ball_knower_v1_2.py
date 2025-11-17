"""
Ball Knower v1.2 - ML Correction Layer

Properly trained model using historical data with:
- Ridge regression for robust coefficient estimation
- Time-series cross-validation (no look-ahead bias)
- Multiple features (ELO, rest, divisional, etc.)
- Holdout test set for final evaluation

Training data: 2009-2024 seasons (4,345 games)
Test data: 2025 season (165 games)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BALL KNOWER v1.2 - ML CORRECTION LAYER")
print("="*80)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/6] Loading historical data...")

# Load nfelo historical games
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

# Extract season/week/teams
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Filter to complete data
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"✓ Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n[2/6] Engineering features...")

# Primary feature: ELO differential
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# Situational adjustments from nfelo
df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']  # Combined rest effect

df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

# QB adjustments
df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# Target: Vegas closing line (what we're trying to predict)
df['vegas_line'] = df['home_line_close']

# Feature set
feature_cols = [
    'nfelo_diff',
    'rest_advantage',
    'div_game',
    'surface_mod',
    'time_advantage',
    'qb_diff'
]

X = df[feature_cols].copy()
y = df['vegas_line'].copy()

# Remove any remaining NaN rows
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]
df = df[mask].reset_index(drop=True)

print(f"✓ Engineered {len(feature_cols)} features for {len(X):,} games")
print(f"  Features: {', '.join(feature_cols)}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[3/6] Splitting train/test sets...")

# Time-based split: train on 2009-2024, test on 2025
train_mask = df['season'] < 2025
test_mask = df['season'] >= 2025

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"✓ Training set: {len(X_train):,} games (2009-2024)")
print(f"✓ Test set:     {len(X_test):,} games (2025)")

# ============================================================================
# CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# ============================================================================

print("\n[4/6] Cross-validating Ridge alpha parameter...")

# Try different regularization strengths
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
tscv = TimeSeriesSplit(n_splits=5)

best_alpha = None
best_cv_score = float('inf')

cv_results = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train = X_train.iloc[train_idx]
        y_cv_train = y_train.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_val = y_train.iloc[val_idx]

        model.fit(X_cv_train, y_cv_train)
        y_pred = model.predict(X_cv_val)
        mae = mean_absolute_error(y_cv_val, y_pred)
        cv_scores.append(mae)

    mean_cv_mae = np.mean(cv_scores)
    cv_results.append({
        'alpha': alpha,
        'cv_mae': mean_cv_mae
    })

    if mean_cv_mae < best_cv_score:
        best_cv_score = mean_cv_mae
        best_alpha = alpha

cv_df = pd.DataFrame(cv_results)
print("\nCross-validation results:")
print(cv_df.to_string(index=False))
print(f"\n✓ Best alpha: {best_alpha} (CV MAE: {best_cv_score:.2f})")

# ============================================================================
# TRAIN FINAL MODEL
# ============================================================================

print("\n[5/6] Training final model...")

model = Ridge(alpha=best_alpha)
model.fit(X_train, y_train)

print(f"✓ Model trained with alpha={best_alpha}")

# Show feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature coefficients:")
print(feature_importance.to_string(index=False))
print(f"Intercept: {model.intercept_:.3f}")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n[6/6] Evaluating on 2025 test set...")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Test metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)

print(f"\nTraining Set (2009-2024, n={len(X_train):,}):")
print(f"  MAE:  {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")
print(f"  R²:   {train_r2:.3f}")

print(f"\nTest Set (2025, n={len(X_test):,}):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  RMSE: {test_rmse:.2f} points")
print(f"  R²:   {test_r2:.3f}")

# Edge analysis on test set
df_test = df[test_mask].copy()
df_test['bk_v1_2_pred'] = y_test_pred
df_test['edge'] = df_test['bk_v1_2_pred'] - df_test['vegas_line']
df_test['abs_edge'] = df_test['edge'].abs()

print("\n" + "="*80)
print("TEST SET EDGE ANALYSIS (2025 Season)")
print("="*80)

print(f"\nMean absolute edge: {df_test['abs_edge'].mean():.2f} points")
print(f"Median absolute edge: {df_test['abs_edge'].median():.2f} points")
print(f"Max edge: {df_test['abs_edge'].max():.2f} points")

# Edge bins
edge_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for threshold in edge_bins:
    count = len(df_test[df_test['abs_edge'] >= threshold])
    pct = count / len(df_test) * 100
    print(f"Games with {threshold}+ edge: {count:3} ({pct:4.1f}%)")

# Show biggest edges in 2025
print("\n" + "="*80)
print("LARGEST EDGES IN 2025 TEST SET")
print("="*80)

top_edges = df_test.nlargest(10, 'abs_edge')[
    ['game_id', 'vegas_line', 'bk_v1_2_pred', 'edge', 'abs_edge']
].copy()
top_edges = top_edges.round(2)
print("\n" + top_edges.to_string(index=False))

# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

# Save model coefficients
model_params = {
    'intercept': model.intercept_,
    'coefficients': dict(zip(feature_cols, model.coef_)),
    'alpha': best_alpha,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_r2': train_r2,
    'test_r2': test_r2
}

import json
with open(output_dir / 'ball_knower_v1_2_model.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"\n✓ Model parameters saved to: {output_dir / 'ball_knower_v1_2_model.json'}")

# Save test set predictions
test_results = df_test[['game_id', 'season', 'week', 'away_team', 'home_team',
                         'vegas_line', 'bk_v1_2_pred', 'edge', 'abs_edge']].copy()
test_results.to_csv(output_dir / 'ball_knower_v1_2_test_predictions.csv', index=False)

print(f"✓ Test predictions saved to: {output_dir / 'ball_knower_v1_2_test_predictions.csv'}")

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)

print(f"""
Ball Knower v1.2 successfully trained!

Key Improvements over v1.0/v1.1:
- Trained on 4,345 historical games (not just 14!)
- Time-series cross-validation (no look-ahead bias)
- Regularization (Ridge α={best_alpha}) prevents overfitting
- Multiple features (ELO + situational factors)
- Proper holdout test set (2025 season)

Performance:
- Test MAE: {test_mae:.2f} points (vs Vegas closing lines)
- Test R²: {test_r2:.3f}
- {len(df_test[df_test['abs_edge'] >= 2.0])} games with 2+ point edge in 2025

Next Steps:
- Apply v1.2 to current week for live betting recommendations
- Track performance vs actual game outcomes (not just Vegas)
- Implement bankroll management and bet sizing
- Add more features (weather, injuries, line movement)
""")

print("="*80 + "\n")
