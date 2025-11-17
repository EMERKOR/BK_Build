"""
Ball Knower v1.3 - Enhanced with Rolling EPA Features

Builds on v1.2 by adding:
- Rolling EPA offense (3, 5, 10 game windows)
- Rolling EPA defense (3, 5, 10 game windows)
- Rolling EPA margin (net EPA advantage)

All features are leak-free (using .shift(1) to exclude current game).

Training data: 2009-2024 seasons
Test data: 2025 season
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
print("BALL KNOWER v1.3 - ENHANCED WITH ROLLING EPA FEATURES")
print("="*80)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/7] Loading historical data...")

# Load nfelo historical games (has EPA data)
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
# FEATURE ENGINEERING - v1.2 FEATURES
# ============================================================================

print("\n[2/7] Engineering v1.2 baseline features...")

# Primary feature: ELO differential
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# Situational adjustments from nfelo
df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

# QB adjustments
df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# Target: Vegas closing line
df['vegas_line'] = df['home_line_close']

print(f"✓ Created 6 baseline features (v1.2)")

# ============================================================================
# FEATURE ENGINEERING - ROLLING EPA FEATURES
# ============================================================================

print("\n[3/7] Engineering rolling EPA features (leak-free)...")

# Load pre-aggregated EPA data (now includes 2025)
epa_file = Path('/home/user/BK_Build/team_week_epa_2013_2025.csv')

if epa_file.exists():
    print("  Loading team-week EPA data...")

    epa_df = pd.read_csv(epa_file)

    # Rename columns for clarity
    epa_df = epa_df.rename(columns={
        'off_epa_per_play': 'offense_epa',
        'def_epa_per_play': 'defense_epa'
    })

    print(f"  Loaded {len(epa_df):,} team-week records (2013-2024)")

    # Sort by team and time
    epa_df = epa_df.sort_values(['team', 'season', 'week'])

    # Calculate rolling EPA features (LEAK-FREE with .shift(1))
    windows = [3, 5, 10]

    for window in windows:
        # Offensive EPA rolling average
        epa_df[f'epa_off_L{window}'] = (
            epa_df.groupby('team')['offense_epa']
            .shift(1)  # CRITICAL: Exclude current game
            .rolling(window, min_periods=1)
            .mean()
        )

        # Defensive EPA rolling average (note: defensive EPA is opponent EPA allowed)
        # Lower is better for defense, so we want negative values
        epa_df[f'epa_def_L{window}'] = (
            epa_df.groupby('team')['defense_epa']
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

        # EPA margin (off - def allowed, so higher is better)
        epa_df[f'epa_margin_L{window}'] = (
            epa_df[f'epa_off_L{window}'] - epa_df[f'epa_def_L{window}']
        )

    # Merge EPA features back to games dataframe
    # Home team
    df = df.merge(
        epa_df[['season', 'week', 'team'] + [f'epa_off_L{w}' for w in windows] +
               [f'epa_def_L{w}' for w in windows] + [f'epa_margin_L{w}' for w in windows]],
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_home')
    ).drop(columns=['team'], errors='ignore')

    # Rename home features
    for window in windows:
        df = df.rename(columns={
            f'epa_off_L{window}': f'home_epa_off_L{window}',
            f'epa_def_L{window}': f'home_epa_def_L{window}',
            f'epa_margin_L{window}': f'home_epa_margin_L{window}'
        })

    # Away team
    df = df.merge(
        epa_df[['season', 'week', 'team'] + [f'epa_off_L{w}' for w in windows] +
               [f'epa_def_L{w}' for w in windows] + [f'epa_margin_L{w}' for w in windows]],
        left_on=['season', 'week', 'away_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_away')
    ).drop(columns=['team'], errors='ignore')

    # Rename away features
    for window in windows:
        df = df.rename(columns={
            f'epa_off_L{window}': f'away_epa_off_L{window}',
            f'epa_def_L{window}': f'away_epa_def_L{window}',
            f'epa_margin_L{window}': f'away_epa_margin_L{window}'
        })

    # Calculate EPA differentials for each window
    for window in windows:
        df[f'epa_margin_diff_L{window}'] = df[f'home_epa_margin_L{window}'] - df[f'away_epa_margin_L{window}']
        df[f'epa_off_diff_L{window}'] = df[f'home_epa_off_L{window}'] - df[f'away_epa_off_L{window}']
        df[f'epa_def_diff_L{window}'] = df[f'home_epa_def_L{window}'] - df[f'away_epa_def_L{window}']

    print(f"✓ Created {len(windows) * 3} rolling EPA differential features")

else:
    print("  ⚠ EPA file not found, skipping EPA features")
    windows = []

# ============================================================================
# PREPARE FEATURE SET
# ============================================================================

print("\n[4/7] Preparing feature matrix...")

# v1.2 baseline features
baseline_features = [
    'nfelo_diff',
    'rest_advantage',
    'div_game',
    'surface_mod',
    'time_advantage',
    'qb_diff'
]

# v1.3 new features: Rolling EPA differentials
epa_features = []
for window in windows:
    epa_features.extend([
        f'epa_margin_diff_L{window}',
        f'epa_off_diff_L{window}',
        f'epa_def_diff_L{window}'
    ])

feature_cols = baseline_features + epa_features

X = df[feature_cols].copy()
y = df['vegas_line'].copy()

# Remove any remaining NaN rows
mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

print(f"✓ Feature matrix: {len(X):,} games × {len(feature_cols)} features")
print(f"  Baseline (v1.2): {len(baseline_features)} features")
print(f"  EPA features:    {len(epa_features)} features")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[5/7] Splitting train/test sets...")

# Time-based split: train on 2013-2024, test on 2025
# (Now we have EPA data through 2025!)
train_mask = df['season'] < 2025
test_mask = df['season'] >= 2025

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"✓ Training set: {len(X_train):,} games (2013-2024)")
print(f"✓ Test set:     {len(X_test):,} games (2025)")

# ============================================================================
# CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# ============================================================================

print("\n[6/7] Cross-validating Ridge alpha parameter...")

# Try different regularization strengths
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
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

print("\n[7/7] Training final v1.3 model...")

model = Ridge(alpha=best_alpha)
model.fit(X_train, y_train)

print(f"✓ Model trained with alpha={best_alpha}")

# Show feature importance (top 10)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 feature coefficients:")
print(feature_importance.head(10).to_string(index=False))
print(f"\nIntercept: {model.intercept_:.3f}")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("EVALUATING v1.3 PERFORMANCE")
print("="*80)

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

print(f"\nTraining Set (2013-2024, n={len(X_train):,}):")
print(f"  MAE:  {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")
print(f"  R²:   {train_r2:.3f}")

print(f"\nTest Set (2025, n={len(X_test):,}):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  RMSE: {test_rmse:.2f} points")
print(f"  R²:   {test_r2:.3f}")

# ============================================================================
# COMPARE TO v1.2
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: v1.3 vs v1.2")
print("="*80)

# Load v1.2 results
import json
v1_2_file = Path('/home/user/BK_Build/output/ball_knower_v1_2_model.json')

if v1_2_file.exists():
    with open(v1_2_file, 'r') as f:
        v1_2_params = json.load(f)

    print("\nModel Performance Comparison:")
    print(f"\n{'Metric':<20} {'v1.2':<12} {'v1.3':<12} {'Change':<12}")
    print("-" * 60)

    # Test MAE
    v1_2_mae = v1_2_params['test_mae']
    mae_change = test_mae - v1_2_mae
    mae_pct = (mae_change / v1_2_mae) * 100
    print(f"{'Test MAE':<20} {v1_2_mae:<12.2f} {test_mae:<12.2f} {mae_change:>+.2f} ({mae_pct:>+.1f}%)")

    # Test R²
    v1_2_r2 = v1_2_params['test_r2']
    r2_change = test_r2 - v1_2_r2
    print(f"{'Test R²':<20} {v1_2_r2:<12.3f} {test_r2:<12.3f} {r2_change:>+.3f}")

    # Features
    print(f"{'Features':<20} {6:<12} {len(feature_cols):<12} {len(feature_cols) - 6:>+d}")

    if mae_change < 0:
        print(f"\n✓ v1.3 IMPROVED by {abs(mae_change):.2f} MAE points ({abs(mae_pct):.1f}%)")
    else:
        print(f"\n⚠ v1.3 performed worse by {mae_change:.2f} MAE points ({mae_pct:.1f}%)")

else:
    print("\n⚠ v1.2 results not found, skipping comparison")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING v1.3 MODEL")
print("="*80)

output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

# Save model coefficients
model_params = {
    'version': '1.3',
    'description': 'Enhanced with rolling EPA features (3, 5, 10 game windows)',
    'intercept': model.intercept_,
    'coefficients': dict(zip(feature_cols, model.coef_)),
    'features': feature_cols,
    'alpha': best_alpha,
    'train_mae': train_mae,
    'train_r2': train_r2,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'n_train': len(X_train),
    'n_test': len(X_test)
}

with open(output_dir / 'ball_knower_v1_3_model.json', 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"\n✓ Model parameters saved to: {output_dir / 'ball_knower_v1_3_model.json'}")

# Save test set predictions
df_test = df[test_mask].copy()
df_test['bk_v1_3_pred'] = y_test_pred
df_test['edge'] = df_test['bk_v1_3_pred'] - df_test['vegas_line']
df_test['abs_edge'] = df_test['edge'].abs()

test_results = df_test[['game_id', 'season', 'week', 'away_team', 'home_team',
                         'vegas_line', 'bk_v1_3_pred', 'edge', 'abs_edge']].copy()
test_results.to_csv(output_dir / 'ball_knower_v1_3_test_predictions.csv', index=False)

print(f"✓ Test predictions saved to: {output_dir / 'ball_knower_v1_3_test_predictions.csv'}")

# Feature importance
feature_importance.to_csv(output_dir / 'ball_knower_v1_3_feature_importance.csv', index=False)
print(f"✓ Feature importance saved to: {output_dir / 'ball_knower_v1_3_feature_importance.csv'}")

print("\n" + "="*80)
print("v1.3 SUMMARY")
print("="*80)

print(f"""
Ball Knower v1.3 successfully trained!

Key Enhancements over v1.2:
- Added rolling EPA features (3, 5, 10 game windows)
- 9 new EPA differential features (offense, defense, margin)
- Total features: {len(feature_cols)} (v1.2: 6)
- All features are leak-free (using .shift(1))

Performance:
- Test MAE: {test_mae:.2f} points
- Test R²: {test_r2:.3f}
- Test RMSE: {test_rmse:.2f} points

Next Steps:
- v1.4: Add Next Gen Stats
- v1.5: Add weather, injuries, roster changes
- Compare all versions to identify best model
""")

print("="*80 + "\n")
