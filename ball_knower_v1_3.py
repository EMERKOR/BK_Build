"""
Ball Knower v1.3 - EPA-Enhanced Model

Builds on v1.2 by adding professional EPA (Expected Points Added) features
from play-by-play data aggregated to team-week level.

New Features:
- EPA per play (offensive & defensive)
- Success rate (offensive & defensive)
- EPA differentials
- Success rate differentials

Training: 2009-2024 games (same as v1.2)
Test: 2025 season (same holdout as v1.2)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import config, team_mapping

print("\n" + "="*80)
print("BALL KNOWER v1.3 - EPA-ENHANCED MODEL")
print("="*80)

# ============================================================================
# LOAD EPA DATA
# ============================================================================

print("\n[1/7] Loading EPA data...")

epa_file = project_root / 'data' / 'team_week_epa_2013_2024.csv'

if not epa_file.exists():
    print(f"  ✗ ERROR: EPA file not found: {epa_file}")
    print(f"  Please upload the file and re-run this script.")
    sys.exit(1)

epa_df = pd.read_csv(epa_file)

print(f"  ✓ Loaded {len(epa_df):,} team-week records")
print(f"  Seasons: {epa_df['season'].min()}-{epa_df['season'].max()}")
print(f"  Teams: {epa_df['team'].nunique()}")
print(f"  Columns: {epa_df.columns.tolist()}")

# Normalize team names
epa_df['team'] = epa_df['team'].map(team_mapping.NFLVERSE_TO_STD).fillna(epa_df['team'])

# ============================================================================
# LOAD NFELO HISTORICAL DATA
# ============================================================================

print("\n[2/7] Loading nfelo historical data...")

nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

# Extract season/week/teams
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Normalize team names
df['away_team'] = df['away_team'].map(team_mapping.NFELO_TO_STD).fillna(df['away_team'])
df['home_team'] = df['home_team'].map(team_mapping.NFELO_TO_STD).fillna(df['home_team'])

# Filter to complete data
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"  ✓ Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

# ============================================================================
# MERGE EPA FEATURES
# ============================================================================

print("\n[3/7] Merging EPA features...")

# Merge home team EPA
epa_home = epa_df.copy()
epa_home.columns = ['season', 'week', 'home_team'] + [f'home_{col}' for col in epa_df.columns if col not in ['season', 'week', 'team']]

df = df.merge(
    epa_home,
    on=['season', 'week', 'home_team'],
    how='left'
)

print(f"  ✓ Merged home team EPA")

# Merge away team EPA
epa_away = epa_df.copy()
epa_away.columns = ['season', 'week', 'away_team'] + [f'away_{col}' for col in epa_df.columns if col not in ['season', 'week', 'team']]

df = df.merge(
    epa_away,
    on=['season', 'week', 'away_team'],
    how='left'
)

print(f"  ✓ Merged away team EPA")

# Count coverage
epa_coverage = df[['home_off_epa_per_play', 'away_off_epa_per_play']].notna().all(axis=1).sum()
print(f"  EPA coverage: {epa_coverage:,}/{len(df):,} games ({epa_coverage/len(df)*100:.1f}%)")

# ============================================================================
# ENGINEER FEATURES
# ============================================================================

print("\n[4/7] Engineering features...")

# v1.2 baseline features
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# NEW: EPA differential features
df['epa_off_diff'] = df['home_off_epa_per_play'] - df['away_off_epa_per_play']
df['epa_def_diff'] = df['home_def_epa_per_play'] - df['away_def_epa_per_play']  # Lower is better for defense
df['success_rate_off_diff'] = df['home_off_success_rate'] - df['away_off_success_rate']
df['success_rate_def_diff'] = df['home_def_success_rate'] - df['away_def_success_rate']

# Target: Vegas closing line
df['vegas_line'] = df['home_line_close']

# Remove NaN rows
v1_2_features = ['nfelo_diff', 'rest_advantage', 'div_game', 'surface_mod', 'time_advantage', 'qb_diff']
v1_3_features = v1_2_features + ['epa_off_diff', 'epa_def_diff', 'success_rate_off_diff', 'success_rate_def_diff']

# For v1.3, only use games with EPA data
mask = df[v1_3_features + ['vegas_line']].notna().all(axis=1)
df_v1_3 = df[mask].copy()

print(f"  ✓ v1.3 training set: {len(df_v1_3):,} games with complete EPA data")
print(f"  v1.2 features: {len(v1_2_features)}")
print(f"  v1.3 features: {len(v1_3_features)} (+{len(v1_3_features) - len(v1_2_features)} EPA features)")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[5/7] Splitting train/test...")

train_df = df_v1_3[df_v1_3['season'] < 2025].copy()
test_df = df_v1_3[df_v1_3['season'] == 2025].copy()

print(f"  Training: {len(train_df):,} games (2013-2024)")
print(f"  Test:     {len(test_df):,} games (2025)")

X_train = train_df[v1_3_features].values
y_train = train_df['vegas_line'].values

X_test = test_df[v1_3_features].values
y_test = test_df['vegas_line'].values

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n[6/7] Training v1.3 model...")

# Use same Ridge alpha as v1.2 for fair comparison
model = Ridge(alpha=100.0)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print(f"\n  Training Performance:")
print(f"    MAE:  {train_mae:.3f} points")
print(f"    RMSE: {train_rmse:.3f} points")
print(f"    R²:   {train_r2:.3f}")

print(f"\n  Test Performance:")
print(f"    MAE:  {test_mae:.3f} points")
print(f"    RMSE: {test_rmse:.3f} points")
print(f"    R²:   {test_r2:.3f}")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': v1_3_features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n  Feature Importance (by absolute coefficient):")
print(feature_importance.to_string(index=False))

# ============================================================================
# COMPARE TO v1.2
# ============================================================================

print("\n[7/7] Comparing v1.3 to v1.2...")

# Load v1.2 model
v1_2_model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'
with open(v1_2_model_file, 'r') as f:
    v1_2_params = json.load(f)

print(f"\nv1.2 Baseline (from saved model):")
print(f"  Test MAE:  {v1_2_params['test_mae']:.3f} points")
print(f"  Test R²:   {v1_2_params['test_r2']:.3f}")

print(f"\nv1.3 EPA-Enhanced:")
print(f"  Test MAE:  {test_mae:.3f} points")
print(f"  Test R²:   {test_r2:.3f}")

mae_improvement = v1_2_params['test_mae'] - test_mae
mae_improvement_pct = (mae_improvement / v1_2_params['test_mae']) * 100

r2_improvement = test_r2 - v1_2_params['test_r2']

print(f"\nImprovement:")
print(f"  MAE: {mae_improvement:+.3f} points ({mae_improvement_pct:+.1f}%)")
print(f"  R²:  {r2_improvement:+.3f}")

if mae_improvement > 0:
    print(f"\n  ✓ v1.3 is BETTER than v1.2!")
else:
    print(f"\n  ⚠ v1.3 is not better than v1.2 (may need feature tuning)")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\nSaving v1.3 model outputs...")

# Save model parameters
model_params = {
    'model_type': 'Ridge',
    'alpha': 100.0,
    'features': v1_3_features,
    'intercept': float(model.intercept_),
    'coefficients': {feat: float(coef) for feat, coef in zip(v1_3_features, model.coef_)},
    'train_games': len(train_df),
    'test_games': len(test_df),
    'train_mae': float(train_mae),
    'train_rmse': float(train_rmse),
    'train_r2': float(train_r2),
    'test_mae': float(test_mae),
    'test_rmse': float(test_rmse),
    'test_r2': float(test_r2),
    'seasons': f"{train_df['season'].min()}-{train_df['season'].max()}",
}

model_file = config.OUTPUT_DIR / 'ball_knower_v1_3_model.json'
with open(model_file, 'w') as f:
    json.dump(model_params, f, indent=2)
print(f"  ✓ Model parameters: {model_file}")

# Save test predictions
test_df['bk_v1_3_spread'] = test_pred
test_df['v1_3_edge'] = test_pred - y_test

test_output = test_df[['game_id', 'season', 'week', 'away_team', 'home_team',
                        'vegas_line', 'bk_v1_3_spread', 'v1_3_edge']].copy()

test_pred_file = config.OUTPUT_DIR / 'ball_knower_v1_3_test_predictions.csv'
test_output.to_csv(test_pred_file, index=False)
print(f"  ✓ Test predictions: {test_pred_file}")

# Save feature importance
feature_importance.to_csv(config.OUTPUT_DIR / 'ball_knower_v1_3_feature_importance.csv', index=False)
print(f"  ✓ Feature importance: ball_knower_v1_3_feature_importance.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ball Knower v1.3 - EPA-Enhanced Model', fontsize=16, fontweight='bold')

# 1. Feature importance
ax1 = axes[0, 0]
feature_importance_sorted = feature_importance.sort_values('coefficient')
colors = ['red' if c < 0 else 'green' for c in feature_importance_sorted['coefficient']]
ax1.barh(range(len(feature_importance_sorted)), feature_importance_sorted['coefficient'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(feature_importance_sorted)))
ax1.set_yticklabels(feature_importance_sorted['feature'], fontsize=9)
ax1.set_xlabel('Coefficient', fontsize=10)
ax1.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(True, alpha=0.3, axis='x')

# 2. Actual vs Predicted
ax2 = axes[0, 1]
ax2.scatter(y_test, test_pred, alpha=0.5, s=30)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Vegas Line', fontsize=10)
ax2.set_ylabel('v1.3 Prediction', fontsize=10)
ax2.set_title(f'Test Set: Actual vs Predicted (R²={test_r2:.3f})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Residuals distribution
ax3 = axes[1, 0]
residuals = test_pred - y_test
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Prediction Error (points)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title(f'Test Set Residuals (MAE={test_mae:.2f})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. v1.2 vs v1.3 comparison
ax4 = axes[1, 1]
comparison_data = {
    'Model': ['v1.2\n(nfelo only)', 'v1.3\n(nfelo + EPA)'],
    'MAE': [v1_2_params['test_mae'], test_mae],
    'R²': [v1_2_params['test_r2'], test_r2]
}

x_pos = np.arange(len(comparison_data['Model']))
width = 0.35

ax4_twin = ax4.twinx()
bars1 = ax4.bar(x_pos - width/2, comparison_data['MAE'], width, label='MAE (lower is better)', color='salmon', alpha=0.7)
bars2 = ax4_twin.bar(x_pos + width/2, comparison_data['R²'], width, label='R² (higher is better)', color='lightblue', alpha=0.7)

ax4.set_xlabel('Model', fontsize=10)
ax4.set_ylabel('MAE (points)', fontsize=10, color='salmon')
ax4_twin.set_ylabel('R²', fontsize=10, color='lightblue')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(comparison_data['Model'], fontsize=10)
ax4.set_title('v1.2 vs v1.3 Performance', fontsize=12, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='salmon')
ax4_twin.tick_params(axis='y', labelcolor='lightblue')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

plot_file = config.OUTPUT_DIR / 'ball_knower_v1_3_analysis.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Visualization: {plot_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("v1.3 MODEL SUMMARY")
print("="*80)

print(f"""
Training Data:
  Games: {len(train_df):,} (2013-2024 with EPA coverage)
  Features: {len(v1_3_features)}
    - v1.2 baseline: {', '.join(v1_2_features)}
    - NEW EPA: {', '.join([f for f in v1_3_features if f not in v1_2_features])}

Test Performance (2025 season):
  MAE:  {test_mae:.3f} points
  RMSE: {test_rmse:.3f} points
  R²:   {test_r2:.3f}

Comparison to v1.2:
  v1.2 MAE: {v1_2_params['test_mae']:.3f} points
  v1.3 MAE: {test_mae:.3f} points
  Improvement: {mae_improvement:+.3f} points ({mae_improvement_pct:+.1f}%)

Top 3 Most Important Features:
  1. {feature_importance.iloc[0]['feature']}: {feature_importance.iloc[0]['coefficient']:.4f}
  2. {feature_importance.iloc[1]['feature']}: {feature_importance.iloc[1]['coefficient']:.4f}
  3. {feature_importance.iloc[2]['feature']}: {feature_importance.iloc[2]['coefficient']:.4f}

Files Saved:
  - {model_file.name}
  - {test_pred_file.name}
  - ball_knower_v1_3_feature_importance.csv
  - ball_knower_v1_3_analysis.png
""")

print("="*80 + "\n")
