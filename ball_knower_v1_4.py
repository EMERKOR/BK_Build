"""
Ball Knower v1.4 - Enhanced Feature Engineering

Improves on v1.3 by adding:
- Rolling averages for EPA (3-game and 5-game windows)
- Recent form indicators (last 3 games performance)
- Interaction terms (EPA × ELO)
- Momentum features
- Schedule strength adjustments

Goal: Beat v1.2 baseline by properly leveraging EPA data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import config

print("\n" + "="*80)
print("BALL KNOWER v1.4 - ENHANCED FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# LOAD EPA DATA AND BUILD ROLLING FEATURES
# ============================================================================

print("\n[1/8] Loading and enhancing EPA data...")

epa_file = project_root / 'data' / 'team_week_epa_2013_2024.csv'

if not epa_file.exists():
    print(f"  ✗ ERROR: EPA file required for v1.4")
    sys.exit(1)

epa_df = pd.read_csv(epa_file)
print(f"  ✓ Loaded {len(epa_df):,} team-week records")

# Sort by team, season, week for rolling calculations
epa_df = epa_df.sort_values(['team', 'season', 'week'])

# Calculate rolling averages (3-game and 5-game windows)
print("  Calculating rolling averages...")

epa_enhanced = []

for team in epa_df['team'].unique():
    team_data = epa_df[epa_df['team'] == team].copy()

    # Rolling 3-game averages
    team_data['off_epa_roll3'] = team_data['off_epa_per_play'].rolling(window=3, min_periods=1).mean()
    team_data['def_epa_roll3'] = team_data['def_epa_per_play'].rolling(window=3, min_periods=1).mean()
    team_data['success_off_roll3'] = team_data['off_success_rate'].rolling(window=3, min_periods=1).mean()
    team_data['success_def_roll3'] = team_data['def_success_rate'].rolling(window=3, min_periods=1).mean()

    # Rolling 5-game averages
    team_data['off_epa_roll5'] = team_data['off_epa_per_play'].rolling(window=5, min_periods=1).mean()
    team_data['def_epa_roll5'] = team_data['def_epa_per_play'].rolling(window=5, min_periods=1).mean()

    # Recent momentum (last 3 vs previous 3)
    team_data['off_epa_momentum'] = (
        team_data['off_epa_per_play'].rolling(window=3, min_periods=1).mean() -
        team_data['off_epa_per_play'].shift(3).rolling(window=3, min_periods=1).mean()
    )

    epa_enhanced.append(team_data)

epa_df = pd.concat(epa_enhanced, ignore_index=True)

print(f"  ✓ Added rolling averages and momentum features")

# ============================================================================
# LOAD NFELO DATA
# ============================================================================

print("\n[2/8] Loading nfelo historical data...")

nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"  ✓ Loaded {len(df):,} games")

# ============================================================================
# MERGE ENHANCED EPA FEATURES
# ============================================================================

print("\n[3/8] Merging enhanced EPA features...")

# Merge home team EPA (rolling averages)
epa_home = epa_df.copy()
epa_home_cols = ['season', 'week', 'team', 'off_epa_roll3', 'def_epa_roll3',
                 'success_off_roll3', 'success_def_roll3', 'off_epa_roll5',
                 'def_epa_roll5', 'off_epa_momentum']

epa_home = epa_home[epa_home_cols].copy()
epa_home.columns = ['season', 'week', 'home_team'] + [f'home_{col}' for col in epa_home_cols[3:]]

df = df.merge(epa_home, on=['season', 'week', 'home_team'], how='left')

# Merge away team EPA
epa_away = epa_df[epa_home_cols].copy()
epa_away.columns = ['season', 'week', 'away_team'] + [f'away_{col}' for col in epa_home_cols[3:]]

df = df.merge(epa_away, on=['season', 'week', 'away_team'], how='left')

epa_coverage = df[['home_off_epa_roll3', 'away_off_epa_roll3']].notna().all(axis=1).sum()
print(f"  ✓ EPA coverage: {epa_coverage:,}/{len(df):,} games ({epa_coverage/len(df)*100:.1f}%)")

# ============================================================================
# ENGINEER ENHANCED FEATURES
# ============================================================================

print("\n[4/8] Engineering enhanced features...")

# Baseline features
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
df['rest_advantage'] = df['home_bye_mod'].fillna(0) + df['away_bye_mod'].fillna(0)
df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# Enhanced EPA differentials (using rolling averages)
df['epa_off_diff_roll3'] = df['home_off_epa_roll3'] - df['away_off_epa_roll3']
df['epa_def_diff_roll3'] = df['home_def_epa_roll3'] - df['away_def_epa_roll3']
df['success_off_diff_roll3'] = df['home_success_off_roll3'] - df['away_success_off_roll3']
df['success_def_diff_roll3'] = df['home_success_def_roll3'] - df['away_success_def_roll3']

df['epa_off_diff_roll5'] = df['home_off_epa_roll5'] - df['away_off_epa_roll5']
df['epa_def_diff_roll5'] = df['home_def_epa_roll5'] - df['away_def_epa_roll5']

# Momentum differential
df['momentum_diff'] = df['home_off_epa_momentum'].fillna(0) - df['away_off_epa_momentum'].fillna(0)

# Interaction terms (EPA × ELO)
df['epa_elo_interaction'] = df['epa_off_diff_roll3'] * df['nfelo_diff']

# Target
df['vegas_line'] = df['home_line_close']

# Define feature sets
baseline_features = ['nfelo_diff', 'rest_advantage', 'div_game', 'surface_mod', 'time_advantage', 'qb_diff']

enhanced_features = baseline_features + [
    'epa_off_diff_roll3', 'epa_def_diff_roll3',
    'success_off_diff_roll3', 'success_def_diff_roll3',
    'epa_off_diff_roll5', 'epa_def_diff_roll5',
    'momentum_diff', 'epa_elo_interaction'
]

# Filter to complete data
mask = df[enhanced_features + ['vegas_line']].notna().all(axis=1)
df_v1_4 = df[mask].copy()

print(f"  v1.4 features: {len(enhanced_features)}")
print(f"  Training set: {len(df_v1_4):,} games")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[5/8] Splitting train/test...")

train_df = df_v1_4[df_v1_4['season'] < 2024].copy()
test_df = df_v1_4[df_v1_4['season'] == 2024].copy()

print(f"  Training: {len(train_df):,} games (2013-2023)")
print(f"  Test:     {len(test_df):,} games (2024)")

X_train = train_df[enhanced_features].values
y_train = train_df['vegas_line'].values

X_test = test_df[enhanced_features].values
y_test = test_df['vegas_line'].values

# ============================================================================
# TRAIN MODEL WITH CROSS-VALIDATION
# ============================================================================

print("\n[6/8] Training v1.4 with cross-validation...")

# Try multiple alpha values
alphas = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]

model = RidgeCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)

print(f"  ✓ Best alpha: {model.alpha_}")

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

# Feature importance
feature_importance = pd.DataFrame({
    'feature': enhanced_features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n  Top 10 Features:")
print(feature_importance.head(10)[['feature', 'coefficient']].to_string(index=False))

# ============================================================================
# COMPARE TO v1.2 AND v1.3
# ============================================================================

print("\n[7/8] Comparing to previous versions...")

# Load v1.2
v1_2_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'
# Load v1.3
v1_3_file = config.OUTPUT_DIR / 'ball_knower_v1_3_model.json'

print(f"\nModel Comparison:")
print(f"{'Model':<10s} {'Test MAE':<12s} {'Test R²':<10s} {'Features':<10s}")
print("-" * 45)

if v1_2_file.exists():
    with open(v1_2_file, 'r') as f:
        v1_2 = json.load(f)
    print(f"{'v1.2':<10s} {v1_2['test_mae']:<12.3f} {v1_2['test_r2']:<10.3f} {len(v1_2['coefficients']):<10d}")

if v1_3_file.exists():
    with open(v1_3_file, 'r') as f:
        v1_3 = json.load(f)
    print(f"{'v1.3':<10s} {v1_3['test_mae']:<12.3f} {v1_3['test_r2']:<10.3f} {len(v1_3['coefficients']):<10d}")

print(f"{'v1.4':<10s} {test_mae:<12.3f} {test_r2:<10.3f} {len(enhanced_features):<10d}")

if v1_2_file.exists():
    improvement_v1_2 = ((v1_2['test_mae'] - test_mae) / v1_2['test_mae']) * 100
    print(f"\nv1.4 vs v1.2: {improvement_v1_2:+.1f}% MAE improvement")

if v1_3_file.exists():
    improvement_v1_3 = ((v1_3['test_mae'] - test_mae) / v1_3['test_mae']) * 100
    print(f"v1.4 vs v1.3: {improvement_v1_3:+.1f}% MAE improvement")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n[8/8] Saving model outputs...")

model_params = {
    'model_type': 'RidgeCV',
    'alpha': float(model.alpha_),
    'features': enhanced_features,
    'intercept': float(model.intercept_),
    'coefficients': {feat: float(coef) for feat, coef in zip(enhanced_features, model.coef_)},
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

model_file = config.OUTPUT_DIR / 'ball_knower_v1_4_model.json'
with open(model_file, 'w') as f:
    json.dump(model_params, f, indent=2)
print(f"  ✓ {model_file}")

# Save test predictions
test_df['bk_v1_4_spread'] = test_pred
test_df['v1_4_edge'] = test_pred - y_test

test_output = test_df[['game_id', 'season', 'week', 'away_team', 'home_team',
                        'vegas_line', 'bk_v1_4_spread', 'v1_4_edge']].copy()

test_pred_file = config.OUTPUT_DIR / 'ball_knower_v1_4_test_predictions.csv'
test_output.to_csv(test_pred_file, index=False)
print(f"  ✓ {test_pred_file}")

# Save feature importance
feature_importance.to_csv(config.OUTPUT_DIR / 'ball_knower_v1_4_feature_importance.csv', index=False)
print(f"  ✓ ball_knower_v1_4_feature_importance.csv")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ball Knower v1.4 - Enhanced Feature Engineering', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test, test_pred, alpha=0.5, s=30)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Vegas Line', fontsize=10)
ax1.set_ylabel('v1.4 Prediction', fontsize=10)
ax1.set_title(f'Predictions (R²={test_r2:.3f})', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Residuals
ax2 = axes[0, 1]
residuals = test_pred - y_test
ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Error (points)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title(f'Residuals (MAE={test_mae:.2f})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Feature importance
ax3 = axes[1, 0]
top_10 = feature_importance.head(10).sort_values('coefficient')
colors = ['red' if c < 0 else 'green' for c in top_10['coefficient']]
ax3.barh(range(len(top_10)), top_10['coefficient'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(top_10)))
ax3.set_yticklabels(top_10['feature'], fontsize=9)
ax3.set_xlabel('Coefficient', fontsize=10)
ax3.set_title('Top 10 Features', fontsize=12, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.grid(True, alpha=0.3, axis='x')

# 4. Model comparison
ax4 = axes[1, 1]
models = []
maes = []
r2s = []

if v1_2_file.exists():
    models.append('v1.2')
    maes.append(v1_2['test_mae'])
    r2s.append(v1_2['test_r2'])

if v1_3_file.exists():
    models.append('v1.3')
    maes.append(v1_3['test_mae'])
    r2s.append(v1_3['test_r2'])

models.append('v1.4')
maes.append(test_mae)
r2s.append(test_r2)

x_pos = np.arange(len(models))
width = 0.35

ax4_twin = ax4.twinx()
bars1 = ax4.bar(x_pos - width/2, maes, width, label='MAE', color='salmon', alpha=0.7)
bars2 = ax4_twin.bar(x_pos + width/2, r2s, width, label='R²', color='lightblue', alpha=0.7)

ax4.set_ylabel('MAE (points)', fontsize=10, color='salmon')
ax4_twin.set_ylabel('R²', fontsize=10, color='lightblue')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models, fontsize=10)
ax4.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='salmon')
ax4_twin.tick_params(axis='y', labelcolor='lightblue')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

plot_file = config.OUTPUT_DIR / 'ball_knower_v1_4_analysis.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  ✓ {plot_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("v1.4 SUMMARY")
print("="*80)

print(f"""
Enhanced Features ({len(enhanced_features)} total):
  Baseline (6): nfelo_diff, rest, div_game, surface, time, qb_diff
  EPA Rolling (6): 3-game and 5-game averages for off/def EPA
  Success Rolling (2): 3-game averages for off/def success rate
  Momentum (1): Recent trend indicator
  Interactions (1): EPA × ELO

Test Performance (2024):
  MAE:  {test_mae:.3f} points
  RMSE: {test_rmse:.3f} points
  R²:   {test_r2:.3f}

Best Alpha: {model.alpha_}

Top 3 Features:
  1. {feature_importance.iloc[0]['feature']}: {feature_importance.iloc[0]['coefficient']:.4f}
  2. {feature_importance.iloc[1]['feature']}: {feature_importance.iloc[1]['coefficient']:.4f}
  3. {feature_importance.iloc[2]['feature']}: {feature_importance.iloc[2]['coefficient']:.4f}
""")

print("="*80 + "\n")
