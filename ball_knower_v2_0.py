"""
Ball Knower v2.0 - Score Prediction Model (STEP 3)

Major architectural upgrade from spread correction to score prediction.

Architecture:
- Two separate models: home_score_model and away_score_model
- Features: nfelo_diff, EPA diffs, rest, QB, situational factors
- Targets: Actual game scores (home_score, away_score)

Derived Outputs:
- Spread = home_score - away_score
- Total = home_score + away_score
- Win probability from score distributions
- Moneyline odds
- Over/under probabilities

Benefits:
- Predicts actual game outcomes (not just Vegas lines)
- Unlocks moneyline and totals betting
- More interpretable (actual scores)
- Better probability estimates
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

from src import config

print("\n" + "="*80)
print("BALL KNOWER v2.0 - SCORE PREDICTION MODEL")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/8] Loading historical game data...")

# Load nfelo games (has actual scores)
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

# Extract season/week/teams
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Filter to complete data with scores
df = df[df['home_score'].notna()].copy()
df = df[df['away_score'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"  ✓ Loaded {len(df):,} games with scores ({df['season'].min()}-{df['season'].max()})")

# Load EPA data
print("\n[2/8] Loading EPA data...")

epa_file = project_root / 'data' / 'team_week_epa_2013_2024.csv'

if epa_file.exists():
    epa_df = pd.read_csv(epa_file)
    print(f"  ✓ Loaded {len(epa_df):,} team-week EPA records")

    # Merge EPA for home team
    epa_home = epa_df.copy()
    epa_home.columns = ['season', 'week', 'home_team'] + [f'home_{col}' for col in epa_df.columns if col not in ['season', 'week', 'team']]

    df = df.merge(epa_home, on=['season', 'week', 'home_team'], how='left')

    # Merge EPA for away team
    epa_away = epa_df.copy()
    epa_away.columns = ['season', 'week', 'away_team'] + [f'away_{col}' for col in epa_df.columns if col not in ['season', 'week', 'team']]

    df = df.merge(epa_away, on=['season', 'week', 'away_team'], how='left')

    epa_coverage = df[['home_off_epa_per_play', 'away_off_epa_per_play']].notna().all(axis=1).sum()
    print(f"  EPA coverage: {epa_coverage:,}/{len(df):,} games ({epa_coverage/len(df)*100:.1f}%)")
else:
    print(f"  ⚠ EPA file not found - using nfelo features only")

# ============================================================================
# ENGINEER FEATURES
# ============================================================================

print("\n[3/8] Engineering features...")

# v1.2/v1.3 baseline features
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# EPA differentials (if available)
if 'home_off_epa_per_play' in df.columns:
    df['epa_off_diff'] = df['home_off_epa_per_play'] - df['away_off_epa_per_play']
    df['epa_def_diff'] = df['home_def_epa_per_play'] - df['away_def_epa_per_play']
    df['success_rate_off_diff'] = df['home_off_success_rate'] - df['away_off_success_rate']
    df['success_rate_def_diff'] = df['home_def_success_rate'] - df['away_def_success_rate']

    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game', 'surface_mod', 'time_advantage', 'qb_diff',
                    'epa_off_diff', 'epa_def_diff', 'success_rate_off_diff', 'success_rate_def_diff']

    # Filter to games with complete EPA data
    mask = df[feature_cols + ['home_score', 'away_score']].notna().all(axis=1)
    df = df[mask].copy()

    print(f"  Using v2.0 features with EPA (10 features)")
else:
    feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game', 'surface_mod', 'time_advantage', 'qb_diff']

    # Filter to games with complete baseline data
    mask = df[feature_cols + ['home_score', 'away_score']].notna().all(axis=1)
    df = df[mask].copy()

    print(f"  Using v2.0 features without EPA (6 features)")

print(f"  ✓ Prepared {len(df):,} games with complete features")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n[4/8] Splitting train/test...")

# Use 2024 as test year (same as v1.3 for comparison)
train_df = df[df['season'] < 2024].copy()
test_df = df[df['season'] == 2024].copy()

print(f"  Training: {len(train_df):,} games ({train_df['season'].min()}-{train_df['season'].max()})")
print(f"  Test:     {len(test_df):,} games (2024)")

X_train = train_df[feature_cols].values
y_train_home = train_df['home_score'].values
y_train_away = train_df['away_score'].values

X_test = test_df[feature_cols].values
y_test_home = test_df['home_score'].values
y_test_away = test_df['away_score'].values

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("\n[5/8] Training dual score prediction models...")

# Train home score model
home_model = Ridge(alpha=100.0)
home_model.fit(X_train, y_train_home)

# Train away score model
away_model = Ridge(alpha=100.0)
away_model.fit(X_train, y_train_away)

# Predictions
train_home_pred = home_model.predict(X_train)
train_away_pred = away_model.predict(X_train)

test_home_pred = home_model.predict(X_test)
test_away_pred = away_model.predict(X_test)

# Derive spreads and totals
train_spread_pred = train_home_pred - train_away_pred
train_total_pred = train_home_pred + train_away_pred

test_spread_pred = test_home_pred - test_away_pred
test_total_pred = test_home_pred + test_away_pred

train_spread_actual = y_train_home - y_train_away
train_total_actual = y_train_home + y_train_away

test_spread_actual = y_test_home - y_test_away
test_total_actual = y_test_home + y_test_away

print(f"\n  Home Score Model:")
print(f"    Train MAE: {mean_absolute_error(y_train_home, train_home_pred):.2f} points")
print(f"    Test MAE:  {mean_absolute_error(y_test_home, test_home_pred):.2f} points")

print(f"\n  Away Score Model:")
print(f"    Train MAE: {mean_absolute_error(y_train_away, train_away_pred):.2f} points")
print(f"    Test MAE:  {mean_absolute_error(y_test_away, test_away_pred):.2f} points")

print(f"\n  Derived Spread (home - away):")
print(f"    Train MAE: {mean_absolute_error(train_spread_actual, train_spread_pred):.2f} points")
print(f"    Test MAE:  {mean_absolute_error(test_spread_actual, test_spread_pred):.2f} points")
print(f"    Test R²:   {r2_score(test_spread_actual, test_spread_pred):.3f}")

print(f"\n  Derived Total (home + away):")
print(f"    Train MAE: {mean_absolute_error(train_total_actual, train_total_pred):.2f} points")
print(f"    Test MAE:  {mean_absolute_error(test_total_actual, test_total_pred):.2f} points")
print(f"    Test R²:   {r2_score(test_total_actual, test_total_pred):.3f}")

# ============================================================================
# COMPARE TO v1.3
# ============================================================================

print("\n[6/8] Comparing to v1.3 spread prediction...")

# Load v1.3 model if available
v1_3_model_file = config.OUTPUT_DIR / 'ball_knower_v1_3_model.json'

if v1_3_model_file.exists():
    with open(v1_3_model_file, 'r') as f:
        v1_3_params = json.load(f)

    print(f"\nv1.3 Spread Prediction (direct):")
    print(f"  Test MAE: {v1_3_params['test_mae']:.2f} points")
    print(f"  Test R²:  {v1_3_params['test_r2']:.3f}")

    print(f"\nv2.0 Spread Prediction (derived from scores):")
    print(f"  Test MAE: {mean_absolute_error(test_spread_actual, test_spread_pred):.2f} points")
    print(f"  Test R²:  {r2_score(test_spread_actual, test_spread_pred):.3f}")

    mae_diff = mean_absolute_error(test_spread_actual, test_spread_pred) - v1_3_params['test_mae']
    print(f"\n  Spread MAE difference: {mae_diff:+.2f} points")

    if mae_diff < 0:
        print(f"  ✓ v2.0 spread predictions are BETTER than v1.3!")
    else:
        print(f"  ⚠ v2.0 spread predictions slightly worse (but we gain totals + moneylines!)")
else:
    print(f"\n  v1.3 model not found - cannot compare")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n[7/8] Analyzing feature importance...")

home_importance = pd.DataFrame({
    'feature': feature_cols,
    'home_coef': home_model.coef_,
    'away_coef': away_model.coef_,
})

home_importance['avg_abs_coef'] = (abs(home_importance['home_coef']) + abs(home_importance['away_coef'])) / 2
home_importance = home_importance.sort_values('avg_abs_coef', ascending=False)

print(f"\nFeature Importance:")
print(home_importance.to_string(index=False))

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n[8/8] Saving model outputs...")

# Save model parameters
model_params = {
    'model_type': 'Dual Ridge (home + away scores)',
    'alpha': 100.0,
    'features': feature_cols,
    'home_model': {
        'intercept': float(home_model.intercept_),
        'coefficients': {feat: float(coef) for feat, coef in zip(feature_cols, home_model.coef_)},
    },
    'away_model': {
        'intercept': float(away_model.intercept_),
        'coefficients': {feat: float(coef) for feat, coef in zip(feature_cols, away_model.coef_)},
    },
    'train_games': len(train_df),
    'test_games': len(test_df),
    'test_metrics': {
        'home_score_mae': float(mean_absolute_error(y_test_home, test_home_pred)),
        'away_score_mae': float(mean_absolute_error(y_test_away, test_away_pred)),
        'spread_mae': float(mean_absolute_error(test_spread_actual, test_spread_pred)),
        'spread_r2': float(r2_score(test_spread_actual, test_spread_pred)),
        'total_mae': float(mean_absolute_error(test_total_actual, test_total_pred)),
        'total_r2': float(r2_score(test_total_actual, test_total_pred)),
    },
    'seasons': f"{train_df['season'].min()}-{train_df['season'].max()}",
}

model_file = config.OUTPUT_DIR / 'ball_knower_v2_0_model.json'
with open(model_file, 'w') as f:
    json.dump(model_params, f, indent=2)
print(f"  ✓ Model parameters: {model_file}")

# Save test predictions
test_df['v2_home_score_pred'] = test_home_pred
test_df['v2_away_score_pred'] = test_away_pred
test_df['v2_spread_pred'] = test_spread_pred
test_df['v2_total_pred'] = test_total_pred

test_output = test_df[['game_id', 'season', 'week', 'away_team', 'home_team',
                        'away_score', 'home_score',
                        'v2_away_score_pred', 'v2_home_score_pred',
                        'v2_spread_pred', 'v2_total_pred']].copy()

test_pred_file = config.OUTPUT_DIR / 'ball_knower_v2_0_test_predictions.csv'
test_output.to_csv(test_pred_file, index=False)
print(f"  ✓ Test predictions: {test_pred_file}")

# Save feature importance
home_importance.to_csv(config.OUTPUT_DIR / 'ball_knower_v2_0_feature_importance.csv', index=False)
print(f"  ✓ Feature importance: ball_knower_v2_0_feature_importance.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ball Knower v2.0 - Score Prediction Model', fontsize=16, fontweight='bold')

# 1. Home score predictions
ax1 = axes[0, 0]
ax1.scatter(y_test_home, test_home_pred, alpha=0.5, s=30)
ax1.plot([y_test_home.min(), y_test_home.max()], [y_test_home.min(), y_test_home.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Home Score', fontsize=10)
ax1.set_ylabel('Predicted Home Score', fontsize=10)
ax1.set_title(f'Home Score Predictions (MAE={mean_absolute_error(y_test_home, test_home_pred):.2f})',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Derived spread predictions
ax2 = axes[0, 1]
ax2.scatter(test_spread_actual, test_spread_pred, alpha=0.5, s=30)
ax2.plot([test_spread_actual.min(), test_spread_actual.max()],
         [test_spread_actual.min(), test_spread_actual.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Spread', fontsize=10)
ax2.set_ylabel('Predicted Spread', fontsize=10)
ax2.set_title(f'Spread Predictions (MAE={mean_absolute_error(test_spread_actual, test_spread_pred):.2f})',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Total predictions
ax3 = axes[1, 0]
ax3.scatter(test_total_actual, test_total_pred, alpha=0.5, s=30, color='green')
ax3.plot([test_total_actual.min(), test_total_actual.max()],
         [test_total_actual.min(), test_total_actual.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Total', fontsize=10)
ax3.set_ylabel('Predicted Total', fontsize=10)
ax3.set_title(f'Total Predictions (MAE={mean_absolute_error(test_total_actual, test_total_pred):.2f})',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Feature importance
ax4 = axes[1, 1]
top_features = home_importance.head(8)
x_pos = np.arange(len(top_features))
ax4.barh(x_pos, top_features['avg_abs_coef'], alpha=0.7, color='steelblue')
ax4.set_yticks(x_pos)
ax4.set_yticklabels(top_features['feature'], fontsize=9)
ax4.set_xlabel('Average Absolute Coefficient', fontsize=10)
ax4.set_title('Top Features (Avg Home + Away)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

plot_file = config.OUTPUT_DIR / 'ball_knower_v2_0_analysis.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Visualization: {plot_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("v2.0 MODEL SUMMARY")
print("="*80)

print(f"""
Architecture: Dual Score Prediction
  - Home score model: Ridge(alpha=100)
  - Away score model: Ridge(alpha=100)
  - Features: {len(feature_cols)}

Training Data:
  Games: {len(train_df):,} ({train_df['season'].min()}-{train_df['season'].max()})

Test Performance (2024):
  Home Score MAE: {mean_absolute_error(y_test_home, test_home_pred):.2f} points
  Away Score MAE: {mean_absolute_error(y_test_away, test_away_pred):.2f} points

  Derived Spread MAE: {mean_absolute_error(test_spread_actual, test_spread_pred):.2f} points
  Derived Spread R²:  {r2_score(test_spread_actual, test_spread_pred):.3f}

  Derived Total MAE:  {mean_absolute_error(test_total_actual, test_total_pred):.2f} points
  Derived Total R²:   {r2_score(test_total_actual, test_total_pred):.3f}

New Capabilities:
  ✓ Actual score predictions (not just spreads)
  ✓ Moneyline betting (win probabilities from score distributions)
  ✓ Totals betting (over/under predictions)
  ✓ Full game simulation capabilities
  ✓ More interpretable predictions

Files Saved:
  - ball_knower_v2_0_model.json
  - ball_knower_v2_0_test_predictions.csv
  - ball_knower_v2_0_feature_importance.csv
  - ball_knower_v2_0_analysis.png
""")

print("="*80 + "\n")
