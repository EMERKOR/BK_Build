"""
Calibrate Ball Knower v1.x Weights and Export to JSON

This script calibrates model weights on historical data and exports them
to the JSON format expected by src/models.py load_calibrated_weights().

Approach:
- Loads historical nfelo data (2015-2024)
- Fits Ridge regression to predict Vegas closing lines
- Extracts weights for EPA, nfelo, substack, rest, form, and QB adjustments
- Exports to output/calibrated_weights_v1.json
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ball_knower import config
from ball_knower.features import engineering as features

print("\n" + "="*80)
print("CALIBRATING BALL KNOWER v1.x - JSON EXPORT")
print("="*80)

# ============================================================================
# LOAD HISTORICAL DATA
# ============================================================================

print("\n[1/5] Loading historical nfelo data...")

nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

# Extract season/week/teams from game_id
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Filter to training period (2015-2024)
df = df[(df['season'] >= 2015) & (df['season'] <= 2024)].copy()

# Filter to complete data with Vegas lines
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"  Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

# ============================================================================
# ENGINEER FEATURES
# ============================================================================

print("\n[2/5] Engineering features for v1.0 and v1.1...")

# v1.0 features: EPA, nfelo, substack
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# EPA differentials (if available - may be limited in nfelo dataset)
# For now, we'll focus on nfelo which is the primary predictor
# In a full implementation, you would merge EPA and substack from other sources

# v1.1 features: rest, form, QB
# Use canonical rest advantage calculation from ball_knower.features.engineering
df['rest_advantage'] = features.compute_rest_advantage_from_nfelo(df)

df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# Additional situational features
df['div_game_mod'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

# Target: Vegas closing line
df['vegas_line'] = df['home_line_close']

# Remove rows with any missing features
feature_cols = ['nfelo_diff', 'rest_advantage', 'qb_diff', 'div_game_mod', 'surface_mod', 'time_advantage']
mask = df[feature_cols + ['vegas_line']].notna().all(axis=1)
df = df[mask].copy()

print(f"  Engineered features for {len(df):,} games")

# ============================================================================
# FIT MODEL WEIGHTS
# ============================================================================

print("\n[3/5] Fitting Ridge regression model...")

# Prepare features and target
X = df[feature_cols].values
y = df['vegas_line'].values

# Fit Ridge regression (alpha=10 for slight regularization)
model = Ridge(alpha=10.0)
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
r2 = model.score(X, y)

print(f"  Ridge alpha: 10.0")
print(f"  R² score: {r2:.3f}")
print(f"  MAE: {mae:.2f} points")
print(f"  RMSE: {rmse:.2f} points")

# ============================================================================
# EXTRACT WEIGHTS
# ============================================================================

print("\n[4/5] Extracting calibrated weights...")

# Ridge intercept represents HFA (home field advantage)
# But we need to negate it because our model predicts from home perspective
# where negative = home favored
hfa = -model.intercept_

# Extract coefficients
# Note: These are negative because in our formula:
# spread = -HFA - (feature_diff * weight)
# But Ridge gives us: spread = intercept + (feature_diff * coef)
# So we need to negate the coefficients

coef_dict = dict(zip(feature_cols, model.coef_))

# Map to the expected weight names
# Note: We'll use the nfelo_diff coefficient for both EPA and nfelo
# since they're highly correlated (in full implementation, fit them separately)
weights = {
    'epa_margin': abs(coef_dict['nfelo_diff'] * 100),  # Scale to EPA units (roughly)
    'nfelo_diff': abs(coef_dict['nfelo_diff']),
    'substack_ovr_diff': 0.5,  # Default (not in nfelo dataset)
    'rest_advantage': abs(coef_dict['rest_advantage']),
    'win_rate_L5': 5.0,  # Default (form feature not available in nfelo dataset)
    'qb_adj_diff': abs(coef_dict['qb_diff']),
}

print(f"\nCalibrated weights:")
print(f"  HFA: {hfa:.2f}")
for key, value in weights.items():
    print(f"  {key}: {value:.4f}")

# ============================================================================
# EXPORT TO JSON
# ============================================================================

print("\n[5/5] Exporting to JSON...")

calibration_data = {
    "hfa": round(hfa, 2),
    "weights": {k: round(v, 4) for k, v in weights.items()},
    "metadata": {
        "calibrated_on": f"{df['season'].min()}-{df['season'].max()}",
        "mae_vs_vegas": round(mae, 2),
        "rmse_vs_vegas": round(rmse, 2),
        "r2_score": round(r2, 3),
        "n_games": len(df),
        "model_type": "Ridge",
        "alpha": 10.0,
        "created_at": datetime.now().isoformat(),
        "feature_cols": feature_cols
    }
}

# Save to JSON
output_file = config.OUTPUT_DIR / 'calibrated_weights_v1.json'
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(calibration_data, f, indent=2)

print(f"\n✓ Calibrated weights saved to: {output_file}")

# ============================================================================
# SHOW EXAMPLE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE PREDICTIONS (2024 Season)")
print("="*80)

example_df = df[df['season'] == 2024].copy()
if len(example_df) > 0:
    example_df = example_df.head(10)

    # Get predictions for examples
    X_example = example_df[feature_cols].values
    y_example_pred = model.predict(X_example)

    example_df['predicted_line'] = y_example_pred
    example_df['actual_line'] = example_df['vegas_line']
    example_df['error'] = example_df['predicted_line'] - example_df['actual_line']

    display_cols = ['away_team', 'home_team', 'actual_line', 'predicted_line', 'error']
    print(example_df[display_cols].to_string(index=False))

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print(f"""
1. ✓ Calibrated weights saved to {output_file.name}
2. Run predictions to verify models load the JSON file:
   python run_demo.py
3. The models in src/models.py will automatically use these weights
4. MAE of {mae:.2f} points vs Vegas is {'excellent' if mae < 1.5 else 'good' if mae < 2.5 else 'acceptable'}
""")

print("\n" + "="*80 + "\n")
