"""
ARCHIVE FILE — uses legacy loaders by design, do not modify

Calibrate Using Linear Regression

Since we have strong correlations (0.915 for nfelo, 0.886 for substack),
use linear regression to find optimal weights.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("CALIBRATION VIA LINEAR REGRESSION")
print("="*80)

# Load data
print("\n[1/3] Loading Week 11 2025 data...")
games = nflverse.games(season=2025, week=11)
team_ratings = data_loader.merge_current_week_ratings()

games = games[games['spread_line'].notna()].copy()
matchups = games[['away_team', 'home_team', 'spread_line']].copy()

# Merge ratings
matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']],
    left_on='home_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={
    'nfelo': 'home_nfelo',
    'epa_margin': 'home_epa',
    'Ovr.': 'home_substack'
})

matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']],
    left_on='away_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={
    'nfelo': 'away_nfelo',
    'epa_margin': 'away_epa',
    'Ovr.': 'away_substack'
})

matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups['epa_diff'] = matchups['home_epa'] - matchups['away_epa']
matchups['substack_diff'] = matchups['home_substack'] - matchups['away_substack']

matchups = matchups.dropna()
print(f"Complete data for {len(matchups)} games")

# Fit regression models
print("\n[2/3] Fitting regression models...")

y = matchups['spread_line'].values

# Model 1: nfelo only
X1 = matchups[['nfelo_diff']].values
model1 = LinearRegression(fit_intercept=True)
model1.fit(X1, y)

# Model 2: nfelo + epa
X2 = matchups[['nfelo_diff', 'epa_diff']].values
model2 = LinearRegression(fit_intercept=True)
model2.fit(X2, y)

# Model 3: nfelo + epa + substack
X3 = matchups[['nfelo_diff', 'epa_diff', 'substack_diff']].values
model3 = LinearRegression(fit_intercept=True)
model3.fit(X3, y)

# Model 4: substack only (for comparison)
X4 = matchups[['substack_diff']].values
model4 = LinearRegression(fit_intercept=True)
model4.fit(X4, y)

# Predictions and metrics
pred1 = model1.predict(X1)
pred2 = model2.predict(X2)
pred3 = model3.predict(X3)
pred4 = model4.predict(X4)

mae1 = np.mean(np.abs(pred1 - y))
mae2 = np.mean(np.abs(pred2 - y))
mae3 = np.mean(np.abs(pred3 - y))
mae4 = np.mean(np.abs(pred4 - y))

rmse1 = np.sqrt(np.mean((pred1 - y) ** 2))
rmse2 = np.sqrt(np.mean((pred2 - y) ** 2))
rmse3 = np.sqrt(np.mean((pred3 - y) ** 2))
rmse4 = np.sqrt(np.mean((pred4 - y) ** 2))

r2_1 = model1.score(X1, y)
r2_2 = model2.score(X2, y)
r2_3 = model3.score(X3, y)
r2_4 = model4.score(X4, y)

print("\n[3/3] Evaluating models...")

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

print(f"\nModel 1: nfelo only")
print(f"  Coefficient: {model1.coef_[0]:.6f}")
print(f"  Intercept (implied HFA): {model1.intercept_:.2f}")
print(f"  Conversion: {1/model1.coef_[0]:.1f} nfelo points ≈ 1 spread point")
print(f"  MAE: {mae1:.2f} | RMSE: {rmse1:.2f} | R²: {r2_1:.3f}")

print(f"\nModel 2: nfelo + epa")
print(f"  nfelo coef: {model2.coef_[0]:.6f}")
print(f"  epa coef: {model2.coef_[1]:.6f}")
print(f"  Intercept: {model2.intercept_:.2f}")
print(f"  MAE: {mae2:.2f} | RMSE: {rmse2:.2f} | R²: {r2_2:.3f}")

print(f"\nModel 3: nfelo + epa + substack")
print(f"  nfelo coef: {model3.coef_[0]:.6f}")
print(f"  epa coef: {model3.coef_[1]:.6f}")
print(f"  substack coef: {model3.coef_[2]:.6f}")
print(f"  Intercept: {model3.intercept_:.2f}")
print(f"  MAE: {mae3:.2f} | RMSE: {rmse3:.2f} | R²: {r2_3:.3f}")

print(f"\nModel 4: substack only (baseline)")
print(f"  Coefficient: {model4.coef_[0]:.6f}")
print(f"  Intercept: {model4.intercept_:.2f}")
print(f"  MAE: {mae4:.2f} | RMSE: {rmse4:.2f} | R²: {r2_4:.3f}")

# Select best model (Model 1 for simplicity and interpretability)
best_model = model1
nfelo_weight = -model1.coef_[0]  # Negative because spread = -nfelo_diff * weight
hfa = -model1.intercept_  # Negative because spread = -HFA for home favorite

matchups['predicted_spread'] = pred1
matchups['error'] = pred1 - y

print("\n" + "="*80)
print("SELECTED MODEL: nfelo only (simplest, most interpretable)")
print("="*80)

print(f"\nCalibrated Weights:")
print(f"  nfelo_weight: {nfelo_weight:.6f}")
print(f"  HFA (implied): {hfa:.2f}")
print(f"\nFormula: spread = {hfa:.2f} - (nfelo_diff × {nfelo_weight:.6f})")

print("\n" + "="*80)
print("PREDICTIONS VS VEGAS")
print("="*80)

results = matchups[[
    'away_team', 'home_team', 'spread_line', 'predicted_spread', 'error'
]].copy()

results = results.round(1)
results = results.sort_values('error', key=abs, ascending=False)

print("\n" + results.to_string(index=False))

# Save weights
weights_file = config.OUTPUT_DIR / 'calibrated_weights_regression.txt'
with open(weights_file, 'w') as f:
    f.write(f"# Ball Knower v1.0 - Regression Calibration\n")
    f.write(f"# Fitted to Week 11 2025 Vegas lines ({len(matchups)} games)\n")
    f.write(f"# MAE: {mae1:.2f} pts, RMSE: {rmse1:.2f} pts, R²: {r2_1:.3f}\n\n")
    f.write(f"nfelo_weight = {nfelo_weight:.6f}\n")
    f.write(f"hfa = {hfa:.2f}\n")
    f.write(f"\n# Multi-feature models (for reference):\n")
    f.write(f"# Model 2 (nfelo + epa): MAE={mae2:.2f}, R²={r2_2:.3f}\n")
    f.write(f"# Model 3 (nfelo + epa + substack): MAE={mae3:.2f}, R²={r2_3:.3f}\n")

print(f"\n\nWeights saved to: {weights_file}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print(f"""
Best Model Performance:
- R² = {r2_1:.3f} (explains {r2_1*100:.1f}% of variance in Vegas lines)
- MAE = {mae1:.2f} points
- RMSE = {rmse1:.2f} points

Conversion Factor:
- {1/nfelo_weight:.1f} nfelo points ≈ 1 spread point
- This {'matches' if abs(1/nfelo_weight - 40) < 10 else 'differs from'} the research baseline of 40:1

Model Quality: {"Excellent - ready for v1.0" if mae1 < 2.0 else "Good - usable baseline" if mae1 < 3.0 else "Needs improvement"}

Next Steps:
1. Update src/models.py with nfelo_weight = {nfelo_weight:.6f}
2. Re-run predictions with calibrated model
3. Add v1.1 contextual adjustments for remaining errors
""")

print("\n" + "="*80 + "\n")
