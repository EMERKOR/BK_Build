"""
ARCHIVE FILE — uses legacy loaders by design, do not modify

Calibrate Ball Knower Model to Current Vegas Lines

Simpler approach: Fit model weights to minimize difference from current Vegas lines.
This aligns with the principle: "Model the market bias, not the game."
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("CALIBRATING BALL KNOWER TO CURRENT VEGAS LINES")
print("="*80)

# Step 1: Load current week data
print("\n[1/3] Loading Week 11 2025 data...")
games = nflverse.games(season=2025, week=11)
team_ratings = data_loader.merge_current_week_ratings()

# Filter games with Vegas lines
games = games[games['spread_line'].notna()].copy()
print(f"Loaded {len(games)} games with Vegas lines")

# Build matchups
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

# Calculate differentials
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups['epa_diff'] = matchups['home_epa'] - matchups['away_epa']
matchups['substack_diff'] = matchups['home_substack'] - matchups['away_substack']

# Remove any rows with missing data
matchups = matchups.dropna()
print(f"Complete data for {len(matchups)} games")

# Step 2: Fit optimal weights
print("\n[2/3] Optimizing weights to minimize error vs Vegas...")

HFA = config.HOME_FIELD_ADVANTAGE

def predict_spread(params):
    """Model: spread = -HFA - (nfelo_diff * w1) - (epa_diff * w2) - (substack_diff * w3)"""
    nfelo_weight, epa_weight, substack_weight = params

    predicted = (
        -HFA
        - (matchups['nfelo_diff'] * nfelo_weight)
        - (matchups['epa_diff'] * epa_weight)
        - (matchups['substack_diff'] * substack_weight)
    )

    return predicted

def loss_function(params):
    """Mean squared error vs Vegas lines"""
    predicted = predict_spread(params)
    actual = matchups['spread_line'].values
    mse = np.mean((predicted - actual) ** 2)
    return mse

# Initial guess (from research and previous attempts)
x0 = [0.025, 0.01, 0.3]  # nfelo, epa, substack weights

# Bounds to keep weights reasonable
bounds = [
    (0.0, 0.1),   # nfelo: 10-100 points = 1 spread point
    (0.0, 5.0),   # epa: 0-5x multiplier
    (0.0, 1.0)    # substack: 0-1x (already in point scale)
]

# Optimize
result = minimize(loss_function, x0, method='L-BFGS-B', bounds=bounds)

optimal_weights = result.x
nfelo_weight, epa_weight, substack_weight = optimal_weights

# Calculate final predictions
matchups['predicted_spread'] = predict_spread(optimal_weights)
matchups['error'] = matchups['predicted_spread'] - matchups['spread_line']

# Step 3: Evaluate results
print("\n[3/3] Evaluating calibrated model...")

mae = np.mean(np.abs(matchups['error']))
rmse = np.sqrt(np.mean(matchups['error'] ** 2))

print("\n" + "="*80)
print("CALIBRATION RESULTS")
print("="*80)

print(f"\nOptimal Weights:")
print(f"  nfelo_weight:     {nfelo_weight:.4f}  (1 spread pt ≈ {1/nfelo_weight:.1f} nfelo pts)")
print(f"  epa_weight:       {epa_weight:.4f}")
print(f"  substack_weight:  {substack_weight:.4f}")
print(f"  HFA:              {HFA}")

print(f"\nModel Performance vs Vegas:")
print(f"  Mean Absolute Error: {mae:.2f} points")
print(f"  Root Mean Squared Error: {rmse:.2f} points")
print(f"  Games analyzed: {len(matchups)}")

print("\n" + "="*80)
print("PREDICTIONS VS VEGAS (Current Week 11)")
print("="*80)

results = matchups[[
    'away_team', 'home_team', 'spread_line', 'predicted_spread', 'error'
]].copy()

results = results.round(1)
results = results.sort_values('error', key=abs, ascending=False)

print("\n" + results.to_string(index=False))

# Save calibrated weights
print("\n" + "="*80)
print("SAVING CALIBRATED WEIGHTS")
print("="*80)

weights_file = config.OUTPUT_DIR / 'calibrated_weights_v1.txt'
with open(weights_file, 'w') as f:
    f.write(f"# Ball Knower v1.0 Calibrated Weights\n")
    f.write(f"# Fitted to Week 11 2025 Vegas lines ({len(matchups)} games)\n")
    f.write(f"# MAE: {mae:.2f} pts, RMSE: {rmse:.2f} pts\n\n")
    f.write(f"nfelo_weight = {nfelo_weight:.6f}\n")
    f.write(f"epa_weight = {epa_weight:.6f}\n")
    f.write(f"substack_weight = {substack_weight:.6f}\n")
    f.write(f"hfa = {HFA}\n")

print(f"\nWeights saved to: {weights_file}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print(f"""
These weights were optimized to match Vegas lines as closely as possible.

Key insights:
- nfelo conversion: {1/nfelo_weight:.1f} nfelo points ≈ 1 spread point
- EPA contribution: {'Significant' if epa_weight > 0.5 else 'Moderate' if epa_weight > 0.1 else 'Small'}
- Substack weight: {'High' if substack_weight > 0.5 else 'Moderate' if substack_weight > 0.2 else 'Low'}

Average error of {mae:.2f} points means:
- Our baseline model {'matches Vegas very well' if mae < 1.0 else 'is reasonably calibrated to Vegas' if mae < 2.0 else 'needs more data or features'}
- We can now use this as the v1.0 baseline
- v1.1 adjustments should target the specific games with largest errors

NOTE: This calibration uses only Week 11 2025 data.
For production use, re-calibrate on multiple weeks or use historical data if available.
""")

print("\n" + "="*80 + "\n")
