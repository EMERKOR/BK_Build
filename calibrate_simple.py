"""
ARCHIVE FILE — uses legacy loaders by design, do not modify

Simple Calibration - Fit nfelo Weight Only

With only 14 games in Week 11, we can't fit 3 parameters.
This script fits just the nfelo weight to minimize error vs Vegas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("SIMPLE CALIBRATION - NFELO WEIGHT ONLY")
print("="*80)

# Load data
print("\n[1/2] Loading Week 11 2025 data...")
games = nflverse.games(season=2025, week=11)
team_ratings = data_loader.merge_current_week_ratings()

games = games[games['spread_line'].notna()].copy()
matchups = games[['away_team', 'home_team', 'spread_line']].copy()

# Merge nfelo ratings
matchups = matchups.merge(
    team_ratings[['team', 'nfelo']],
    left_on='home_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={'nfelo': 'home_nfelo'})

matchups = matchups.merge(
    team_ratings[['team', 'nfelo']],
    left_on='away_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={'nfelo': 'away_nfelo'})

matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups = matchups.dropna()

print(f"Complete data for {len(matchups)} games")

# Fit optimal nfelo weight
print("\n[2/2] Optimizing nfelo weight...")

HFA = config.HOME_FIELD_ADVANTAGE

def loss_function(nfelo_weight):
    """Mean squared error vs Vegas lines"""
    predicted = -HFA - (matchups['nfelo_diff'] * nfelo_weight)
    actual = matchups['spread_line'].values
    mse = np.mean((predicted - actual) ** 2)
    return mse

# Optimize with bounds
result = minimize_scalar(loss_function, bounds=(0.0, 0.1), method='bounded')

optimal_nfelo_weight = result.x

# Calculate predictions
matchups['predicted_spread'] = -HFA - (matchups['nfelo_diff'] * optimal_nfelo_weight)
matchups['error'] = matchups['predicted_spread'] - matchups['spread_line']

mae = np.mean(np.abs(matchups['error']))
rmse = np.sqrt(np.mean(matchups['error'] ** 2))

print("\n" + "="*80)
print("CALIBRATION RESULTS")
print("="*80)

print(f"\nOptimal nfelo Weight: {optimal_nfelo_weight:.6f}")
print(f"Conversion: {1/optimal_nfelo_weight:.1f} nfelo points ≈ 1 spread point")
print(f"Home Field Advantage: {HFA}")

print(f"\nModel Performance vs Vegas:")
print(f"  Mean Absolute Error: {mae:.2f} points")
print(f"  Root Mean Squared Error: {rmse:.2f} points")
print(f"  Games analyzed: {len(matchups)}")

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
weights_file = config.OUTPUT_DIR / 'calibrated_weights_simple.txt'
with open(weights_file, 'w') as f:
    f.write(f"# Ball Knower v1.0 Simple Calibration\n")
    f.write(f"# Fitted to Week 11 2025 Vegas lines ({len(matchups)} games)\n")
    f.write(f"# MAE: {mae:.2f} pts, RMSE: {rmse:.2f} pts\n\n")
    f.write(f"nfelo_weight = {optimal_nfelo_weight:.6f}\n")
    f.write(f"hfa = {HFA}\n")

print(f"\n\nWeights saved to: {weights_file}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print(f"""
Calibrated Model: spread = -{HFA} - (nfelo_diff × {optimal_nfelo_weight:.4f})

Comparison to research baseline (40 nfelo pts = 1 spread pt):
  Research:  0.0250 weight (40:1 ratio)
  Calibrated: {optimal_nfelo_weight:.4f} weight ({1/optimal_nfelo_weight:.1f}:1 ratio)

Average error of {mae:.2f} points is {"excellent (< 1pt)" if mae < 1.0 else "good (1-2pts)" if mae < 2.0 else "acceptable (2-3pts)" if mae < 3.0 else "high, needs more features"}

Next steps:
1. Use this weight as v1.0 baseline
2. Add form/contextual adjustments in v1.1
3. Consider adding EPA/Substack as correction factors if errors remain systematic
""")

print("\n" + "="*80 + "\n")
