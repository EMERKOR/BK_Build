"""
Ball Knower v1.0 - Production Model
Calibrated to Week 11 2025 Vegas Lines

Model Performance:
- R² = 0.836 (explains 83.6% of Vegas line variance)
- MAE = 1.95 points
- RMSE = 2.28 points

Formula: spread = -2.67 + (nfelo_diff × 0.0447)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# CALIBRATED WEIGHTS (from calibrate_regression.py)
NFELO_COEF = 0.044703
INTERCEPT = 2.67

print("\n" + "="*80)
print("BALL KNOWER v1.0 - PRODUCTION MODEL")
print("="*80)
print(f"\nModel: spread = {INTERCEPT} + ({NFELO_COEF} × nfelo_diff)")
print(f"Calibration: R²=0.836, MAE=1.95pts on Week 11 2025 Vegas lines")

# Load current week data
print("\n[1/3] Loading Week 11 2025 data...")
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

print(f"Loaded {len(matchups)} games")

# Generate predictions
print("\n[2/3] Generating predictions...")

matchups['bk_v1_spread'] = INTERCEPT + (matchups['nfelo_diff'] * NFELO_COEF)
matchups['edge'] = matchups['bk_v1_spread'] - matchups['spread_line']
matchups['abs_edge'] = matchups['edge'].abs()

# Results
print("\n[3/3] Analysis complete")

print("\n" + "="*80)
print("WEEK 11 PREDICTIONS")
print("="*80)

results = matchups[[
    'away_team', 'home_team', 'spread_line', 'bk_v1_spread', 'edge', 'abs_edge'
]].copy()

results = results.sort_values('abs_edge', ascending=False)
results = results.drop(columns=['abs_edge']).round(1)

print("\n" + results.to_string(index=False))

# Summary statistics
mae = matchups['abs_edge'].mean()
rmse = np.sqrt((matchups['edge'] ** 2).mean())

print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)

print(f"\nAverage Absolute Edge: {mae:.2f} points")
print(f"RMSE: {rmse:.2f} points")
print(f"Max Edge: {matchups['abs_edge'].max():.2f} points")

# Value bets
print("\n" + "="*80)
print("VALUE BETS")
print("="*80)

value_threshold = 2.0  # 2+ point edge
value_bets = matchups[matchups['abs_edge'] >= value_threshold].copy()

print(f"\nGames with {value_threshold}+ point edge: {len(value_bets)}")

if len(value_bets) > 0:
    value_bets['recommendation'] = value_bets['edge'].apply(
        lambda x: f"Bet HOME (edge: {x:.1f})" if x < 0 else f"Bet AWAY (edge: +{x:.1f})"
    )

    value_results = value_bets[[
        'away_team', 'home_team', 'spread_line', 'bk_v1_spread', 'edge', 'abs_edge', 'recommendation'
    ]].copy()

    value_results = value_results.sort_values('abs_edge', ascending=False)
    value_results = value_results.drop(columns=['abs_edge']).round(1)

    print("\n" + value_results.to_string(index=False))
else:
    print("\nNo value bets at this threshold.")
    print(f"Largest edge: {matchups['abs_edge'].max():.2f} points")

# Save predictions
output_file = config.OUTPUT_DIR / 'week_11_predictions_v1.csv'
matchups.to_csv(output_file, index=False)
print(f"\n\nPredictions saved to: {output_file}")

print("\n" + "="*80)
print("USAGE NOTES")
print("="*80)

print("""
Ball Knower v1.0 is a baseline model using only nfelo power ratings.

Strengths:
- Simple, interpretable formula
- Calibrated to current Vegas market (R² = 0.836)
- Stable predictions based on season-long team quality

Limitations:
- Doesn't account for recent form, injuries, weather, rest
- Single power rating source (nfelo)
- Calibrated on small sample (14 Week 11 games)

Next Steps for v1.1:
- Add recent form adjustments (L5 record, trend)
- Incorporate situational factors (rest days, division games)
- Use multi-model ensemble (nfelo + Substack + EPA)
- Expand calibration dataset (more weeks/seasons)

For v1.2:
- Add ML correction layer to model systematic errors
- Integrate injury data, weather, coaching factors
- Build confidence intervals and bet sizing recommendations
""")

print("\n" + "="*80 + "\n")
