"""
ARCHIVE FILE — uses legacy loaders by design, do not modify

Calibrate Ball Knower v1.0 Model Weights

Fits model weights on historical data (2015-2024) to match Vegas lines.
This will produce realistic spread predictions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ball_knower import config, data_loader

print("\n" + "="*80)
print("CALIBRATING BALL KNOWER v1.0 MODEL")
print("="*80)

# Step 1: Load historical schedules
print("\n[1/5] Loading historical schedules (2015-2024)...")
print("This may take 1-2 minutes on first run (downloads from nfl_data_py)...")

schedules = data_loader.load_historical_schedules(2015, 2024)

# Filter for games with Vegas lines
schedules = schedules[schedules['spread_line'].notna()].copy()

# Calculate actual margin (home score - away score)
schedules['actual_margin'] = schedules['home_score'] - schedules['away_score']

print(f"Loaded {len(schedules):,} games with Vegas lines")

# Step 2: Load historical weekly stats for EPA
print("\n[2/5] Loading historical EPA data...")
weekly_stats = data_loader.load_historical_team_stats(2015, 2024, stat_type='weekly')

# Calculate team-level EPA by season
print("\n[3/5] Calculating team EPA by season...")

team_epa_by_season = []

for season in range(2015, 2025):
    season_stats = weekly_stats[weekly_stats['season'] == season].copy()

    # Group by team and calculate season averages
    team_season_epa = season_stats.groupby('recent_team').agg({
        'offense_epa': 'mean',
        'defense_epa': 'mean'
    }).reset_index()

    team_season_epa['season'] = season
    team_season_epa['epa_margin'] = team_season_epa['offense_epa'] - team_season_epa['defense_epa']
    team_season_epa.rename(columns={'recent_team': 'team'}, inplace=True)

    team_epa_by_season.append(team_season_epa)

team_epa_df = pd.concat(team_epa_by_season, ignore_index=True)

print(f"Calculated EPA for {len(team_epa_df)} team-seasons")

# Step 4: Merge EPA into schedules
print("\n[4/5] Merging EPA data into schedules...")

schedules = schedules.merge(
    team_epa_df[['team', 'season', 'epa_margin']],
    left_on=['team_home', 'season'],
    right_on=['team', 'season'],
    how='left'
).drop(columns=['team']).rename(columns={'epa_margin': 'home_epa_margin'})

schedules = schedules.merge(
    team_epa_df[['team', 'season', 'epa_margin']],
    left_on=['team_away', 'season'],
    right_on=['team', 'season'],
    how='left'
).drop(columns=['team']).rename(columns={'epa_margin': 'away_epa_margin'})

# Calculate EPA differential
schedules['epa_diff'] = schedules['home_epa_margin'] - schedules['away_epa_margin']

# Remove games with missing EPA
schedules = schedules[schedules['epa_diff'].notna()].copy()

print(f"Merged EPA for {len(schedules):,} games")

# Step 5: Fit model weights
print("\n[5/5] Fitting model weights to minimize error vs Vegas lines...")

def predict_spread(epa_diff, hfa, epa_weight):
    """Simple model: spread = -HFA - (epa_diff * weight)"""
    return -hfa - (epa_diff * epa_weight)

def loss_function(params):
    """Mean squared error vs Vegas lines"""
    epa_weight = params[0]

    predicted_spreads = predict_spread(
        schedules['epa_diff'].values,
        config.HOME_FIELD_ADVANTAGE,
        epa_weight
    )

    actual_spreads = schedules['spread_line'].values

    mse = np.mean((predicted_spreads - actual_spreads) ** 2)
    return mse

# Initial guess
x0 = [35.0]  # EPA weight

# Optimize
result = minimize(loss_function, x0, method='Nelder-Mead')

optimal_epa_weight = result.x[0]

# Calculate final predictions and metrics
schedules['predicted_spread'] = predict_spread(
    schedules['epa_diff'],
    config.HOME_FIELD_ADVANTAGE,
    optimal_epa_weight
)

mae = np.mean(np.abs(schedules['predicted_spread'] - schedules['spread_line']))
rmse = np.sqrt(np.mean((schedules['predicted_spread'] - schedules['spread_line']) ** 2))

# Results
print("\n" + "="*80)
print("CALIBRATION RESULTS")
print("="*80)
print(f"\nOptimal EPA Weight: {optimal_epa_weight:.2f}")
print(f"Home Field Advantage: {config.HOME_FIELD_ADVANTAGE}")
print(f"\nModel Performance vs Vegas Lines:")
print(f"  Mean Absolute Error: {mae:.2f} points")
print(f"  Root Mean Squared Error: {rmse:.2f} points")
print(f"  Games analyzed: {len(schedules):,}")
print(f"  Seasons: 2015-2024")

# Show example predictions
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS (2024 Season)")
print("="*80)

example_games = schedules[schedules['season'] == 2024].head(10)[
    ['team_away', 'team_home', 'spread_line', 'predicted_spread', 'actual_margin']
].copy()

example_games['error'] = example_games['predicted_spread'] - example_games['spread_line']

print(example_games.to_string(index=False))

# Save calibrated weight
print("\n" + "="*80)
print("SAVING CALIBRATED WEIGHTS")
print("="*80)

calibration_file = config.OUTPUT_DIR / 'calibrated_weights.txt'
with open(calibration_file, 'w') as f:
    f.write(f"# Ball Knower v1.0 Calibrated Weights\n")
    f.write(f"# Fitted on 2015-2024 NFL data ({len(schedules):,} games)\n")
    f.write(f"# MAE vs Vegas: {mae:.2f} pts, RMSE: {rmse:.2f} pts\n\n")
    f.write(f"epa_weight = {optimal_epa_weight:.4f}\n")
    f.write(f"hfa = {config.HOME_FIELD_ADVANTAGE}\n")

print(f"\nCalibrated weights saved to: {calibration_file}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Update src/models.py DeterministicSpreadModel weights:")
print(f"   Change 'epa_margin': 100 to 'epa_margin': {optimal_epa_weight:.2f}")
print("\n2. Re-run predictions: python run_demo.py")
print("\n3. Predictions should now match Vegas lines within ~10 points on average")
print("\n" + "="*80 + "\n")
