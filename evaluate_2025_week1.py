"""
Evaluate Ball Knower Production Model on 2025 Week 1 Games

Shows predictions vs actual results for the first 10 games of 2025.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("\n" + "="*80)
print("BALL KNOWER - 2025 WEEK 1 EVALUATION")
print("="*80)

# Load production model
output_dir = Path('/home/user/BK_Build/output')

with open(output_dir / 'ball_knower_production_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("\n✓ Loaded production model")

# Load 2025 predictions
predictions = pd.read_csv(output_dir / 'ball_knower_production_2025_predictions.csv')

# Filter to first 10 games (should be Week 1)
first_10 = predictions.head(10).copy()

print(f"\n{'='*80}")
print(f"FIRST 10 GAMES OF 2025 SEASON")
print(f"{'='*80}")

print(f"\n{'#':<4} {'Matchup':<25} {'Our Line':<10} {'Vegas Line':<12} {'Actual Result':<15} {'Error':<8}")
print("-" * 85)

for idx, row in first_10.iterrows():
    game_num = idx + 1
    matchup = f"{row['away_team']} @ {row['home_team']}"
    our_line = row['predicted_spread']
    vegas_line = row['actual_spread']  # This is Vegas closing line

    # Get actual game result
    if 'home_score' in row and 'away_score' in row and pd.notna(row['home_score']):
        actual_diff = row['home_score'] - row['away_score']
        actual_result = f"{actual_diff:+.0f}"
    else:
        actual_result = "Not played"

    error = row['prediction_error']

    # Color code error (conceptually - just use symbols)
    if abs(error) <= 3:
        error_marker = "✓"  # Good prediction
    elif abs(error) <= 7:
        error_marker = "~"  # Okay
    else:
        error_marker = "✗"  # Poor

    print(f"{game_num:<4} {matchup:<25} {our_line:>9.1f} {vegas_line:>11.1f} {actual_result:>14} {error:>7.1f} {error_marker}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS (First 10 Games)")
print("="*80)

mae = first_10['prediction_error'].abs().mean()
rmse = np.sqrt((first_10['prediction_error'] ** 2).mean())
max_error = first_10['prediction_error'].abs().max()
good_predictions = (first_10['prediction_error'].abs() <= 3).sum()

print(f"\nPrediction Accuracy:")
print(f"  Mean Absolute Error (MAE): {mae:.2f} points")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} points")
print(f"  Maximum Error: {max_error:.2f} points")
print(f"  Within 3 points: {good_predictions}/10 games ({good_predictions/10*100:.0f}%)")

# Compare to full 2025 season
print(f"\n{'='*80}")
print("COMPARISON: Week 1 vs Full 2025 Season")
print("="*80)

full_mae = predictions['prediction_error'].abs().mean()
full_good = (predictions['prediction_error'].abs() <= 3).sum()

print(f"\nWeek 1 (first 10 games):")
print(f"  MAE: {mae:.2f} points")
print(f"  Within 3 points: {good_predictions/10*100:.0f}%")

print(f"\nFull 2025 Season ({len(predictions)} games):")
print(f"  MAE: {full_mae:.2f} points")
print(f"  Within 3 points: {full_good/len(predictions)*100:.0f}%")

# Show biggest misses
print(f"\n{'='*80}")
print("BIGGEST PREDICTION MISSES (First 10 Games)")
print("="*80)

biggest_misses = first_10.nlargest(3, 'prediction_error', keep='all')[
    ['away_team', 'home_team', 'predicted_spread', 'actual_spread', 'prediction_error']
]

print(f"\n{'Matchup':<25} {'Our Line':<12} {'Vegas Line':<12} {'Miss By'}")
print("-" * 65)
for _, row in biggest_misses.iterrows():
    matchup = f"{row['away_team']} @ {row['home_team']}"
    print(f"{matchup:<25} {row['predicted_spread']:>11.1f} {row['actual_spread']:>11.1f} {row['prediction_error']:>11.1f}")

print("\n" + "="*80 + "\n")
