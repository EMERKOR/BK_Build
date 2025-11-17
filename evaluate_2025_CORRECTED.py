"""
CORRECTED evaluation - accounting for spread sign conventions

In NFL spreads:
- Positive number = home team favored by that many points
- Negative number = home team underdog by that many points
- Actual result = home_score - away_score
"""

import pandas as pd
import numpy as np

print("\n" + "="*100)
print("BALL KNOWER - 2025 SEASON EVALUATION (CORRECTED)")
print("="*100)

# Load schedules with actual scores
schedules = pd.read_parquet('/home/user/BK_Build/schedules.parquet')
schedules_2025 = schedules[
    (schedules['season'] == 2025) &
    (schedules['game_type'] == 'REG') &
    (schedules['home_score'].notna()) &
    (schedules['spread_line'].notna())
].copy()

# Load our predictions
predictions = pd.read_csv('/home/user/BK_Build/output/ball_knower_production_2025_predictions.csv')

# The nfelo data uses NEGATIVE for home favorites
# We need to flip the sign to match standard NFL convention
predictions['predicted_spread_correct'] = -predictions['predicted_spread']

# Merge
merged = schedules_2025.merge(
    predictions[['season', 'week', 'home_team', 'away_team', 'predicted_spread_correct']],
    on=['season', 'week', 'home_team', 'away_team'],
    how='inner'
)

# Calculate actual result (positive = home team won by that much)
merged['actual_result'] = merged['home_score'] - merged['away_score']

# Vegas spread (positive = home team favored)
# In schedules.parquet, spread_line is for HOME team
merged['vegas_spread'] = merged['spread_line']

# Calculate errors (predicted spread - actual result)
merged['our_error'] = merged['predicted_spread_correct'] - merged['actual_result']
merged['vegas_error'] = merged['vegas_spread'] - merged['actual_result']

print(f"\n✓ Analyzing {len(merged)} completed 2025 games")

# Show first 10 games
first_10 = merged.head(10)

print(f"\n{'='*100}")
print(f"FIRST 10 COMPLETED GAMES")
print(f"{'='*100}")

print(f"\n{'Wk':<4} {'Matchup':<20} {'Score':<12} {'Our Line':<10} {'Vegas':<8} {'Result':<8} {'Our Err':<9} {'Vegas Err':<10} {'Better?'}")
print("-" * 100)

for _, row in first_10.iterrows():
    matchup = f"{row['away_team']} @ {row['home_team']}"
    score = f"{row['away_score']:.0f}-{row['home_score']:.0f}"
    our_line = row['predicted_spread_correct']
    vegas = row['vegas_spread']
    result = row['actual_result']
    our_err = row['our_error']
    vegas_err = row['vegas_error']

    # Who was more accurate?
    if abs(our_err) < abs(vegas_err):
        better = "✓ US"
    elif abs(our_err) > abs(vegas_err):
        better = "VEGAS"
    else:
        better = "TIE"

    print(f"{row['week']:<4} {matchup:<20} {score:<12} {our_line:>9.1f} {vegas:>7.1f} {result:>7.1f} {our_err:>8.1f} {vegas_err:>9.1f} {better}")

# Overall statistics
print(f"\n{'='*100}")
print(f"OVERALL PERFORMANCE ({len(merged)} games)")
print(f"{'='*100}")

our_mae = merged['our_error'].abs().mean()
our_rmse = np.sqrt((merged['our_error'] ** 2).mean())
vegas_mae = merged['vegas_error'].abs().mean()
vegas_rmse = np.sqrt((merged['vegas_error'] ** 2).mean())

our_within_3 = (merged['our_error'].abs() <= 3).sum()
our_within_7 = (merged['our_error'].abs() <= 7).sum()
vegas_within_3 = (merged['vegas_error'].abs() <= 3).sum()
vegas_within_7 = (merged['vegas_error'].abs() <= 7).sum()

beat_vegas = (merged['our_error'].abs() < merged['vegas_error'].abs()).sum()

print(f"\nBall Knower (Our Model):")
print(f"  MAE:  {our_mae:.2f} points")
print(f"  RMSE: {our_rmse:.2f} points")
print(f"  Within 3 points: {our_within_3}/{len(merged)} ({our_within_3/len(merged)*100:.1f}%)")
print(f"  Within 7 points: {our_within_7}/{len(merged)} ({our_within_7/len(merged)*100:.1f}%)")

print(f"\nVegas Closing Lines:")
print(f"  MAE:  {vegas_mae:.2f} points")
print(f"  RMSE: {vegas_rmse:.2f} points")
print(f"  Within 3 points: {vegas_within_3}/{len(merged)} ({vegas_within_3/len(merged)*100:.1f}%)")
print(f"  Within 7 points: {vegas_within_7}/{len(merged)} ({vegas_within_7/len(merged)*100:.1f}%)")

print(f"\nHead-to-Head:")
print(f"  Games we beat Vegas: {beat_vegas}/{len(merged)} ({beat_vegas/len(merged)*100:.1f}%)")

if our_mae < vegas_mae:
    diff = vegas_mae - our_mae
    pct_better = (diff / vegas_mae) * 100
    print(f"\n✓✓✓ WE BEAT VEGAS by {diff:.2f} points ({pct_better:.1f}% better)!")
else:
    diff = our_mae - vegas_mae
    pct_worse = (diff / vegas_mae) * 100
    print(f"\n✗ Vegas beat us by {diff:.2f} points ({pct_worse:.1f}% worse)")

# Show best predictions
print(f"\n{'='*100}")
print(f"OUR 5 BEST PREDICTIONS (smallest absolute error)")
print(f"{'='*100}")

merged['our_error_abs'] = merged['our_error'].abs()
best = merged.nsmallest(5, 'our_error_abs')[
    ['week', 'away_team', 'home_team', 'predicted_spread_correct', 'vegas_spread', 'actual_result', 'our_error']
]

print(f"\n{'Wk':<4} {'Matchup':<25} {'Our Line':<10} {'Vegas':<8} {'Result':<8} {'Error'}")
print("-" * 75)
for _, row in best.iterrows():
    matchup = f"{row['away_team']} @ {row['home_team']}"
    print(f"{row['week']:<4} {matchup:<25} {row['predicted_spread_correct']:>9.1f} {row['vegas_spread']:>7.1f} {row['actual_result']:>7.1f} {row['our_error']:>7.1f}")

# Show where we significantly beat Vegas
print(f"\n{'='*100}")
print(f"GAMES WHERE WE BEAT VEGAS BY 5+ POINTS")
print(f"{'='*100}")

merged['edge_over_vegas'] = merged['vegas_error'].abs() - merged['our_error'].abs()
big_wins = merged[merged['edge_over_vegas'] >= 5].sort_values('edge_over_vegas', ascending=False)

if len(big_wins) > 0:
    print(f"\n{'Wk':<4} {'Matchup':<25} {'Our Err':<10} {'Vegas Err':<12} {'Edge'}")
    print("-" * 75)
    for _, row in big_wins.head(10).iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"{row['week']:<4} {matchup:<25} {row['our_error']:>9.1f} {row['vegas_error']:>11.1f} {row['edge_over_vegas']:>7.1f}")
else:
    print("\nNo games where we beat Vegas by 5+ points")

print("\n" + "="*100 + "\n")
