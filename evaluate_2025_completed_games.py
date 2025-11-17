"""
Evaluate Ball Knower on COMPLETED 2025 games with actual scores
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("BALL KNOWER - 2025 COMPLETED GAMES EVALUATION")
print("="*80)

# Load schedules with actual scores
schedules = pd.read_parquet('/home/user/BK_Build/schedules.parquet')
schedules_2025 = schedules[
    (schedules['season'] == 2025) &
    (schedules['game_type'] == 'REG') &
    (schedules['home_score'].notna())  # Only completed games
].copy()

# Load our predictions
predictions = pd.read_csv('/home/user/BK_Build/output/ball_knower_production_2025_predictions.csv')

# Merge to get our predictions
merged = schedules_2025.merge(
    predictions[['season', 'week', 'home_team', 'away_team', 'predicted_spread']],
    on=['season', 'week', 'home_team', 'away_team'],
    how='inner'
)

# Calculate actual spread
merged['actual_spread_result'] = merged['home_score'] - merged['away_score']

# Calculate errors
merged['our_error'] = merged['predicted_spread'] - merged['actual_spread_result']
merged['vegas_error'] = merged['spread_line'] - merged['actual_spread_result']

print(f"\n✓ Found {len(merged)} completed games in 2025")

if len(merged) == 0:
    print("\n⚠ No completed games found with scores in 2025 data")
    print("   This may mean games haven't been played yet or scores aren't updated")
else:
    # Show first 10 completed games
    first_10 = merged.head(10)

    print(f"\n{'='*95}")
    print(f"FIRST 10 COMPLETED GAMES")
    print(f"{'='*95}")

    print(f"\n{'Week':<6} {'Matchup':<22} {'Our Line':<10} {'Vegas':<8} {'Actual':<8} {'Our Error':<11} {'Vegas Error'}")
    print("-" * 95)

    for _, row in first_10.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        week = row['week']
        our_line = row['predicted_spread']
        vegas_line = row['spread_line']
        actual = row['actual_spread_result']
        our_error = row['our_error']
        vegas_error = row['vegas_error']

        # Mark if we beat Vegas
        if abs(our_error) < abs(vegas_error):
            marker = " ✓ BEAT VEGAS"
        elif abs(our_error) == abs(vegas_error):
            marker = " = TIED"
        else:
            marker = ""

        print(f"{week:<6} {matchup:<22} {our_line:>9.1f} {vegas_line:>7.1f} {actual:>7.1f} {our_error:>10.1f} {vegas_error:>12.1f}{marker}")

    # Summary statistics
    print(f"\n{'='*95}")
    print(f"PERFORMANCE SUMMARY ({len(merged)} completed games)")
    print(f"{'='*95}")

    our_mae = merged['our_error'].abs().mean()
    vegas_mae = merged['vegas_error'].abs().mean()

    our_within_3 = (merged['our_error'].abs() <= 3).sum()
    vegas_within_3 = (merged['vegas_error'].abs() <= 3).sum()

    our_within_7 = (merged['our_error'].abs() <= 7).sum()
    vegas_within_7 = (merged['vegas_error'].abs() <= 7).sum()

    beat_vegas = (merged['our_error'].abs() < merged['vegas_error'].abs()).sum()
    tied_vegas = (merged['our_error'].abs() == merged['vegas_error'].abs()).sum()

    print(f"\nBall Knower (Our Model):")
    print(f"  MAE: {our_mae:.2f} points")
    print(f"  Within 3 points: {our_within_3}/{len(merged)} ({our_within_3/len(merged)*100:.1f}%)")
    print(f"  Within 7 points: {our_within_7}/{len(merged)} ({our_within_7/len(merged)*100:.1f}%)")

    print(f"\nVegas Closing Lines:")
    print(f"  MAE: {vegas_mae:.2f} points")
    print(f"  Within 3 points: {vegas_within_3}/{len(merged)} ({vegas_within_3/len(merged)*100:.1f}%)")
    print(f"  Within 7 points: {vegas_within_7}/{len(merged)} ({vegas_within_7/len(merged)*100:.1f}%)")

    print(f"\nHead-to-Head vs Vegas:")
    print(f"  We beat Vegas: {beat_vegas}/{len(merged)} games ({beat_vegas/len(merged)*100:.1f}%)")
    print(f"  Tied with Vegas: {tied_vegas}/{len(merged)} games ({tied_vegas/len(merged)*100:.1f}%)")
    print(f"  Vegas beat us: {len(merged)-beat_vegas-tied_vegas}/{len(merged)} games ({(len(merged)-beat_vegas-tied_vegas)/len(merged)*100:.1f}%)")

    if our_mae < vegas_mae:
        diff = vegas_mae - our_mae
        print(f"\n✓ We BEAT Vegas by {diff:.2f} points MAE!")
    elif our_mae == vegas_mae:
        print(f"\n= We TIED with Vegas")
    else:
        diff = our_mae - vegas_mae
        print(f"\n✗ Vegas beat us by {diff:.2f} points MAE")

    # Show games where we significantly beat Vegas
    print(f"\n{'='*95}")
    print(f"BIGGEST WINS OVER VEGAS (Our error < Vegas error)")
    print(f"{'='*95}")

    merged['vegas_advantage'] = merged['vegas_error'].abs() - merged['our_error'].abs()
    big_wins = merged.nlargest(5, 'vegas_advantage')[
        ['week', 'away_team', 'home_team', 'predicted_spread', 'spread_line',
         'actual_spread_result', 'our_error', 'vegas_error', 'vegas_advantage']
    ]

    print(f"\n{'Week':<6} {'Matchup':<22} {'Our Line':<10} {'Vegas':<8} {'Actual':<8} {'Advantage'}")
    print("-" * 80)
    for _, row in big_wins.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"{row['week']:<6} {matchup:<22} {row['predicted_spread']:>9.1f} {row['spread_line']:>7.1f} {row['actual_spread_result']:>7.1f} {row['vegas_advantage']:>9.1f}")

print("\n" + "="*95 + "\n")
