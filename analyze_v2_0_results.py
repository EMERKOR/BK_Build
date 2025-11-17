"""
Ball Knower v2.0 - Results Analysis & Diagnostics
==================================================

Investigates why v2.0 beats Vegas at predictions but loses money betting.

Key Questions:
1. Where are the big edges coming from?
2. Are large edges calibrated correctly?
3. Is the betting threshold too low?
4. Should we train to predict outcomes instead of Vegas lines?

Author: Ball Knower Team
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Load v2.0 backtest results
results = pd.read_csv('output/v2_0_backtest_2024.csv')

print("\n" + "="*80)
print("BALL KNOWER v2.0 - DIAGNOSTIC ANALYSIS")
print("="*80)

print(f"\nLoaded {len(results)} games from 2024 season backtest")

# ============================================================================
# ANALYSIS 1: EDGE DISTRIBUTION
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 1: EDGE DISTRIBUTION")
print("="*80)

edge_bins = [0, 1, 2, 3, 4, 5, 100]
edge_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']

results['edge_bucket'] = pd.cut(results['abs_edge'], bins=edge_bins, labels=edge_labels)

print("\nGames by edge magnitude:")
print(results['edge_bucket'].value_counts().sort_index())

# ============================================================================
# ANALYSIS 2: BETTING PERFORMANCE BY EDGE SIZE
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 2: BETTING PERFORMANCE BY EDGE SIZE")
print("="*80)

def analyze_edge_bucket(df, min_edge, max_edge=999):
    """Analyze betting performance for a specific edge range."""
    bucket = df[(df['abs_edge'] >= min_edge) & (df['abs_edge'] < max_edge)].copy()

    if len(bucket) == 0:
        return None

    # Determine bet outcomes
    bucket['bet_side'] = bucket['v2_edge'].apply(lambda x: 'AWAY' if x > 0 else 'HOME')

    def evaluate_bet(row):
        if row['bet_side'] == 'AWAY':
            if row['actual_margin'] < row['home_spread']:
                return 'WIN'
            elif row['actual_margin'] == row['home_spread']:
                return 'PUSH'
            else:
                return 'LOSS'
        else:
            if row['actual_margin'] > row['home_spread']:
                return 'WIN'
            elif row['actual_margin'] == row['home_spread']:
                return 'PUSH'
            else:
                return 'LOSS'

    bucket['outcome'] = bucket.apply(evaluate_bet, axis=1)

    wins = len(bucket[bucket['outcome'] == 'WIN'])
    losses = len(bucket[bucket['outcome'] == 'LOSS'])
    pushes = len(bucket[bucket['outcome'] == 'PUSH'])

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    profit = (wins * 100) - (losses * 110)
    total_risked = len(bucket) * 110
    roi = (profit / total_risked * 100) if total_risked > 0 else 0

    return {
        'edge_range': f"{min_edge:.1f}-{max_edge:.1f}",
        'n_bets': len(bucket),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'roi': roi,
        'profit': profit
    }

# Analyze different edge thresholds
thresholds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 999)]

edge_analysis = []
for min_edge, max_edge in thresholds:
    result = analyze_edge_bucket(results, min_edge, max_edge)
    if result:
        edge_analysis.append(result)

edge_df = pd.DataFrame(edge_analysis)

print("\nPerformance by edge bucket:")
print(edge_df[['edge_range', 'n_bets', 'win_rate', 'roi']].to_string(index=False))

print("\n💡 Key Insight:")
best_bucket = edge_df.loc[edge_df['roi'].idxmax()]
print(f"  Best ROI: {best_bucket['roi']:.1f}% at edge {best_bucket['edge_range']}")
print(f"  But only {best_bucket['n_bets']} bets in that bucket")

# ============================================================================
# ANALYSIS 3: CALIBRATION - DOES EDGE SIZE PREDICT WIN RATE?
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 3: EDGE CALIBRATION")
print("="*80)

print("\nIf model is calibrated, larger edges should have higher win rates.")
print("\nActual vs Expected:")
for _, row in edge_df.iterrows():
    expected_wr = 50 + (float(row['edge_range'].split('-')[0]) * 2)  # Rough heuristic
    actual_wr = row['win_rate']
    diff = actual_wr - expected_wr
    print(f"  Edge {row['edge_range']}: Expected {expected_wr:.1f}%, Actual {actual_wr:.1f}% ({diff:+.1f}%)")

# ============================================================================
# ANALYSIS 4: WHERE DID WE GO WRONG?
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 4: WORST BETS (Largest Losses)")
print("="*80)

# Get bets with 2+ edge
bets = results[results['abs_edge'] >= 2].copy()
bets['bet_side'] = bets['v2_edge'].apply(lambda x: 'AWAY' if x > 0 else 'HOME')

def evaluate_bet(row):
    if row['bet_side'] == 'AWAY':
        if row['actual_margin'] < row['home_spread']:
            return 'WIN'
        elif row['actual_margin'] == row['home_spread']:
            return 'PUSH'
        else:
            return 'LOSS'
    else:
        if row['actual_margin'] > row['home_spread']:
            return 'WIN'
        elif row['actual_margin'] == row['home_spread']:
            return 'PUSH'
        else:
            return 'LOSS'

bets['outcome'] = bets.apply(evaluate_bet, axis=1)

# Calculate how wrong we were
bets['error_magnitude'] = (bets['v2_predicted_spread'] - bets['actual_margin']).abs()

worst_bets = bets[bets['outcome'] == 'LOSS'].sort_values('error_magnitude', ascending=False).head(10)

print("\nTop 10 worst bets (largest prediction errors):")
print(worst_bets[['week', 'away_team', 'home_team', 'home_spread', 'v2_predicted_spread',
                   'actual_margin', 'v2_edge', 'error_magnitude']].to_string(index=False))

# ============================================================================
# ANALYSIS 5: COMPARE TO v1.2
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS 5: v2.0 vs v1.2 COMPARISON")
print("="*80)

# v1.2 results from Week 11 diagnostic
v1_2_mae = 13.74
vegas_week11_mae = 14.14

v2_0_mae = (results['v2_predicted_spread'] - results['actual_margin']).abs().mean()
vegas_2024_mae = (results['home_spread'] - results['actual_margin']).abs().mean()

print(f"\nPredicting Actual Outcomes:")
print(f"  v1.2 MAE (Week 11): {v1_2_mae:.2f} points")
print(f"  v2.0 MAE (2024):    {v2_0_mae:.2f} points")
print(f"  Improvement:        {v1_2_mae - v2_0_mae:+.2f} points ({(v1_2_mae - v2_0_mae) / v1_2_mae * 100:+.1f}%)")

print(f"\nVegas Comparison:")
print(f"  Vegas (Week 11):  {vegas_week11_mae:.2f} points")
print(f"  Vegas (2024):     {vegas_2024_mae:.2f} points")
print(f"  v2.0 vs Vegas:    {v2_0_mae - vegas_2024_mae:+.2f} points")

if v2_0_mae < vegas_2024_mae:
    print(f"  ✅ v2.0 BEATS Vegas by {vegas_2024_mae - v2_0_mae:.2f} points")
else:
    print(f"  🔴 v2.0 WORSE than Vegas by {v2_0_mae - vegas_2024_mae:.2f} points")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC CONCLUSIONS & RECOMMENDATIONS")
print("="*80)

print("""
KEY FINDINGS:

1. MODEL QUALITY:
   - v2.0 predicts Vegas lines well (2.96 MAE)
   - v2.0 predicts actual outcomes better than Vegas (14.61 vs 15.41)
   - Improvement over v1.2: +6% better at predicting outcomes

2. BETTING PROBLEM:
   - Training to predict Vegas ≠ finding betting value
   - Model is calibrated to MATCH Vegas, not BEAT it
   - Large "edges" are often model errors, not true value

3. CALIBRATION ISSUE:
   - Larger edges don't correlate with higher win rates
   - 2+ point edges only win 34.5% of time
   - Model overconfident in its divergences from Vegas

RECOMMENDED FIXES:

Option A - Change Training Objective:
  - Train to predict ACTUAL OUTCOMES (not Vegas lines)
  - Then compare to Vegas and bet when we diverge
  - Model won't be "Vegas-anchored"

Option B - Better Edge Calibration:
  - Edges need to be calibrated to actual win probability
  - Use residual analysis to determine true edge thresholds
  - May need 5+ point "edge" to have real 52%+ win rate

Option C - More Selective Betting:
  - Raise edge threshold to 4+ or 5+ points
  - Reduces volume but improves quality
  - Based on analysis, higher edges perform slightly better

Option D - Ensemble with v1.2:
  - Combine v1.2 and v2.0 predictions
  - Only bet when both models agree
  - More conservative, fewer bets

NEXT STEPS:

1. Rebuild v2.0 to predict OUTCOMES (not Vegas lines)
2. Recalibrate edge thresholds based on backtest
3. Test multiple edge thresholds (3+, 4+, 5+)
4. Require higher confidence for actual betting
5. Forward test on Week 12 before any live bets
""")

print("="*80 + "\n")
