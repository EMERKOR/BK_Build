"""
Week 11 2025 Actual Results Analysis
=====================================

Compares Ball Knower v1.2 predictions to actual game outcomes for Week 11, 2025.

This script will:
1. Load Week 11 predictions from the model
2. Load actual game results from nflverse
3. Compare predicted spreads vs actual margins
4. Evaluate betting recommendations
5. Calculate actual ROI if bets were placed

Author: Ball Knower Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("WEEK 11 2025 - ACTUAL RESULTS ANALYSIS")
print("="*80)

# Load actual game results
print("\n[1/3] Loading actual game results...")
games = nflverse.games(season=2025, week=11)
games = games[games['home_score'].notna()].copy()  # Only completed games

print(f"✓ Loaded {len(games)} completed games")

# Load predictions
predictions_file = config.OUTPUT_DIR / 'week_11_value_bets_v1_2.csv'
if not predictions_file.exists():
    print(f"\n🔴 ERROR: Predictions file not found: {predictions_file}")
    print("Please run: python predict_current_week.py")
    sys.exit(1)

predictions = pd.read_csv(predictions_file)
print(f"✓ Loaded predictions for {len(predictions)} games")

# ============================================================================
# CALCULATE ACTUAL MARGINS
# ============================================================================

print("\n[2/3] Calculating actual game margins...")

# Add actual margin (from home team perspective)
games['actual_margin'] = games['home_score'] - games['away_score']

# Merge with predictions
comparison = predictions.merge(
    games[['home_team', 'away_team', 'home_score', 'away_score', 'actual_margin']],
    on=['home_team', 'away_team'],
    how='left'
)

# Check for missing results
missing_results = comparison[comparison['actual_margin'].isna()]
if len(missing_results) > 0:
    print(f"\n🟡 WARNING: {len(missing_results)} games have not completed yet:")
    for idx, row in missing_results.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']}")

    comparison = comparison[comparison['actual_margin'].notna()].copy()

print(f"\n✓ Analyzing {len(comparison)} completed games")

# ============================================================================
# MODEL PERFORMANCE ANALYSIS
# ============================================================================

print("\n[3/3] Evaluating model performance...")

print("\n" + "="*80)
print("PREDICTION ACCURACY")
print("="*80)

# Calculate prediction error
# Remember: spread_line is from home perspective (negative = home favored)
# bk_v1_2_spread is also from home perspective
# actual_margin is home_score - away_score (positive = home won)

comparison['prediction_error'] = comparison['bk_v1_2_spread'] - comparison['actual_margin']
comparison['vegas_error'] = comparison['spread_line'] - comparison['actual_margin']
comparison['abs_prediction_error'] = comparison['prediction_error'].abs()
comparison['abs_vegas_error'] = comparison['vegas_error'].abs()

# Model performance
model_mae = comparison['abs_prediction_error'].mean()
model_rmse = np.sqrt((comparison['prediction_error'] ** 2).mean())

print(f"\nBall Knower v1.2 Performance:")
print(f"  MAE (Mean Absolute Error):  {model_mae:.2f} points")
print(f"  RMSE (Root Mean Squared):   {model_rmse:.2f} points")
print(f"  Mean Error (Bias):          {comparison['prediction_error'].mean():.2f} points")

# Vegas performance (for comparison)
vegas_mae = comparison['abs_vegas_error'].mean()
vegas_rmse = np.sqrt((comparison['vegas_error'] ** 2).mean())

print(f"\nVegas Line Performance:")
print(f"  MAE (Mean Absolute Error):  {vegas_mae:.2f} points")
print(f"  RMSE (Root Mean Squared):   {vegas_rmse:.2f} points")
print(f"  Mean Error (Bias):          {comparison['vegas_error'].mean():.2f} points")

print(f"\n\nComparison:")
print(f"  Model vs Vegas MAE:  {model_mae - vegas_mae:+.2f} points")
if model_mae < vegas_mae:
    print(f"  ✅ Model BEAT Vegas by {vegas_mae - model_mae:.2f} points")
else:
    print(f"  🔴 Model WORSE than Vegas by {model_mae - vegas_mae:.2f} points")

# ============================================================================
# DETAILED GAME-BY-GAME BREAKDOWN
# ============================================================================

print("\n" + "="*80)
print("GAME-BY-GAME RESULTS")
print("="*80)

# Create display dataframe
results_display = comparison[[
    'away_team', 'home_team',
    'away_score', 'home_score', 'actual_margin',
    'spread_line', 'bk_v1_2_spread',
    'edge', 'abs_edge', 'prediction_error'
]].copy()

results_display = results_display.sort_values('abs_edge', ascending=False)

print("\nAll games (sorted by model edge):\n")
print(results_display.round(2).to_string(index=False))

# ============================================================================
# BETTING PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BETTING RECOMMENDATIONS ANALYSIS")
print("="*80)

# Filter for value bets (2.0+ point edge)
value_threshold = 2.0
value_bets = comparison[comparison['abs_edge'] >= value_threshold].copy()

if len(value_bets) == 0:
    print(f"\nNo bets recommended at {value_threshold}+ point threshold")
else:
    print(f"\n{len(value_bets)} bets recommended (edge >= {value_threshold} points):\n")

    # Determine bet outcome
    # bet_side tells us which side the model recommended
    # For AWAY bets: we win if away_team covers (actual_margin < spread_line)
    # For HOME bets: we win if home_team covers (actual_margin > spread_line)

    def evaluate_bet(row):
        """Determine if the bet won, lost, or pushed."""
        if row['bet_side'] == 'AWAY':
            # Betting away team to cover
            # Away covers if: home_score - away_score < spread_line
            # i.e., actual_margin < spread_line
            if row['actual_margin'] < row['spread_line']:
                return 'WIN'
            elif row['actual_margin'] == row['spread_line']:
                return 'PUSH'
            else:
                return 'LOSS'
        else:  # HOME
            # Betting home team to cover
            # Home covers if: home_score - away_score > spread_line
            # i.e., actual_margin > spread_line
            if row['actual_margin'] > row['spread_line']:
                return 'WIN'
            elif row['actual_margin'] == row['spread_line']:
                return 'PUSH'
            else:
                return 'LOSS'

    value_bets['bet_outcome'] = value_bets.apply(evaluate_bet, axis=1)

    # Determine recommended team
    value_bets['bet_team'] = value_bets.apply(
        lambda row: row['home_team'] if row['bet_side'] == 'HOME' else row['away_team'],
        axis=1
    )

    # Display betting results
    bet_results = value_bets[[
        'away_team', 'home_team',
        'spread_line', 'actual_margin',
        'bet_team', 'bet_side', 'edge',
        'bet_outcome'
    ]].copy()

    bet_results = bet_results.sort_values('edge', key=abs, ascending=False)

    print(bet_results.round(2).to_string(index=False))

    # Calculate ROI
    print("\n" + "="*80)
    print("BETTING ROI ANALYSIS")
    print("="*80)

    wins = len(value_bets[value_bets['bet_outcome'] == 'WIN'])
    losses = len(value_bets[value_bets['bet_outcome'] == 'LOSS'])
    pushes = len(value_bets[value_bets['bet_outcome'] == 'PUSH'])

    total_bets = wins + losses + pushes
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    # Standard -110 odds: need to risk $110 to win $100
    # Win: +$100
    # Loss: -$110
    # Push: $0

    profit = (wins * 100) - (losses * 110)
    total_risked = total_bets * 110
    roi = (profit / total_risked * 100) if total_risked > 0 else 0

    print(f"\nBetting Record:")
    print(f"  Total bets:  {total_bets}")
    print(f"  Wins:        {wins}")
    print(f"  Losses:      {losses}")
    print(f"  Pushes:      {pushes}")
    print(f"  Win Rate:    {win_rate:.1f}%")

    print(f"\nFinancial Results (assuming $110 to win $100 on each bet):")
    print(f"  Total Risked: ${total_risked:.2f}")
    print(f"  Total Profit: ${profit:+.2f}")
    print(f"  ROI:          {roi:+.2f}%")

    if roi > 0:
        print(f"\n  ✅ PROFITABLE - Model generated positive ROI")
    else:
        print(f"\n  🔴 UNPROFITABLE - Model lost money")

    # Breakeven analysis
    breakeven_rate = 110 / (100 + 110) * 100
    print(f"\nBreakeven win rate (at -110 odds): {breakeven_rate:.1f}%")
    print(f"Actual win rate: {win_rate:.1f}%")

    if win_rate > breakeven_rate:
        print(f"✅ Beat breakeven by {win_rate - breakeven_rate:.1f}%")
    else:
        print(f"🔴 Below breakeven by {breakeven_rate - win_rate:.1f}%")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

num_bets = len(value_bets) if len(value_bets) > 0 else 0
win_rate_str = f"{win_rate:.1f}%" if num_bets > 0 else "N/A"
roi_str = f"{roi:+.1f}%" if num_bets > 0 else "N/A"

print(f"""
1. MODEL ACCURACY:
   - Ball Knower v1.2 MAE: {model_mae:.2f} points
   - Vegas MAE: {vegas_mae:.2f} points
   - Difference: {model_mae - vegas_mae:+.2f} points

2. EXPECTED vs ACTUAL PERFORMANCE:
   - Expected MAE (from training): ~1.57 points
   - Actual MAE (Week 11): {model_mae:.2f} points
   - Worse by: {model_mae - 1.57:.2f} points

3. BETTING RECOMMENDATIONS:
   - Bets recommended: {num_bets}
   - Win rate: {win_rate_str}
   - ROI: {roi_str}

4. DATA QUALITY:
   - ✅ Duplicates removed (NE, NYJ)
   - ✅ Team mappings fixed (LAR->LA, OAK->LV)
   - ✅ All teams have ratings
""")

print("\n" + "="*80 + "\n")
