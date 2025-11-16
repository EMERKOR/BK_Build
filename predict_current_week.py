"""
Current Week Predictions - Ball Knower v1.2

Generates betting recommendations for Week 11, 2025 using the trained v1.2 model.

Model Performance (trained on 2009-2024):
- Test R² = 0.884
- Test MAE = 1.57 points
- Trained on 4,345 historical games
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("BALL KNOWER v1.2 - CURRENT WEEK PREDICTIONS")
print("="*80)

# Load trained model
model_file = Path('/home/user/BK_Build/output/ball_knower_v1_2_model.json')
with open(model_file, 'r') as f:
    model_params = json.load(f)

print(f"\nLoaded v1.2 model:")
print(f"  Training R²: {model_params['train_r2']:.3f}")
print(f"  Test R²:     {model_params['test_r2']:.3f}")
print(f"  Test MAE:    {model_params['test_mae']:.2f} points")

# ============================================================================
# LOAD CURRENT WEEK DATA
# ============================================================================

print("\n[1/4] Loading Week 11 2025 data...")

# Load games
games = nflverse.games(season=2025, week=11)
games = games[games['spread_line'].notna()].copy()

# Load team ratings from nfelo snapshot
nfelo_snapshot_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'
nfelo_ratings = pd.read_csv(nfelo_snapshot_url)

# Handle duplicate teams (take first occurrence - likely most recent)
nfelo_ratings = nfelo_ratings.drop_duplicates(subset=['team'], keep='first')

print(f"Loaded {len(games)} games with Vegas lines")
print(f"Loaded {len(nfelo_ratings)} team ratings")

# ============================================================================
# PREPARE FEATURES FOR PREDICTION
# ============================================================================

print("\n[2/4] Engineering features for current week...")

# Merge ratings
matchups = games[['away_team', 'home_team', 'spread_line',
                  'home_rest', 'away_rest', 'div_game']].copy()

matchups = matchups.merge(
    nfelo_ratings[['team', 'nfelo']],
    left_on='home_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={'nfelo': 'home_nfelo'})

matchups = matchups.merge(
    nfelo_ratings[['team', 'nfelo']],
    left_on='away_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={'nfelo': 'away_nfelo'})

matchups = matchups.dropna(subset=['home_nfelo', 'away_nfelo'])

# Engineer features (matching v1.2 training)
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']

# Rest advantage (simplified - we don't have bye/surface data for current week easily)
# Use rest days as proxy
matchups['home_rest'] = matchups['home_rest'].fillna(7)
matchups['away_rest'] = matchups['away_rest'].fillna(7)

# Simplified rest advantage
def calc_rest_adv(row):
    home_bye = 1 if row['home_rest'] >= 14 else 0
    away_bye = 1 if row['away_rest'] >= 14 else 0
    return home_bye - away_bye

matchups['rest_advantage'] = matchups.apply(calc_rest_adv, axis=1)

# Divisional game flag
matchups['div_game'] = matchups['div_game'].fillna(0)

# Placeholder for features we don't have real-time
matchups['surface_mod'] = 0.0  # Would need stadium data
matchups['time_advantage'] = 0.0  # Would need detailed timezone tracking
matchups['qb_diff'] = 0.0  # Would need current QB ratings

print(f"✓ Prepared features for {len(matchups)} games")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n[3/4] Generating predictions with v1.2 model...")

# Apply model coefficients
intercept = model_params['intercept']
coefs = model_params['coefficients']

matchups['bk_v1_2_spread'] = intercept + \
    (matchups['nfelo_diff'] * coefs['nfelo_diff']) + \
    (matchups['rest_advantage'] * coefs['rest_advantage']) + \
    (matchups['div_game'] * coefs['div_game']) + \
    (matchups['surface_mod'] * coefs['surface_mod']) + \
    (matchups['time_advantage'] * coefs['time_advantage']) + \
    (matchups['qb_diff'] * coefs['qb_diff'])

# Calculate edge
matchups['edge'] = matchups['bk_v1_2_spread'] - matchups['spread_line']
matchups['abs_edge'] = matchups['edge'].abs()

print("✓ Predictions generated")

# ============================================================================
# IDENTIFY VALUE BETS
# ============================================================================

print("\n[4/4] Identifying value bets...")

print("\n" + "="*80)
print("ALL WEEK 11 PREDICTIONS")
print("="*80)

display_cols = ['away_team', 'home_team', 'spread_line', 'bk_v1_2_spread',
                'edge', 'home_rest', 'away_rest', 'div_game']

all_preds = matchups[display_cols].copy()
all_preds = all_preds.sort_values('edge', ascending=False)
all_preds = all_preds.round(2)

print("\n" + all_preds.to_string(index=False))

# Value bet thresholds
print("\n" + "="*80)
print("VALUE BET ANALYSIS")
print("="*80)

thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]

print("\nBets by edge threshold:")
for threshold in thresholds:
    count = len(matchups[matchups['abs_edge'] >= threshold])
    print(f"  {threshold}+ points: {count} games")

# Recommendations at 2.0 point threshold
value_threshold = 2.0
value_bets = matchups[matchups['abs_edge'] >= value_threshold].copy()

print(f"\n" + "="*80)
print(f"VALUE BETS ({value_threshold}+ POINT EDGE)")
print("="*80)

if len(value_bets) > 0:
    value_bets['bet_side'] = value_bets['edge'].apply(
        lambda x: 'HOME' if x < 0 else 'AWAY'
    )

    value_bets['bet_team'] = value_bets.apply(
        lambda row: row['home_team'] if row['bet_side'] == 'HOME' else row['away_team'],
        axis=1
    )

    value_bets['recommendation'] = value_bets.apply(
        lambda row: f"Bet {row['bet_team']} {row['spread_line']:.1f} (edge: {abs(row['edge']):.1f})",
        axis=1
    )

    bet_display = value_bets[[
        'away_team', 'home_team', 'spread_line', 'bk_v1_2_spread',
        'edge', 'bet_side', 'recommendation'
    ]].copy()

    bet_display = bet_display.sort_values('edge', key=abs, ascending=False)
    bet_display = bet_display.round(2)

    print(f"\n{len(value_bets)} recommended bets:\n")
    print(bet_display.to_string(index=False))

    # Risk analysis
    print("\n" + "="*80)
    print("BETTING RECOMMENDATIONS - RISK ANALYSIS")
    print("="*80)

    print(f"""
Total recommended bets: {len(value_bets)}

Confidence tiers (based on edge size):
""")

    for bet_idx, bet in bet_display.iterrows():
        edge_mag = abs(bet['edge'])
        if edge_mag >= 3.0:
            confidence = "HIGH"
        elif edge_mag >= 2.5:
            confidence = "MODERATE-HIGH"
        elif edge_mag >= 2.0:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        print(f"  {bet['away_team']} @ {bet['home_team']}: "
              f"{bet['recommendation']} - {confidence} CONFIDENCE")

else:
    print(f"\nNo games meet the {value_threshold}+ point threshold this week.")
    print(f"Largest edge: {matchups['abs_edge'].max():.2f} points")

# Model comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = matchups[['away_team', 'home_team', 'spread_line', 'bk_v1_2_spread', 'edge']].copy()

print("\nv1.2 Performance on Week 11:")
print(f"  Mean absolute edge: {matchups['abs_edge'].mean():.2f} points")
print(f"  Median edge: {matchups['edge'].median():.2f} points")
print(f"  Std dev: {matchups['edge'].std():.2f} points")

# Save predictions
output_file = config.OUTPUT_DIR / 'week_11_value_bets_v1_2.csv'
matchups.to_csv(output_file, index=False)

print(f"\n✓ Predictions saved to: {output_file}")

print("\n" + "="*80)
print("BETTING GUIDELINES")
print("="*80)

print("""
Recommended Betting Strategy:

1. EDGE REQUIREMENTS:
   - Only bet when edge >= 2.0 points
   - Higher confidence for edges >= 2.5 points
   - Max confidence for edges >= 3.0 points

2. BANKROLL MANAGEMENT:
   - Never bet more than 2-3% of bankroll per game
   - Scale bet size with edge (Kelly Criterion)
   - Track all bets for long-term analysis

3. LINE SHOPPING:
   - These edges are vs current lines
   - Shop multiple sportsbooks for best available
   - Consider line movement (early week often better)

4. MODEL LIMITATIONS:
   - Test MAE = 1.57 points (expect ~1.5 point error)
   - Missing some situational factors (weather, injuries)
   - Based on closing line value, not actual outcomes
   - Past performance doesn't guarantee future results

5. RESPONSIBLE GAMBLING:
   - Only bet what you can afford to lose
   - This is for entertainment/education
   - Not financial advice
   - Track results honestly

Good luck! 🏈
""")

print("="*80 + "\n")
