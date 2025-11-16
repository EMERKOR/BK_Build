"""
Check if v1.2 model predictions align with actual Week 11 results
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n" + "="*80)
print("MODEL PREDICTION ACCURACY CHECK - WEEK 11 2025")
print("="*80)

# Load Week 11 games
games = nflverse.games(season=2025, week=11)
games = games[games['spread_line'].notna()].copy()

# Load nfelo ratings
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'
nfelo_ratings = pd.read_csv(nfelo_url)
nfelo_ratings = nfelo_ratings.drop_duplicates(subset=['team'], keep='first')

# Load v1.2 model
model_file = Path('/home/user/BK_Build/output/ball_knower_v1_2_model.json')
with open(model_file, 'r') as f:
    model_params = json.load(f)

# Prepare features
matchups = games[['away_team', 'home_team', 'spread_line', 'away_score', 'home_score',
                  'home_rest', 'away_rest', 'div_game']].copy()

# CRITICAL: Convert spread_line from AWAY to HOME perspective
matchups['vegas_line_home'] = -1 * matchups['spread_line']

# Merge nfelo ratings
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

# Engineer features
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups['home_rest'] = matchups['home_rest'].fillna(7)
matchups['away_rest'] = matchups['away_rest'].fillna(7)

def calc_rest_adv(row):
    home_bye = 1 if row['home_rest'] >= 14 else 0
    away_bye = 1 if row['away_rest'] >= 14 else 0
    return home_bye - away_bye

matchups['rest_advantage'] = matchups.apply(calc_rest_adv, axis=1)
matchups['div_game'] = matchups['div_game'].fillna(0)
matchups['surface_mod'] = 0.0
matchups['time_advantage'] = 0.0
matchups['qb_diff'] = 0.0

# Apply model
intercept = model_params['intercept']
coefs = model_params['coefficients']

matchups['model_pred'] = intercept + \
    (matchups['nfelo_diff'] * coefs['nfelo_diff']) + \
    (matchups['rest_advantage'] * coefs['rest_advantage']) + \
    (matchups['div_game'] * coefs['div_game']) + \
    (matchups['surface_mod'] * coefs['surface_mod']) + \
    (matchups['time_advantage'] * coefs['time_advantage']) + \
    (matchups['qb_diff'] * coefs['qb_diff'])

# Calculate actual margin (home score - away score)
matchups['actual_margin'] = matchups['home_score'] - matchups['away_score']

# Calculate errors (only for completed games)
completed = matchups[matchups['actual_margin'].notna()].copy()
completed['model_error'] = completed['model_pred'] - completed['actual_margin']
completed['vegas_error'] = completed['vegas_line_home'] - completed['actual_margin']

print("\n" + "="*80)
print("COMPLETED GAMES - MODEL vs VEGAS vs ACTUAL")
print("="*80)

print("\nLegend:")
print("  Vegas Line: From HOME perspective (negative = home favored)")
print("  Model Pred: Model's prediction (negative = home favored)")
print("  Actual: Actual margin (positive = home won, negative = away won)")
print("  Model Err: How far off model was")
print("  Vegas Err: How far off Vegas was")

display = completed[['away_team', 'home_team', 'vegas_line_home', 'model_pred',
                     'actual_margin', 'model_error', 'vegas_error']].copy()
display = display.round(2)
print("\n" + display.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY STATISTICS (Completed Games)")
print("="*80)

print(f"\nModel Performance:")
print(f"  Mean Absolute Error: {completed['model_error'].abs().mean():.2f} points")
print(f"  RMSE: {np.sqrt((completed['model_error']**2).mean()):.2f} points")
print(f"  Games: {len(completed)}")

print(f"\nVegas Performance:")
print(f"  Mean Absolute Error: {completed['vegas_error'].abs().mean():.2f} points")
print(f"  RMSE: {np.sqrt((completed['vegas_error']**2).mean()):.2f} points")

# Now check the betting recommendation logic
print("\n" + "="*80)
print("BETTING RECOMMENDATION ANALYSIS")
print("="*80)

matchups['edge'] = matchups['model_pred'] - matchups['vegas_line_home']

print("\nAll Week 11 games (including incomplete):")
bet_analysis = matchups[['away_team', 'home_team', 'vegas_line_home', 'model_pred',
                          'edge', 'actual_margin']].copy()
bet_analysis = bet_analysis.round(2)
print("\n" + bet_analysis.to_string(index=False))

print("\n" + "="*80)
print("EDGE INTERPRETATION")
print("="*80)

# Pick NE vs NYJ as example
ne_game = matchups[matchups['home_team'] == 'NE'].iloc[0]
print(f"\nExample: {ne_game['away_team']} @ {ne_game['home_team']}")
print(f"  Vegas Line (home perspective): {ne_game['vegas_line_home']:.2f}")
print(f"  Model Prediction: {ne_game['model_pred']:.2f}")
print(f"  Edge: {ne_game['edge']:.2f}")
print(f"  Actual Margin: {ne_game['actual_margin']:.2f}")

print(f"\nInterpretation:")
print(f"  Vegas says: {ne_game['home_team']} {ne_game['vegas_line_home']:.2f}")
print(f"  Model says: {ne_game['home_team']} {ne_game['model_pred']:.2f}")

if ne_game['edge'] > 0:
    print(f"  Edge is POSITIVE (+{ne_game['edge']:.2f})")
    print(f"    Model thinks home wins by MORE than Vegas")
    print(f"    Value bet: {ne_game['home_team']} (home)")
elif ne_game['edge'] < 0:
    print(f"  Edge is NEGATIVE ({ne_game['edge']:.2f})")
    print(f"    Model thinks home wins by LESS than Vegas")
    print(f"    Value bet: {ne_game['away_team']} (away)")

if not pd.isna(ne_game['actual_margin']):
    print(f"\n  What actually happened: {ne_game['home_team']} won by {ne_game['actual_margin']:.1f}")
    if ne_game['edge'] > 0:
        print(f"    Betting {ne_game['home_team']}: " +
              ("WIN ✓" if ne_game['actual_margin'] > ne_game['vegas_line_home'] else "LOSS ✗"))
    else:
        print(f"    Betting {ne_game['away_team']}: " +
              ("WIN ✓" if ne_game['actual_margin'] < ne_game['vegas_line_home'] else "LOSS ✗"))
