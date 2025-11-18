"""
Current Week Predictions - Ball Knower v1.2

Generates betting recommendations using the trained v1.2 model.

Model Performance (trained on 2009-2024):
- Test R² = 0.884
- Test MAE = 1.57 points
- Trained on 4,345 historical games
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import config, betting_utils

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate Ball Knower v1.2 predictions for a given week')
parser.add_argument('--season', type=int, default=config.CURRENT_SEASON,
                    help=f'NFL season year (default: {config.CURRENT_SEASON})')
parser.add_argument('--week', type=int, required=True,
                    help='Week number (required)')
args = parser.parse_args()

SEASON = args.season
WEEK = args.week

print("\n" + "="*80)
print(f"BALL KNOWER v1.2 - WEEK {WEEK} {SEASON} PREDICTIONS")
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

print(f"\n[1/4] Loading Week {WEEK} {SEASON} data...")

# Load games
games = nflverse.games(season=SEASON, week=WEEK)
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

# CRITICAL FIX: nflverse spread_line is from AWAY team perspective
# But v1.2 was trained on nfelo home_line_close (HOME perspective)
# Convert: home_line = -1 * away_line
matchups['spread_line'] = -1 * matchups['spread_line']

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
# BETTING ANALYTICS
# ============================================================================

print("\n[3.5/4] Calculating betting metrics (probabilities, EV, Kelly)...")

# Use model test residual std for probability calibration (from model params)
residual_std = 1.88  # From v1.2 test set

# Convert spread predictions to win probabilities
matchups['home_win_prob'] = matchups['bk_v1_2_spread'].apply(
    lambda x: betting_utils.spread_to_win_probability(x, residual_std=residual_std)
)
matchups['away_win_prob'] = 1 - matchups['home_win_prob']

# Determine bet side (which side has the edge)
matchups['bet_side'] = matchups['edge'].apply(lambda x: 'HOME' if x < 0 else 'AWAY')
matchups['bet_prob'] = matchups.apply(
    lambda row: row['home_win_prob'] if row['bet_side'] == 'HOME' else row['away_win_prob'],
    axis=1
)

# Assume standard -110 odds for spread betting
standard_odds = -110

# Calculate Kelly sizing (quarter Kelly recommended)
matchups['kelly_full'] = matchups.apply(
    lambda row: betting_utils.kelly_criterion(
        model_prob=row['bet_prob'],
        odds=standard_odds
    ),
    axis=1
)
matchups['kelly_quarter'] = matchups['kelly_full'] * 0.25

# Calculate EV (simplified - assumes -110 on all bets)
matchups['ev_per_100'] = matchups.apply(
    lambda row: betting_utils.calculate_ev(
        model_prob=row['bet_prob'],
        market_prob=0.5,  # Simplified assumption for spread bets
        odds=standard_odds,
        stake=100
    ),
    axis=1
)

print(f"✓ Added win probabilities and Kelly sizing")

# ============================================================================
# IDENTIFY VALUE BETS
# ============================================================================

print("\n[4/4] Identifying value bets...")

print("\n" + "="*80)
print(f"ALL WEEK {WEEK} PREDICTIONS")
print("="*80)

display_cols = ['away_team', 'home_team', 'spread_line', 'bk_v1_2_spread',
                'edge', 'home_win_prob', 'kelly_quarter', 'ev_per_100']

all_preds = matchups[display_cols].copy()
all_preds = all_preds.sort_values('edge', ascending=False)

# Format for display
all_preds_display = all_preds.copy()
all_preds_display['home_win_prob'] = (all_preds_display['home_win_prob'] * 100).round(1)
all_preds_display['kelly_quarter'] = (all_preds_display['kelly_quarter'] * 100).round(2)
all_preds_display[['spread_line', 'bk_v1_2_spread', 'edge', 'ev_per_100']] = \
    all_preds_display[['spread_line', 'bk_v1_2_spread', 'edge', 'ev_per_100']].round(2)

# Rename for clarity
all_preds_display.columns = ['Away', 'Home', 'Vegas', 'BK_v1.2', 'Edge',
                              'Home_Win_%', '1/4_Kelly_%', 'EV_per_$100']

print("\n" + all_preds_display.to_string(index=False))

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
    value_bets['bet_team'] = value_bets.apply(
        lambda row: row['home_team'] if row['bet_side'] == 'HOME' else row['away_team'],
        axis=1
    )

    value_bets['recommendation'] = value_bets.apply(
        lambda row: f"Bet {row['bet_team']} {row['spread_line']:.1f} (edge: {abs(row['edge']):.1f})",
        axis=1
    )

    # Format for display
    bet_display = value_bets[[
        'away_team', 'home_team', 'spread_line', 'bk_v1_2_spread',
        'edge', 'bet_side', 'kelly_quarter', 'ev_per_100', 'recommendation'
    ]].copy()

    bet_display = bet_display.sort_values('edge', key=abs, ascending=False)

    # Format percentages
    bet_display_formatted = bet_display.copy()
    bet_display_formatted['kelly_pct'] = (bet_display_formatted['kelly_quarter'] * 100).round(2)
    bet_display_formatted = bet_display_formatted.drop(columns=['kelly_quarter'])
    bet_display_formatted = bet_display_formatted.round(2)

    print(f"\n{len(value_bets)} recommended bets:\n")
    print(bet_display_formatted.to_string(index=False))

    # Risk analysis
    print("\n" + "="*80)
    print("BETTING RECOMMENDATIONS - PROFESSIONAL ANALYSIS")
    print("="*80)

    print(f"\nTotal recommended bets: {len(value_bets)}\n")

    for bet_idx, bet in bet_display.iterrows():
        edge_mag = abs(bet['edge'])
        kelly_pct = bet['kelly_quarter'] * 100

        if edge_mag >= 3.0:
            confidence = "HIGH"
        elif edge_mag >= 2.5:
            confidence = "MODERATE-HIGH"
        elif edge_mag >= 2.0:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        print(f"  {bet['away_team']} @ {bet['home_team']}")
        print(f"    {bet['recommendation']}")
        print(f"    Confidence: {confidence} | 1/4 Kelly: {kelly_pct:.2f}% | EV: ${bet['ev_per_100']:.2f}/bet")
        print()

else:
    print(f"\nNo games meet the {value_threshold}+ point threshold this week.")
    print(f"Largest edge: {matchups['abs_edge'].max():.2f} points")

# Model comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = matchups[['away_team', 'home_team', 'spread_line', 'bk_v1_2_spread', 'edge']].copy()

print(f"\nv1.2 Performance on Week {WEEK}:")
print(f"  Mean absolute edge: {matchups['abs_edge'].mean():.2f} points")
print(f"  Median edge: {matchups['edge'].median():.2f} points")
print(f"  Std dev: {matchups['edge'].std():.2f} points")

# Save predictions
output_file = config.OUTPUT_DIR / f'week_{WEEK}_value_bets_v1_2.csv'
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

6. PROBABILITY AND KELLY:
   - Win probabilities derived from spread predictions
   - 1/4 Kelly sizing is conservative and recommended
   - Never bet more than Kelly suggests
   - Adjust for model uncertainty and bankroll risk tolerance
""")

print("="*80 + "\n")
