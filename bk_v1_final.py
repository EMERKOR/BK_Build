"""
ARCHIVE FILE — uses legacy loaders by design, do not modify

Ball Knower v1.0 - Final Implementation

Uses real Vegas lines from nflverse + nfelo power ratings.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import config, data_loader
from src.live_data_fetcher import fetch_current_week_lines

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("BALL KNOWER v1.0 - FINAL")
print("="*80)

# Step 1: Get real Vegas lines
print("\n[1/4] Fetching real Vegas lines from nflverse...")
vegas_lines = fetch_current_week_lines(2025)

# Convert spread to home team perspective
# nflverse spread_line is away team perspective (positive = away underdog)
# Convert to: negative = home favored, positive = home underdog
vegas_lines['home_spread'] = -vegas_lines['spread_line']

print(f"\nVegas lines loaded: {len(vegas_lines)} games")

# Step 2: Load team ratings
print("\n[2/4] Loading team ratings...")
data = data_loader.load_all_current_week_data()
team_ratings = data_loader.merge_current_week_ratings()

print(f"Team ratings loaded: {len(team_ratings)} teams")

# Step 3: Merge ratings into games
print("\n[3/4] Building predictions...")

matchups = vegas_lines[['away_team', 'home_team', 'home_spread']].copy()

# Add team ratings
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

# Calculate nfelo differential
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']

# v1.0 Model: Rating differential + HFA
# Standard ELO conversion: ~40 ELO points = 1 point spread
# So weight = 1/40 = 0.025

HFA = 2.5
NFELO_WEIGHT = 0.025

matchups['bk_spread'] = -HFA - (matchups['nfelo_diff'] * NFELO_WEIGHT)
matchups['edge'] = matchups['bk_spread'] - matchups['home_spread']

# Step 4: Results
print("\n[4/4] Analyzing results...")

results = matchups[['away_team', 'home_team', 'home_spread', 'bk_spread', 'edge']].copy()
results['bk_spread'] = results['bk_spread'].round(1)
results['edge'] = results['edge'].round(1)

print("\n" + "="*80)
print("WEEK 11 PREDICTIONS vs REAL VEGAS LINES")
print("="*80)
print(f"\nModel: spread = -2.5 - (nfelo_diff * 0.025)")
print(f"Convention: Negative = Home Favored\n")

print(results.sort_values('edge', key=abs, ascending=False).to_string(index=False))

# Metrics
avg_edge = results['edge'].abs().mean()
max_edge = results['edge'].abs().max()
rmse = np.sqrt((results['edge'] ** 2).mean())

print(f"\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)
print(f"Average absolute edge: {avg_edge:.1f} points")
print(f"Max edge: {max_edge:.1f} points")
print(f"RMSE: {rmse:.1f} points")

value_bets = results[results['edge'].abs() >= 2.0].copy()
print(f"\nValue bets (|edge| >= 2.0 pts): {len(value_bets)}")

if len(value_bets) > 0:
    print("\n" + "="*80)
    print("VALUE BETS")
    print("="*80 + "\n")
    value_bets['recommendation'] = value_bets.apply(
        lambda r: f"Bet {r['home_team']}" if r['edge'] < 0 else f"Bet {r['away_team']}",
        axis=1
    )
    print(value_bets[['away_team', 'home_team', 'home_spread', 'bk_spread', 'edge', 'recommendation']].to_string(index=False))

print("\n" + "="*80 + "\n")
