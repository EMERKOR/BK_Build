"""
ARCHIVE FILE — uses legacy loaders by design, do not modify

Ball Knower v1.0 - CORRECT Implementation

Deterministic spread model using power rating differentials.
spread_pred = rating_diff + HFA

Inputs:
- nfelo power ratings (already normalized to standard scale)
- Substack overall ratings (offensive + defensive combined)
- Home field advantage (calibrated)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ball_knower import config
from src import data_loader

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("BALL KNOWER v1.0 - CORRECT REBUILD")
print("="*80)

# Load data
print("\n[1/3] Loading Week 11 data...")
data = data_loader.load_all_current_week_data()

# Get team ratings
team_ratings = data_loader.merge_current_week_ratings()

# Get weekly matchups
matchups = data['substack_weekly'][['team_away', 'team_home', 'substack_spread_line']].copy()

print(f"\nTeam ratings loaded: {len(team_ratings)} teams")
print(f"Matchups loaded: {len(matchups)} games")

# Merge team ratings
print("\n[2/3] Merging team ratings into matchups...")

matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'Ovr.']],
    left_on='team_home',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={'nfelo': 'home_nfelo', 'Ovr.': 'home_substack'})

matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'Ovr.']],
    left_on='team_away',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={'nfelo': 'away_nfelo', 'Ovr.': 'away_substack'})

# Calculate rating differentials
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups['substack_diff'] = matchups['home_substack'] - matchups['away_substack']

# v1.0 Model: Simple weighted combination
print("\n[3/3] Building v1.0 deterministic model...")

# These weights need calibration on historical data
# For now, using research-based estimates:
# nfelo: ~25 point difference = ~1 point spread
# Substack: direct point ratings
# HFA: 2.5 points

NFELO_WEIGHT = 0.04      # 25 nfelo points = 1 spread point
SUBSTACK_WEIGHT = 0.4    # Substack already scaled to points
HFA = 2.5

matchups['bk_v1_spread'] = (
    -HFA +                                    # Start with HFA (negative = home favored)
    -(matchups['nfelo_diff'] * NFELO_WEIGHT) +      # nfelo contribution
    -(matchups['substack_diff'] * SUBSTACK_WEIGHT)  # Substack contribution
)

matchups['edge'] = matchups['bk_v1_spread'] - matchups['substack_spread_line']

# Results
print("\n" + "="*80)
print("WEEK 11 PREDICTIONS - v1.0 DETERMINISTIC MODEL")
print("="*80)
print(f"\nModel Formula:")
print(f"  spread = -HFA - (nfelo_diff * {NFELO_WEIGHT}) - (substack_diff * {SUBSTACK_WEIGHT})")
print(f"  HFA = {HFA}")

results = matchups[['team_away', 'team_home', 'substack_spread_line', 'bk_v1_spread', 'edge']].copy()
results['bk_v1_spread'] = results['bk_v1_spread'].round(1)
results['edge'] = results['edge'].round(1)

print("\n" + results.sort_values('edge', key=abs, ascending=False).to_string(index=False))

# Show average absolute edge
avg_edge = results['edge'].abs().mean()
print(f"\nAverage absolute edge: {avg_edge:.1f} points")

# Value bets
value_bets = results[results['edge'].abs() >= 0.5].copy()
print(f"\nValue bets (edge >= 0.5): {len(value_bets)}")

print("\n" + "="*80)
print("NOTE: This is v1.0 baseline using uncalibrated weights.")
print("Weights should be fitted on historical data to match Vegas lines.")
print("Expected average edge should be 2-3 points, not 10+")
print("="*80 + "\n")
