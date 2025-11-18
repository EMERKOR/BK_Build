"""
LEGACY SCRIPT

This script predates the unified loader and category-first naming system.
It may still run, but is not guaranteed to follow the current BALL_KNOWER_SPEC.
Prefer using run_demo.py, predict_current_week.py, and ball_knower.io.loaders.

Original purpose: Investigate why calibration isn't working
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n" + "="*80)
print("DATA INVESTIGATION")
print("="*80)

# Load data
games = nflverse.games(season=2025, week=11)
team_ratings = data_loader.merge_current_week_ratings()

games = games[games['spread_line'].notna()].copy()
matchups = games[['away_team', 'home_team', 'spread_line']].copy()

# Merge all ratings
matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']],
    left_on='home_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={
    'nfelo': 'home_nfelo',
    'epa_margin': 'home_epa',
    'Ovr.': 'home_substack'
})

matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']],
    left_on='away_team',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={
    'nfelo': 'away_nfelo',
    'epa_margin': 'away_epa',
    'Ovr.': 'away_substack'
})

matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups['epa_diff'] = matchups['home_epa'] - matchups['away_epa']
matchups['substack_diff'] = matchups['home_substack'] - matchups['away_substack']

print("\nMatchup Data:")
print(matchups[['away_team', 'home_team', 'spread_line', 'nfelo_diff', 'epa_diff', 'substack_diff']].to_string(index=False))

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

print("\nVegas spread_line:")
print(matchups['spread_line'].describe())

print("\nnfelo_diff:")
print(matchups['nfelo_diff'].describe())

print("\nepa_diff:")
print(matchups['epa_diff'].describe())

print("\nsubstack_diff:")
print(matchups['substack_diff'].describe())

print("\n" + "="*80)
print("CORRELATIONS WITH VEGAS LINE")
print("="*80)

# Calculate correlations
print(f"\nCorrelation between spread_line and nfelo_diff: {matchups['spread_line'].corr(matchups['nfelo_diff']):.3f}")
print(f"Correlation between spread_line and epa_diff: {matchups['spread_line'].corr(matchups['epa_diff']):.3f}")
print(f"Correlation between spread_line and substack_diff: {matchups['spread_line'].corr(matchups['substack_diff']):.3f}")

print("\n" + "="*80)
print("TEAM RATINGS DISTRIBUTION")
print("="*80)

print("\nnfelo ratings:")
print(team_ratings[['team', 'nfelo']].sort_values('nfelo', ascending=False).head(10).to_string(index=False))
print(f"\nRange: {team_ratings['nfelo'].min():.1f} to {team_ratings['nfelo'].max():.1f}")
print(f"Std Dev: {team_ratings['nfelo'].std():.1f}")

print("\n" + "="*80 + "\n")
