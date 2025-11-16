"""
Ball Knower v1.1 - Power Ratings + Situational Adjustments

Base Model (v1.0): Power rating differential + HFA
Enhancements (v1.1):
  - Recent form adjustments (last 5 games)
  - Rest advantage
  - Potential for H2H, division, weather, injuries (future)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("BALL KNOWER v1.1 - POWER RATINGS + ADJUSTMENTS")
print("="*80)

# Load base data
print("\n[1/4] Loading Week 11 games and power ratings...")
games = nflverse.games(season=2025, week=11)
team_ratings = data_loader.merge_current_week_ratings()

print(f"Loaded {len(games)} games, {len(team_ratings)} team ratings")

# Build matchups dataframe
print("\n[2/4] Building matchup dataset...")
matchups = games[['away_team', 'home_team', 'spread_line', 'gameday']].copy()

# Merge nfelo ratings
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

matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']

# Base v1.0 model
HFA = 2.5
NFELO_WEIGHT = 0.025  # 40 ELO points = 1 spread point

matchups['base_spread'] = -HFA - (matchups['nfelo_diff'] * NFELO_WEIGHT)

# Add situational adjustments
print("\n[3/4] Adding situational adjustments...")

# ADJUSTMENT 1: Recent Form (Last 5 Games)
# If team is hot (4-1 or 5-0 L5), +0.5 points
# If team is cold (0-5 or 1-4 L5), -0.5 points

matchups['home_form_adj'] = 0.0
matchups['away_form_adj'] = 0.0

for idx, row in matchups.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']

    try:
        home_form = nflverse.get_team_form(home_team, 2025, 10)
        away_form = nflverse.get_team_form(away_team, 2025, 10)

        # Hot team bonus
        if home_form['last_5_wins'] >= 4:
            matchups.at[idx, 'home_form_adj'] = -0.5  # negative = home favored more
        elif home_form['last_5_wins'] <= 1:
            matchups.at[idx, 'home_form_adj'] = 0.5   # positive = home less favored

        if away_form['last_5_wins'] >= 4:
            matchups.at[idx, 'away_form_adj'] = 0.5   # away hot = home less favored
        elif away_form['last_5_wins'] <= 1:
            matchups.at[idx, 'away_form_adj'] = -0.5  # away cold = home more favored

    except Exception as e:
        # If form data unavailable, no adjustment
        pass

matchups['form_adj'] = matchups['home_form_adj'] + matchups['away_form_adj']

# ADJUSTMENT 2: Rest Advantage
# (Would need game dates to calculate - placeholder for now)
matchups['rest_adj'] = 0.0

# Total adjusted spread
matchups['adjusted_spread'] = matchups['base_spread'] + matchups['form_adj'] + matchups['rest_adj']

# Edge calculation
matchups['base_edge'] = matchups['base_spread'] - matchups['spread_line']
matchups['adjusted_edge'] = matchups['adjusted_spread'] - matchups['spread_line']

# Results
print("\n[4/4] Generating predictions...")

print("\n" + "="*80)
print("WEEK 11 PREDICTIONS - v1.1 WITH ADJUSTMENTS")
print("="*80)

results = matchups[[
    'away_team', 'home_team', 'spread_line',
    'base_spread', 'form_adj', 'adjusted_spread',
    'base_edge', 'adjusted_edge'
]].copy()

results = results.round(1)
results = results.sort_values('adjusted_edge', key=abs, ascending=False)

print("\n" + results.to_string(index=False))

# Summary stats
print("\n" + "="*80)
print("MODEL PERFORMANCE")
print("="*80)

print(f"\nv1.0 Base Model (Power Ratings Only):")
print(f"  Mean Absolute Edge: {results['base_edge'].abs().mean():.2f} points")
print(f"  RMSE: {np.sqrt((results['base_edge']**2).mean()):.2f} points")
print(f"  Value Bets (|edge| >= 0.5): {(results['base_edge'].abs() >= 0.5).sum()}")

print(f"\nv1.1 Adjusted Model (+ Form Adjustments):")
print(f"  Mean Absolute Edge: {results['adjusted_edge'].abs().mean():.2f} points")
print(f"  RMSE: {np.sqrt((results['adjusted_edge']**2).mean()):.2f} points")
print(f"  Value Bets (|edge| >= 0.5): {(results['adjusted_edge'].abs() >= 0.5).sum()}")

# Show top bets
print("\n" + "="*80)
print("TOP VALUE BETS (Adjusted Model)")
print("="*80)

value_bets = results[results['adjusted_edge'].abs() >= 0.5].copy()
value_bets['recommendation'] = value_bets['adjusted_edge'].apply(
    lambda x: f"Bet AWAY (+{-x:.1f})" if x > 0 else f"Bet HOME ({x:.1f})"
)

print("\n" + value_bets[['away_team', 'home_team', 'spread_line', 'adjusted_edge', 'recommendation']].to_string(index=False))

print("\n" + "="*80)
print("ADJUSTMENT BREAKDOWN")
print("="*80)

print(f"""
Current Adjustments:
  ✓ Recent Form (Last 5 Games): ±0.5 points for hot/cold teams
  - Rest Advantage: Not yet implemented (need game dates)

Potential Future Adjustments (v1.2+):
  - Head-to-Head History: ±0.5 for rivalry games
  - Division Games: ±0.5 (divisional familiarity)
  - Weather: ±1.0 (dome teams outdoors, wind, etc.)
  - Key Injuries: ±0.5-2.0 (QB, key players)
  - Travel: ±0.5 (cross-country, time zones)
  - Playoff Implications: ±0.5 (desperation factor)

The goal is to keep total adjustments in the ±2-3 point range
so the base power rating model remains the primary driver.
""")

print("\n" + "="*80 + "\n")
