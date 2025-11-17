"""
Ball Knower Demo Script
Run Week 11 predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Ball Knower modules - use unified loader
from ball_knower.io import loaders
from src import config, team_mapping, models

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', None)

print("\n" + "="*80)
print("BALL KNOWER - WEEK 11 PREDICTIONS")
print("="*80)

# Section 1: Load data
print("\n[1/4] Loading Week 11 data...")
all_data = loaders.load_all_sources(season=2025, week=11)

# Section 2: Merge team ratings
print("\n[2/4] Merging team ratings...")
team_ratings = all_data['merged_ratings']

print(f"\nTop 10 Teams by nfelo:")
print(team_ratings[['team', 'nfelo', 'epa_off', 'epa_def', 'Ovr.']].sort_values('nfelo', ascending=False).head(10).to_string(index=False))

# Section 3: Prepare matchups
print("\n[3/4] Preparing matchups...")
# Note: weekly projections contain matchup data, but need to extract team matchups
# For now, we'll need to adapt this - the unified loader returns raw DataFrames
# We'll use the legacy data_loader for weekly projections parsing until we enhance the unified loader
from src import data_loader
weekly_data = data_loader.load_substack_weekly_projections()
matchups = weekly_data[['team_away', 'team_home', 'substack_spread_line']].copy()

# Add home team ratings
matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']],
    left_on='team_home',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={
    'nfelo': 'nfelo_home',
    'epa_margin': 'epa_margin_home',
    'Ovr.': 'Ovr._home'
})

# Add away team ratings
matchups = matchups.merge(
    team_ratings[['team', 'nfelo', 'epa_margin', 'Ovr.']],
    left_on='team_away',
    right_on='team',
    how='left'
).drop(columns=['team']).rename(columns={
    'nfelo': 'nfelo_away',
    'epa_margin': 'epa_margin_away',
    'Ovr.': 'Ovr._away'
})

# Section 4: Generate predictions
print("\n[4/4] Generating predictions with v1.0 model...")
model_v1 = models.DeterministicSpreadModel(hfa=config.HOME_FIELD_ADVANTAGE)

predictions_v1 = []

for idx, game in matchups.iterrows():
    home_features = {
        'nfelo': game.get('nfelo_home'),
        'epa_margin': game.get('epa_margin_home'),
        'Ovr.': game.get('Ovr._home')
    }

    away_features = {
        'nfelo': game.get('nfelo_away'),
        'epa_margin': game.get('epa_margin_away'),
        'Ovr.': game.get('Ovr._away')
    }

    pred_spread = model_v1.predict(home_features, away_features)

    predictions_v1.append({
        'away_team': game['team_away'],
        'home_team': game['team_home'],
        'vegas_line': game['substack_spread_line'],
        'bk_v1_line': round(pred_spread, 1),
        'edge': round(pred_spread - game['substack_spread_line'], 1)
    })

predictions_df = pd.DataFrame(predictions_v1)

# Display all predictions
print("\n" + "="*80)
print("WEEK 11 PREDICTIONS (Sorted by Edge)")
print("="*80)
print("\nSpread Convention: Negative = Home Favored, Positive = Home Underdog")
print("Edge = BK Prediction - Vegas Line\n")

print(predictions_df.sort_values('edge', key=abs, ascending=False).to_string(index=False))

# Value bets
value_bets = predictions_df[predictions_df['edge'].abs() >= config.MIN_BET_EDGE].copy()

value_bets['recommendation'] = value_bets.apply(
    lambda row: f"Bet {row['home_team']}" if row['edge'] < 0 else f"Bet {row['away_team']}",
    axis=1
)

print("\n" + "="*80)
print(f"VALUE BETS (Edge >= {config.MIN_BET_EDGE} pts)")
print("="*80 + "\n")

if len(value_bets) > 0:
    print(value_bets[['away_team', 'home_team', 'vegas_line', 'bk_v1_line', 'edge', 'recommendation']].sort_values('edge', key=abs, ascending=False).to_string(index=False))
    print(f"\nFound {len(value_bets)} value bets")
else:
    print("No value bets found with current threshold")

print("\n" + "="*80)
print("DONE")
print("="*80 + "\n")
