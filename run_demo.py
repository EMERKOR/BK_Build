"""
Ball Knower Demo Script
Run Week 11 predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

# Import Ball Knower modules - use unified loader
from ball_knower.io import loaders
from src import config, team_mapping, models

# Data directory and current week settings
DATA_DIR = PROJECT_ROOT / "data" / "current_season"
CURRENT_SEASON = config.CURRENT_SEASON
CURRENT_WEEK = config.CURRENT_WEEK

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', None)

print("\n" + "="*80)
print("BALL KNOWER - WEEK 11 PREDICTIONS")
print("="*80)

# Section 1: Load data
print("\n[1/4] Loading Week 11 data...")
all_data = loaders.load_all_sources(season=CURRENT_SEASON, week=CURRENT_WEEK, data_dir=DATA_DIR)

# Section 2: Merge team ratings
print("\n[2/4] Merging team ratings...")
team_ratings = all_data['merged_ratings']

# Compute derived EPA metrics for compatibility
if 'EPA/Play' in team_ratings.columns and 'EPA/Play Against' in team_ratings.columns:
    team_ratings['epa_off'] = team_ratings['EPA/Play']
    team_ratings['epa_def'] = team_ratings['EPA/Play Against']
    team_ratings['epa_margin'] = team_ratings['EPA/Play'] - team_ratings['EPA/Play Against']

print(f"\nTop 10 Teams by nfelo:")
print(team_ratings[['team', 'nfelo', 'epa_off', 'epa_def', 'Ovr.']].sort_values('nfelo', ascending=False).head(10).to_string(index=False))

# Section 3: Prepare matchups
print("\n[3/4] Preparing matchups...")
# Parse weekly projections to extract matchup data
weekly_raw = all_data['weekly_projections_ppg_substack']

# Parse matchups from "Matchup" column (format: "Team1 at Team2" or "Team1 vs Team2")
def parse_matchup(matchup):
    if ' at ' in matchup:
        teams = matchup.split(' at ')
        return pd.Series({'team_away_full': teams[0], 'team_home_full': teams[1]})
    elif ' vs ' in matchup:
        teams = matchup.split(' vs ')
        return pd.Series({'team_away_full': teams[0], 'team_home_full': teams[1]})
    else:
        return pd.Series({'team_away_full': None, 'team_home_full': None})

weekly_raw[['team_away_full', 'team_home_full']] = weekly_raw['Matchup'].apply(parse_matchup)

# Normalize team names
weekly_raw['team_away'] = weekly_raw['team_away_full'].apply(team_mapping.normalize_team_name)
weekly_raw['team_home'] = weekly_raw['team_home_full'].apply(team_mapping.normalize_team_name)

# Parse spread from "Favorite" column (e.g., "ATL -5.5")
weekly_raw['substack_spread_line'] = weekly_raw['Favorite'].str.extract(r'([-+]?\d+\.?\d*)')[0].astype(float)

matchups = weekly_raw[['team_away', 'team_home', 'substack_spread_line']].copy()

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

# Sanity check: verify data loaded correctly
print("Demo merged ratings shape:", team_ratings.shape)
print("Demo merged sample columns:", list(team_ratings.columns)[:12])
