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

# Import Ball Knower modules - use unified loader and canonical features
from ball_knower.io import loaders, feature_maps
from src import config, team_mapping, models

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.max_rows', None)

print("\n" + "="*80)
print("BALL KNOWER - WEEK 11 PREDICTIONS (Using Canonical Features)")
print("="*80)

# Section 1: Load data
print("\n[1/4] Loading Week 11 data...")
all_data = loaders.load_all_sources(season=2025, week=11)

# Section 2: Get canonical feature view
print("\n[2/4] Extracting canonical features...")
canonical_ratings = feature_maps.get_canonical_features(all_data['merged_ratings'])

print(f"\nTop 10 Teams by overall_rating (canonical):")
top_teams = canonical_ratings[['team', 'overall_rating', 'epa_offense', 'epa_defense', 'offensive_rating']].sort_values('overall_rating', ascending=False).head(10)
print(top_teams.to_string(index=False))

# Section 3: Prepare matchups
print("\n[3/4] Preparing matchups...")
# Use legacy loader for weekly projections parsing (contains matchup data)
from src import data_loader
weekly_data = data_loader.load_substack_weekly_projections()
matchups_raw = weekly_data[['team_away', 'team_home', 'substack_spread_line']].copy()

# Get feature differentials using canonical features
matchup_features = feature_maps.get_feature_differential(
    canonical_ratings,
    matchups_raw['team_home'],
    matchups_raw['team_away'],
    features=['overall_rating', 'epa_margin', 'offensive_rating', 'defensive_rating']
)

# Add vegas line
matchup_features['vegas_line'] = matchups_raw['substack_spread_line'].values

# Section 4: Generate predictions with canonical features
print("\n[4/4] Generating predictions with v1.0 model (canonical features)...")
model_v1 = models.DeterministicSpreadModel(hfa=config.HOME_FIELD_ADVANTAGE)

predictions_v1 = []

for idx, game in matchup_features.iterrows():
    # Extract canonical features for home team
    home_features = {
        'overall_rating': game.get('overall_rating_home'),
        'epa_margin': game.get('epa_margin_home'),
        'offensive_rating': game.get('offensive_rating_home'),
        'defensive_rating': game.get('defensive_rating_home'),
    }

    # Extract canonical features for away team
    away_features = {
        'overall_rating': game.get('overall_rating_away'),
        'epa_margin': game.get('epa_margin_away'),
        'offensive_rating': game.get('offensive_rating_away'),
        'defensive_rating': game.get('defensive_rating_away'),
    }

    pred_spread = model_v1.predict(home_features, away_features)

    predictions_v1.append({
        'away_team': game['away_team'],
        'home_team': game['home_team'],
        'vegas_line': game['vegas_line'],
        'bk_v1_line': round(pred_spread, 1),
        'edge': round(pred_spread - game['vegas_line'], 1)
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
