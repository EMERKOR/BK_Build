"""
Investigate why calibration isn't working
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from ball_knower.io import loaders, feature_maps

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n" + "="*80)
print("DATA INVESTIGATION (Using Canonical Features)")
print("="*80)

# Load data using unified loader
games = nflverse.games(season=2025, week=11)
all_data = loaders.load_all_sources(season=2025, week=11)

# Get canonical feature view (provider-agnostic)
canonical_ratings = feature_maps.get_canonical_features(
    all_data['merged_ratings'],
    features=['overall_rating', 'epa_margin', 'offensive_rating', 'defensive_rating']
)

games = games[games['spread_line'].notna()].copy()

# Use feature_maps utility to get differentials
matchups = feature_maps.get_feature_differential(
    canonical_ratings,
    games['home_team'],
    games['away_team'],
    features=['overall_rating', 'epa_margin', 'offensive_rating']
)

# Add spread_line back
matchups['spread_line'] = games['spread_line'].values

print("\nMatchup Data (Canonical Features):")
print(matchups[['away_team', 'home_team', 'spread_line',
               'overall_rating_diff', 'epa_margin_diff', 'offensive_rating_diff']].to_string(index=False))

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS (Canonical Feature Differentials)")
print("="*80)

print("\nVegas spread_line:")
print(matchups['spread_line'].describe())

print("\noverall_rating_diff (canonical):")
print(matchups['overall_rating_diff'].describe())

print("\nepa_margin_diff (canonical):")
print(matchups['epa_margin_diff'].describe())

print("\noffensive_rating_diff (canonical):")
print(matchups['offensive_rating_diff'].describe())

print("\n" + "="*80)
print("CORRELATIONS WITH VEGAS LINE (Canonical Features)")
print("="*80)

# Calculate correlations
print(f"\nCorrelation between spread_line and overall_rating_diff: {matchups['spread_line'].corr(matchups['overall_rating_diff']):.3f}")
print(f"Correlation between spread_line and epa_margin_diff: {matchups['spread_line'].corr(matchups['epa_margin_diff']):.3f}")
print(f"Correlation between spread_line and offensive_rating_diff: {matchups['spread_line'].corr(matchups['offensive_rating_diff']):.3f}")

print("\n" + "="*80)
print("TEAM RATINGS DISTRIBUTION (Canonical Features)")
print("="*80)

print("\nOverall ratings (canonical):")
print(canonical_ratings[['team', 'overall_rating']].sort_values('overall_rating', ascending=False).head(10).to_string(index=False))
print(f"\nRange: {canonical_ratings['overall_rating'].min():.1f} to {canonical_ratings['overall_rating'].max():.1f}")
print(f"Std Dev: {canonical_ratings['overall_rating'].std():.1f}")

print("\n" + "="*80)
print("CANONICAL FEATURE AVAILABILITY REPORT")
print("="*80)
feature_maps.print_feature_availability(all_data['merged_ratings'])

print("="*80 + "\n")
