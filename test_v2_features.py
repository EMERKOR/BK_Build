"""Quick test of Ball Knower v2.0 feature engineering"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from feature_engineering_v2 import ComprehensiveFeatureBuilder

print("="*80)
print("Testing Ball Knower v2.0 Feature Engineering")
print("="*80)

# Initialize builder
builder = ComprehensiveFeatureBuilder(data_dir='.')
builder.load_all_data()

# Test single game feature building
print("\nTesting single game (2024 Week 5, KC vs NO)...")
features = builder.build_game_features(
    season=2024,
    week=5,
    home_team='NO',
    away_team='KC'
)

print(f"\nFeatures built: {len(features)}")
print("\nSample features:")
for i, (key, value) in enumerate(list(features.items())[:15]):
    print(f"  {key:35s}: {value}")

print("\n✓ Feature building works!")

# Test building small training set (just 2024 Week 5-7)
print("\n" + "="*80)
print("Testing small training dataset (2024 weeks 5-7)")
print("="*80)

import pandas as pd
schedules = pd.read_parquet('schedules.parquet')
test_games = schedules[
    (schedules['season'] == 2024) &
    (schedules['week'].isin([5, 6, 7])) &
    (schedules['game_type'] == 'REG') &
    (schedules['home_score'].notna())
]

print(f"\nProcessing {len(test_games)} games...")

all_features = []
for idx, game in test_games.iterrows():
    features = builder.build_game_features(
        season=game['season'],
        week=game['week'],
        home_team=game['home_team'],
        away_team=game['away_team']
    )

    # Add identifiers and targets
    features['game_id'] = game['game_id']
    features['actual_margin'] = game['home_score'] - game['away_score']
    features['spread_line'] = game['spread_line']

    all_features.append(features)

df = pd.DataFrame(all_features)
print(f"\n✓ Dataset built: {df.shape}")
print(f"  Columns: {len(df.columns)}")
print(f"  Rows: {len(df)}")

# Check for missing data
print("\nMissing data check:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing)
else:
    print("  No missing data!")

print("\n" + "="*80)
print("Feature engineering test complete!")
print("="*80)
