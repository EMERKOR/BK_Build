"""
Diagnose why the model is failing so badly on Week 11 2025
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

print("\n" + "="*80)
print("DIAGNOSING MODEL FAILURE")
print("="*80)

# Load model
model_file = Path('/home/user/BK_Build/output/ball_knower_v1_2_model.json')
with open(model_file, 'r') as f:
    model_params = json.load(f)

print("\nModel was trained on 2009-2024 data to predict Vegas closing lines")
print(f"Training MAE: {model_params['train_mae']:.2f} points")
print(f"Test MAE (2025 games before Week 11): {model_params['test_mae']:.2f} points")

# Load Week 11 games
games = nflverse.games(season=2025, week=11)
games = games[games['spread_line'].notna()].copy()

# Load nfelo ratings
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'
nfelo_ratings = pd.read_csv(nfelo_url)

print("\n" + "="*80)
print("NFELO RATINGS CHECK")
print("="*80)

print(f"\nLoaded {len(nfelo_ratings)} teams from nfelo snapshot")
print("\nFirst few teams:")
print(nfelo_ratings.head(10))

# Check for duplicates
print(f"\nDuplicate teams: {nfelo_ratings['team'].duplicated().sum()}")
if nfelo_ratings['team'].duplicated().sum() > 0:
    print("Duplicates found:")
    print(nfelo_ratings[nfelo_ratings['team'].duplicated(keep=False)].sort_values('team'))

# Check if Week 11 teams exist in nfelo
all_teams = set(games['home_team']) | set(games['away_team'])
print(f"\nWeek 11 teams: {sorted(all_teams)}")

missing_teams = all_teams - set(nfelo_ratings['team'].unique())
if missing_teams:
    print(f"Teams missing from nfelo: {missing_teams}")
else:
    print("All Week 11 teams found in nfelo ✓")

# Check nfelo rating distribution
print("\n" + "="*80)
print("NFELO RATING DISTRIBUTION")
print("="*80)

print(f"\nNFELO rating stats:")
print(f"  Mean: {nfelo_ratings['nfelo'].mean():.1f}")
print(f"  Std: {nfelo_ratings['nfelo'].std():.1f}")
print(f"  Min: {nfelo_ratings['nfelo'].min():.1f} ({nfelo_ratings.loc[nfelo_ratings['nfelo'].idxmin(), 'team']})")
print(f"  Max: {nfelo_ratings['nfelo'].max():.1f} ({nfelo_ratings.loc[nfelo_ratings['nfelo'].idxmax(), 'team']})")

# For NE vs NYJ, show the nfelo ratings
ne_game = games[games['home_team'] == 'NE'].iloc[0]
ne_rating = nfelo_ratings[nfelo_ratings['team'] == 'NE']['nfelo'].values[0]
nyj_rating = nfelo_ratings[nfelo_ratings['team'] == 'NYJ']['nfelo'].values[0]

print("\n" + "="*80)
print("EXAMPLE: NE vs NYJ")
print("="*80)

print(f"\nNFELO Ratings:")
print(f"  NE: {ne_rating:.1f}")
print(f"  NYJ: {nyj_rating:.1f}")
print(f"  Diff: {ne_rating - nyj_rating:.1f}")

print(f"\nModel coefficients:")
print(f"  Intercept: {model_params['intercept']:.3f}")
print(f"  nfelo_diff: {model_params['coefficients']['nfelo_diff']:.6f}")

# Calculate what the model would predict
nfelo_diff = ne_rating - nyj_rating
model_pred_from_nfelo_alone = model_params['intercept'] + (nfelo_diff * model_params['coefficients']['nfelo_diff'])

print(f"\nModel prediction (nfelo_diff only): {model_pred_from_nfelo_alone:.2f}")
print(f"Vegas line: NE -12.5")
print(f"Actual result: NE won by 13")

# Now check historical nfelo data to see what typical values are
print("\n" + "="*80)
print("CHECKING HISTORICAL NFELO DATA")
print("="*80)

nfelo_hist_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
nfelo_hist = pd.read_csv(nfelo_hist_url)

# Filter to recent data with complete info
recent = nfelo_hist[
    (nfelo_hist['starting_nfelo_home'].notna()) &
    (nfelo_hist['starting_nfelo_away'].notna()) &
    (nfelo_hist['home_line_close'].notna())
].copy()

# Extract season from game_id
recent['season'] = recent['game_id'].str.extract(r'(\d{4})')[0].astype(int)
recent['nfelo_diff'] = recent['starting_nfelo_home'] - recent['starting_nfelo_away']

# Compare 2025 vs historical
recent_2025 = recent[recent['season'] == 2025]
recent_2024 = recent[recent['season'] == 2024]

print(f"\nNFELO diff statistics:")
print(f"\n2024 season:")
print(f"  Mean: {recent_2024['nfelo_diff'].mean():.2f}")
print(f"  Std: {recent_2024['nfelo_diff'].std():.2f}")
print(f"  Games: {len(recent_2024)}")

print(f"\n2025 season:")
print(f"  Mean: {recent_2025['nfelo_diff'].mean():.2f}")
print(f"  Std: {recent_2025['nfelo_diff'].std():.2f}")
print(f"  Games: {len(recent_2025)}")

print("\n" + "="*80)
print("RELATIONSHIP: NFELO_DIFF vs VEGAS LINE (Historical)")
print("="*80)

# Check correlation
correlation = recent['nfelo_diff'].corr(recent['home_line_close'])
print(f"\nCorrelation between nfelo_diff and home_line_close: {correlation:.3f}")

# Show some examples
sample = recent[recent['season'] == 2024].sample(min(10, len(recent_2024)), random_state=42)
print("\nSample games from 2024:")
print(sample[['game_id', 'nfelo_diff', 'home_line_close']].to_string(index=False))

# Calculate what the coefficient should be roughly
print("\n" + "="*80)
print("EXPECTED NFELO_DIFF COEFFICIENT")
print("="*80)

# Simple linear regression
from sklearn.linear_model import LinearRegression
X = recent_2024[['nfelo_diff']]
y = recent_2024['home_line_close']
lr = LinearRegression()
lr.fit(X, y)

print(f"\nSimple linear regression (2024 data):")
print(f"  Coefficient: {lr.coef_[0]:.6f}")
print(f"  Intercept: {lr.intercept_:.3f}")

print(f"\nCompare to v1.2 model:")
print(f"  Coefficient: {model_params['coefficients']['nfelo_diff']:.6f}")
print(f"  Intercept: {model_params['intercept']:.3f}")

if abs(lr.coef_[0] - model_params['coefficients']['nfelo_diff']) > 0.001:
    print("\n⚠️  WARNING: Coefficients differ significantly!")
    print("   This could indicate the model was trained on different data")
