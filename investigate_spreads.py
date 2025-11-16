"""
Investigate spread line issues - are we converting correctly?
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n" + "="*80)
print("INVESTIGATING SPREAD LINE CONVERSION")
print("="*80)

# Load Week 11 2025 games
print("\nLoading Week 11 2025 games from nflverse...")
games = nflverse.games(season=2025, week=11)

# Look at the actual spread data
print("\nRaw nflverse data (first few games):")
print(games[['away_team', 'home_team', 'spread_line', 'away_score', 'home_score']].head(10))

# Check games with spreads
games_with_spreads = games[games['spread_line'].notna()].copy()
print(f"\nFound {len(games_with_spreads)} games with spread_line data")

# Let's examine one specific game
print("\n" + "="*80)
print("EXAMINING SPECIFIC EXAMPLES")
print("="*80)

# Look for NE vs NYJ game
ne_game = games_with_spreads[
    ((games_with_spreads['away_team'] == 'NE') & (games_with_spreads['home_team'] == 'NYJ')) |
    ((games_with_spreads['away_team'] == 'NYJ') & (games_with_spreads['home_team'] == 'NE'))
]

if len(ne_game) > 0:
    print("\nNE vs NYJ game:")
    print(ne_game[['away_team', 'home_team', 'spread_line', 'away_score', 'home_score']].to_string())

    row = ne_game.iloc[0]
    print(f"\nDetailed breakdown:")
    print(f"  Away team: {row['away_team']}")
    print(f"  Home team: {row['home_team']}")
    print(f"  spread_line (raw): {row['spread_line']}")
    print(f"\nInterpretation:")
    print(f"  If spread_line is from AWAY perspective:")
    print(f"    {row['away_team']} {row['spread_line']:+.1f}")
    print(f"  Converting to HOME perspective:")
    print(f"    {row['home_team']} {-1 * row['spread_line']:+.1f}")

# Show all games
print("\n" + "="*80)
print("ALL WEEK 11 GAMES WITH SPREADS")
print("="*80)
print(games_with_spreads[['away_team', 'home_team', 'spread_line', 'away_score', 'home_score']])

# Now load nfelo historical to understand the convention
print("\n" + "="*80)
print("CHECKING NFELO HISTORICAL DATA CONVENTION")
print("="*80)

nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
nfelo_hist = pd.read_csv(nfelo_url)

# Look at a few examples where we know the favorite
print("\nSample nfelo games with home_line_close:")
sample = nfelo_hist[nfelo_hist['home_line_close'].notna()].tail(20)
sample[['season', 'week', 'away', 'home']] = sample['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
print(sample[['season', 'week', 'away', 'home', 'home_line_close', 'result']].tail(10))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Let's verify the spread conventions:

1. nflverse spread_line: Need to determine if this is from away or home perspective
   - Positive values mean underdog gets points
   - Negative values mean favorite gives points

2. nfelo home_line_close: HOME team perspective
   - Negative value = home team favored (e.g., -7 means home favored by 7)
   - Positive value = home team underdog (e.g., +3 means home gets 3 points)

The question: Is nflverse spread_line from away or home perspective?
""")
