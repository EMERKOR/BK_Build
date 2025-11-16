"""
Enhanced Analysis Demo - Using NFLVerse Data

Demonstrates how to use team form, matchup history, and other nflverse
data to inform betting decisions beyond just power ratings.
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("ENHANCED ANALYSIS - Week 11 2025")
print("="*80)

# Get Week 11 games
print("\n[1/3] Loading Week 11 matchups...")
games = nflverse.games(season=2025, week=11)

print(f"Found {len(games)} games")

# Select a few interesting matchups to analyze
interesting_games = [
    ('KC', 'DEN'),
    ('DET', 'PHI'),
    ('BUF', 'TB'),
]

print("\n" + "="*80)
print("DETAILED MATCHUP ANALYSIS")
print("="*80)

for away, home in interesting_games:
    print(f"\n{'='*80}")
    print(f"{away} @ {home}")
    print(f"{'='*80}")

    # Get current season form
    print(f"\n{home} Form (through Week 10):")
    home_form = nflverse.get_team_form(home, 2025, 10)
    for key, val in home_form.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.1f}")
        else:
            print(f"  {key:20s}: {val}")

    print(f"\n{away} Form (through Week 10):")
    away_form = nflverse.get_team_form(away, 2025, 10)
    for key, val in away_form.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.1f}")
        else:
            print(f"  {key:20s}: {val}")

    # Compare key metrics
    print(f"\nKey Comparisons:")
    print(f"  Win % - {home}: {home_form['wins']/home_form['games_played']:.1%} vs {away}: {away_form['wins']/away_form['games_played']:.1%}")
    print(f"  PPG - {home}: {home_form['ppg']:.1f} vs {away}: {away_form['ppg']:.1f}")
    print(f"  Margin - {home}: {home_form['margin']:.1f} vs {away}: {away_form['margin']:.1f}")
    print(f"  L5 Form - {home}: {home_form['last_5_wins']}-{5-home_form['last_5_wins']} vs {away}: {away_form['last_5_wins']}-{5-away_form['last_5_wins']}")

    # Get head-to-head history
    print(f"\nHead-to-Head History (Last 5 games):")
    history = nflverse.get_matchup_history(home, away, 5)

    if len(history) > 0:
        print(history.to_string(index=False))

        # Calculate H2H stats
        h2h_home_wins = 0
        h2h_total_over = 0

        for _, game in history.iterrows():
            if game['home_team'] == home:
                if game['home_score'] > game['away_score']:
                    h2h_home_wins += 1
                total = game['home_score'] + game['away_score']
            else:
                if game['away_score'] > game['home_score']:
                    h2h_home_wins += 1
                total = game['home_score'] + game['away_score']
            h2h_total_over += total

        print(f"\n  H2H: {home} {h2h_home_wins}-{len(history)-h2h_home_wins} vs {away}")
        print(f"  Avg Total: {h2h_total_over/len(history):.1f}")
    else:
        print("  No recent matchups found")

# Show Vegas lines for Week 11
print("\n" + "="*80)
print("WEEK 11 VEGAS LINES")
print("="*80)

lines = games[['away_team', 'home_team', 'spread_line', 'total_line', 'gameday']].copy()
lines = lines.sort_values('gameday')
print("\n" + lines.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS INSIGHTS")
print("="*80)

print("""
This enhanced data allows you to:

1. **Team Form Analysis**: Recent performance trends (last 5 games)
2. **Head-to-Head History**: How teams match up specifically against each other
3. **Season Trends**: PPG, defensive performance, home/away splits
4. **Context for Power Ratings**: Why nfelo/Substack ratings might be high/low

Next Steps for Model Enhancement (v1.1):
- Incorporate recent form as adjustment factor (+/- 1-2 points)
- Use H2H history for divisional/rivalry games
- Add rest days, injuries, weather data
- Build "situational" adjustments on top of base power rating model
""")

print("\n" + "="*80 + "\n")
