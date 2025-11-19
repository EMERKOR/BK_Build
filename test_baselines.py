#!/usr/bin/env python3
"""
Test baseline betting strategies to sanity-check ATS grading logic.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

from src import config

# Load nfelo data
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
print(f"Loading nfelo data...")
df = pd.read_csv(nfelo_url)

# Extract season/week/teams
df[['season', 'week', 'away_team', 'home_team']] = \
    df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Load actual scores
schedules_path = project_root / 'schedules.parquet'
schedules = pd.read_parquet(schedules_path)
df = df.merge(
    schedules[['game_id', 'home_score', 'away_score']],
    on='game_id',
    how='left'
)

# Filter to 2020-2023
test_seasons = [2020, 2021, 2022, 2023]
test_df = df[df['season'].isin(test_seasons)].copy()

# Filter to complete data
required_cols = ['home_line_close', 'home_score', 'away_score']
mask = test_df[required_cols].notna().all(axis=1)
eval_df = test_df[mask].copy()

# Calculate actual margin and vegas line
eval_df['actual_margin'] = eval_df['home_score'] - eval_df['away_score']
eval_df['vegas_line'] = eval_df['home_line_close']

print(f"\nEvaluating {len(eval_df)} games from {test_seasons}\n")


def grade_bet(bet_side, vegas_line, actual_margin):
    """
    Grade a bet using standard ATS rules.

    Args:
        bet_side: 'home' or 'away'
        vegas_line: Spread from home perspective (negative = home favored)
        actual_margin: Actual home - away score

    Returns:
        'WIN', 'LOSS', or 'PUSH'
    """
    if bet_side == 'home':
        # Bet home at vegas_line
        # Home covers if they win by more than the spread
        # For line = -6.5, home must win by > 6.5
        # actual_margin > |vegas_line| when vegas_line < 0
        # Example: line = -6.5, need actual > 6.5
        #          line = +3.0, need actual > -3.0 (home can lose by < 3)
        if vegas_line < 0:
            # Home favored, must win by more than |line|
            if actual_margin > abs(vegas_line):
                return 'WIN'
            elif actual_margin == abs(vegas_line):
                return 'PUSH'
            else:
                return 'LOSS'
        else:
            # Home underdog, covers if loses by less than line
            if actual_margin > -vegas_line:
                return 'WIN'
            elif actual_margin == -vegas_line:
                return 'PUSH'
            else:
                return 'LOSS'
    else:  # away
        # Bet away - they get +vegas_line points
        # Away covers if home wins by less than spread (or away wins)
        if vegas_line < 0:
            # Home favored by |line|, away covers if home wins by < |line|
            # Example: line = -6.5, away covers if actual < 6.5
            if actual_margin < abs(vegas_line):
                return 'WIN'
            elif actual_margin == abs(vegas_line):
                return 'PUSH'
            else:
                return 'LOSS'
        else:
            # Home underdog, away favored
            # Away covers if home loses by more than line
            if actual_margin < -vegas_line:
                return 'WIN'
            elif actual_margin == -vegas_line:
                return 'PUSH'
            else:
                return 'LOSS'


def simulate_always_home(df):
    """Bet home every game."""
    results = []
    for _, row in df.iterrows():
        result = grade_bet('home', row['vegas_line'], row['actual_margin'])
        results.append(result)

    df['result'] = results
    wins = (df['result'] == 'WIN').sum()
    losses = (df['result'] == 'LOSS').sum()
    pushes = (df['result'] == 'PUSH').sum()

    units = wins * 1.0 - losses * 1.1
    decided = wins + losses
    win_rate = wins / decided if decided > 0 else 0
    roi = units / (decided * 1.1) if decided > 0 else 0

    return {
        'strategy': 'Always Home',
        'n_bets': len(df),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'units': units,
        'roi': roi,
    }


def simulate_always_favorite(df):
    """Bet favorite every game."""
    results = []
    for _, row in df.iterrows():
        # Negative line = home favored, positive = away favored
        if row['vegas_line'] < 0:
            bet_side = 'home'
        elif row['vegas_line'] > 0:
            bet_side = 'away'
        else:
            # Pick'em - skip
            results.append('SKIP')
            continue

        result = grade_bet(bet_side, row['vegas_line'], row['actual_margin'])
        results.append(result)

    df['result'] = results
    df = df[df['result'] != 'SKIP']

    wins = (df['result'] == 'WIN').sum()
    losses = (df['result'] == 'LOSS').sum()
    pushes = (df['result'] == 'PUSH').sum()

    units = wins * 1.0 - losses * 1.1
    decided = wins + losses
    win_rate = wins / decided if decided > 0 else 0
    roi = units / (decided * 1.1) if decided > 0 else 0

    return {
        'strategy': 'Always Favorite',
        'n_bets': len(df),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_rate': win_rate,
        'units': units,
        'roi': roi,
    }


# Run baselines
print("BASELINE STRATEGIES")
print("=" * 80)

always_home = simulate_always_home(eval_df.copy())
always_fav = simulate_always_favorite(eval_df.copy())

# Print results
for result in [always_home, always_fav]:
    print(f"\n{result['strategy']}:")
    print(f"  Bets: {result['n_bets']}")
    print(f"  Record: {result['wins']}-{result['losses']}-{result['pushes']}")
    print(f"  Win rate: {result['win_rate']:.1%}")
    print(f"  Units: {result['units']:+.1f}")
    print(f"  ROI: {result['roi']:.1%}")

print("\n" + "=" * 80)
print("\nExpected results:")
print("  - Always Home should be ~50% win rate, slightly negative ROI (vig)")
print("  - Always Favorite should be ~50% win rate, negative ROI (vig + favorite bias)")
print("=" * 80)
