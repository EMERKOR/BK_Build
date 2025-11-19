#!/usr/bin/env python3
"""
Debug script to examine individual ATS bets and verify logic.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

from ball_knower.benchmarks.v1_comparison import compare_v1_models
from src import config

# Load nfelo data
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
print(f"Loading nfelo data from {nfelo_url}...")
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

# Run v1.0 model
NFELO_COEF = 0.0447
INTERCEPT = 2.67

required_cols = ['starting_nfelo_home', 'starting_nfelo_away', 'home_line_close', 'home_score', 'away_score']
mask = test_df[required_cols].notna().all(axis=1)
eval_df = test_df[mask].copy()

# Calculate predictions
eval_df['nfelo_diff'] = eval_df['starting_nfelo_home'] - eval_df['starting_nfelo_away']
eval_df['bk_spread'] = INTERCEPT + (eval_df['nfelo_diff'] * NFELO_COEF)

# Calculate actuals
eval_df['actual_margin'] = eval_df['home_score'] - eval_df['away_score']
eval_df['vegas_line'] = eval_df['home_line_close']

# Calculate edge
edge_threshold = 1.5
eval_df['edge'] = eval_df['bk_spread'] - eval_df['vegas_line']
eval_df['abs_edge'] = eval_df['edge'].abs()

# Filter to bets
bets_df = eval_df[eval_df['abs_edge'] >= edge_threshold].copy()

# Determine bet side and result
def evaluate_bet_debug(row):
    """Evaluate bet and return detailed info."""
    edge = row['edge']
    vegas_line = row['vegas_line']
    actual_margin = row['actual_margin']

    if edge < 0:
        # Bet home
        bet_side = 'home'
        bet_spread = vegas_line
        # Home covers if actual_margin < vegas_line (more negative = bigger win)
        if actual_margin < vegas_line:
            bet_result = 'WIN'
        elif actual_margin == vegas_line:
            bet_result = 'PUSH'
        else:
            bet_result = 'LOSS'
        cover_diff = vegas_line - actual_margin  # Positive if covered
    else:
        # Bet away
        bet_side = 'away'
        bet_spread = -vegas_line  # Away gets opposite
        # Away covers if actual_margin > vegas_line (less negative/more positive)
        if actual_margin > vegas_line:
            bet_result = 'WIN'
        elif actual_margin == vegas_line:
            bet_result = 'PUSH'
        else:
            bet_result = 'LOSS'
        cover_diff = actual_margin - vegas_line  # Positive if covered

    return pd.Series({
        'bet_side': bet_side,
        'bet_spread': bet_spread,
        'bet_result': bet_result,
        'cover_diff': cover_diff
    })

bets_df[['bet_side', 'bet_spread', 'bet_result', 'cover_diff']] = \
    bets_df.apply(evaluate_bet_debug, axis=1)

# Sample 20 random bets
print(f"\nTotal bets placed: {len(bets_df)}")
print(f"Sample of 20 random bets:\n")

sample = bets_df.sample(n=min(20, len(bets_df)), random_state=42)

# Create display columns
display_cols = [
    'season', 'week', 'home_team', 'away_team',
    'vegas_line', 'bk_spread', 'actual_margin', 'edge',
    'bet_side', 'bet_spread', 'bet_result', 'cover_diff'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

print(sample[display_cols].to_string(index=False))

# Summary stats
print(f"\n\nSUMMARY STATISTICS")
print(f"=" * 80)
print(f"Total bets: {len(bets_df)}")
print(f"Wins: {(bets_df['bet_result'] == 'WIN').sum()}")
print(f"Losses: {(bets_df['bet_result'] == 'LOSS').sum()}")
print(f"Pushes: {(bets_df['bet_result'] == 'PUSH').sum()}")
wins = (bets_df['bet_result'] == 'WIN').sum()
losses = (bets_df['bet_result'] == 'LOSS').sum()
if wins + losses > 0:
    win_rate = wins / (wins + losses)
    print(f"Win rate: {win_rate:.1%}")

    units = wins * 1.0 - losses * 1.1
    roi = units / ((wins + losses) * 1.1)
    print(f"Units: {units:+.1f}")
    print(f"ROI: {roi:.1%}")

# Check for home/away bet balance
print(f"\n\nBET DISTRIBUTION")
print(f"=" * 80)
print(f"Home bets: {(bets_df['bet_side'] == 'home').sum()}")
print(f"Away bets: {(bets_df['bet_side'] == 'away').sum()}")

# Check edge distribution
print(f"\n\nEDGE DISTRIBUTION")
print(f"=" * 80)
print(f"Mean edge: {bets_df['edge'].mean():.2f}")
print(f"Mean |edge|: {bets_df['abs_edge'].mean():.2f}")
print(f"Median |edge|: {bets_df['abs_edge'].median():.2f}")
print(f"Max |edge|: {bets_df['abs_edge'].max():.2f}")
