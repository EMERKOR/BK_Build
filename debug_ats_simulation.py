#!/usr/bin/env python3
"""
ATS (Against The Spread) Simulation for Ball Knower v1.0

Simulates betting against Vegas spreads using the v1.0 model.
Reports detailed metrics to verify the sign bug fix.
"""

import pandas as pd
import numpy as np

# v1.0 model parameters (CORRECTED - fixed sign bug)
NFELO_COEF = -0.042
INTERCEPT = -1.46

print("\n" + "="*80)
print("BALL KNOWER v1.0 - ATS SIMULATION")
print("="*80)
print(f"\nModel: spread = {INTERCEPT} + ({NFELO_COEF} × nfelo_diff)")
print(f"(Corrected coefficients - negative correlation)")

# ============================================================================
# LOAD HISTORICAL DATA
# ============================================================================

print("\n[1/3] Loading nfelo historical data...")

# Load nfelo games with ELO ratings and Vegas lines
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
nfelo_df = pd.read_csv(nfelo_url)

print(f"✓ Loaded {len(nfelo_df):,} games from nfelo database")

# Extract season/week/teams from game_id
nfelo_df[['season', 'week', 'away_team', 'home_team']] = \
    nfelo_df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
nfelo_df['season'] = nfelo_df['season'].astype(int)
nfelo_df['week'] = nfelo_df['week'].astype(int)

# Load schedules with actual scores
schedules_df = pd.read_parquet('schedules.parquet')

print(f"✓ Loaded {len(schedules_df):,} games from schedules database")

# Merge nfelo data with schedules to get scores
df = nfelo_df.merge(
    schedules_df[['game_id', 'home_score', 'away_score']],
    on='game_id',
    how='inner'
)

print(f"✓ Merged to {len(df):,} games with complete data")

# Filter to games with complete data (2013-2024 for consistency)
df = df[(df['season'] >= 2013) & (df['season'] <= 2024)].copy()
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()
df = df[df['home_score'].notna()].copy()
df = df[df['away_score'].notna()].copy()

print(f"✓ Filtered to {len(df):,} games (2013-2024 with complete data)")

# ============================================================================
# CALCULATE PREDICTIONS AND EDGES
# ============================================================================

print("\n[2/3] Calculating v1.0 predictions and edges...")

# Calculate nfelo differential
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# Calculate v1.0 predicted spread (home perspective)
df['bk_v1_0_spread'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)

# Calculate actual margin (home perspective)
df['actual_margin'] = df['home_score'] - df['away_score']

# Calculate edge vs Vegas (positive = model thinks home team is undervalued)
df['edge'] = df['bk_v1_0_spread'] - df['home_line_close']
df['abs_edge'] = df['edge'].abs()

# Calculate ATS outcome
# If we bet when edge > 0 (model favors home more than Vegas):
#   - Win if home_score - away_score > home_line_close (home covers)
# If we bet when edge < 0 (model favors away more than Vegas):
#   - Win if away_score - home_score > -home_line_close (away covers)
# Simplified: Win if actual_margin - home_line_close > 0

df['ats_cover'] = df['actual_margin'] - df['home_line_close']
df['ats_win'] = df['ats_cover'] > 0  # True if bet would win

# Determine which side we'd bet on
df['bet_side'] = df['edge'].apply(lambda x: 'home' if x > 0 else 'away' if x < 0 else 'none')

# Filter to actual bets (where we have an edge opinion)
bets_df = df[df['bet_side'] != 'none'].copy()

print(f"✓ Generated {len(bets_df):,} betting opportunities")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("\n[3/3] Computing ATS metrics...")

# Overall metrics
total_bets = len(bets_df)
ats_wins = bets_df['ats_win'].sum()
ats_win_rate = (ats_wins / total_bets * 100) if total_bets > 0 else 0

# ROI calculation (assuming -110 odds)
# Win: +0.909 units (bet 1.1 to win 1)
# Loss: -1.1 units
roi = ((ats_wins * 0.909) - ((total_bets - ats_wins) * 1.1)) / (total_bets * 1.1) * 100 if total_bets > 0 else 0

# Edge metrics
mean_edge = bets_df['edge'].mean()
mean_abs_edge = bets_df['abs_edge'].mean()
large_edge_count = (bets_df['abs_edge'] > 6).sum()
large_edge_pct = (large_edge_count / total_bets * 100) if total_bets > 0 else 0

# Home vs away bet counts
home_bets = (bets_df['bet_side'] == 'home').sum()
away_bets = (bets_df['bet_side'] == 'away').sum()

# ============================================================================
# REPORT RESULTS
# ============================================================================

print("\n" + "="*80)
print("ATS SIMULATION RESULTS (v1.0 - CORRECTED)")
print("="*80)

print(f"\n📊 BET VOLUME:")
print(f"  Total bets: {total_bets:,}")
print(f"  Home bets: {home_bets:,} ({home_bets/total_bets*100:.1f}%)")
print(f"  Away bets: {away_bets:,} ({away_bets/total_bets*100:.1f}%)")

print(f"\n🎯 PERFORMANCE:")
print(f"  ATS wins: {ats_wins:,}")
print(f"  ATS win rate: {ats_win_rate:.1f}%")
print(f"  ROI: {roi:.1f}%")

print(f"\n📈 EDGE DISTRIBUTION:")
print(f"  Mean edge: {mean_edge:.2f} points")
print(f"  Mean absolute edge: {mean_abs_edge:.2f} points")
print(f"  Bets with |edge| > 6: {large_edge_count:,} ({large_edge_pct:.1f}%)")
print(f"  Max absolute edge: {bets_df['abs_edge'].max():.2f} points")

print(f"\n📉 EDGE PERCENTILES:")
print(f"  25th percentile: {bets_df['abs_edge'].quantile(0.25):.2f} points")
print(f"  50th percentile: {bets_df['abs_edge'].quantile(0.50):.2f} points")
print(f"  75th percentile: {bets_df['abs_edge'].quantile(0.75):.2f} points")
print(f"  95th percentile: {bets_df['abs_edge'].quantile(0.95):.2f} points")

print("\n" + "="*80)
print("✓ Simulation complete!")
print("="*80 + "\n")

# Save detailed results
output_path = "output/ats_simulation_v1_0_corrected.csv"
bets_df[['game_id', 'season', 'week', 'home_team', 'away_team',
         'nfelo_diff', 'bk_v1_0_spread', 'home_line_close', 'edge',
         'actual_margin', 'ats_cover', 'ats_win', 'bet_side']].to_csv(
    output_path, index=False
)
print(f"💾 Detailed results saved to: {output_path}")
