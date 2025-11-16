"""
Ball Knower v1.3 - Current Week Predictions with EPA Features

Generates betting recommendations using the EPA-enhanced model.
Requires EPA data for current teams (rolling averages from recent games).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import data_loader, config, betting_utils, team_mapping

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n" + "="*80)
print("BALL KNOWER v1.3 - CURRENT WEEK PREDICTIONS (EPA-ENHANCED)")
print("="*80)

# ============================================================================
# LOAD v1.3 MODEL
# ============================================================================

print("\n[1/5] Loading v1.3 model...")

model_file = config.OUTPUT_DIR / 'ball_knower_v1_3_model.json'

if not model_file.exists():
    print(f"  ✗ ERROR: v1.3 model not found")
    print(f"  Run ball_knower_v1_3.py first to train the model")
    sys.exit(1)

with open(model_file, 'r') as f:
    model_params = json.load(f)

print(f"  ✓ Loaded v1.3 model")
print(f"  Training R²: {model_params['train_r2']:.3f}")
print(f"  Test R²:     {model_params['test_r2']:.3f}")
print(f"  Test MAE:    {model_params['test_mae']:.2f} points")

# ============================================================================
# LOAD CURRENT WEEK DATA
# ============================================================================

print("\n[2/5] Loading Week 11 2025 data...")

# Get current week games
games = nflverse.games(season=2025, week=11)
games = games[games['spread_line'].notna()].copy()

print(f"  Loaded {len(games)} games with Vegas lines")

# Get team ratings from nfelo snapshot
nfelo_snapshot_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'
nfelo_snapshot = pd.read_csv(nfelo_snapshot_url)

# Handle duplicate teams (take first occurrence)
nfelo_snapshot = nfelo_snapshot.drop_duplicates(subset=['team'], keep='first')

print(f"  Loaded {len(nfelo_snapshot)} team ratings")

# ============================================================================
# LOAD EPA DATA
# ============================================================================

print("\n[3/5] Loading EPA data and calculating recent form...")

epa_file = project_root / 'data' / 'team_week_epa_2013_2024.csv'

if not epa_file.exists():
    print(f"  ✗ ERROR: EPA file not found: {epa_file}")
    sys.exit(1)

epa_df = pd.read_csv(epa_file)

# EPA data already uses standard team abbreviations
# Calculate recent form (last 3 games average)
epa_df = epa_df.sort_values(['team', 'season', 'week'])

recent_epa = []

for team in epa_df['team'].unique():
    team_data = epa_df[epa_df['team'] == team].copy()

    # Get last 3 games (most recent)
    recent_3 = team_data.tail(3)

    if len(recent_3) > 0:
        recent_epa.append({
            'team': team,
            'off_epa_per_play': recent_3['off_epa_per_play'].mean(),
            'def_epa_per_play': recent_3['def_epa_per_play'].mean(),
            'off_success_rate': recent_3['off_success_rate'].mean(),
            'def_success_rate': recent_3['def_success_rate'].mean(),
            'games_in_avg': len(recent_3),
        })

recent_epa_df = pd.DataFrame(recent_epa)

print(f"  ✓ Calculated recent form for {len(recent_epa_df)} teams (3-game rolling avg)")

# ============================================================================
# PREPARE MATCHUP FEATURES
# ============================================================================

print("\n[4/5] Engineering features for current week...")

# Base matchup data
matchups = games[['away_team', 'home_team', 'spread_line',
                  'home_rest', 'away_rest', 'div_game']].copy()

# CRITICAL FIX: Convert nflverse spread_line to home perspective
matchups['spread_line'] = -1 * matchups['spread_line']

# Team names from nflverse already match standard format - no mapping needed

# Merge nfelo ratings
nfelo_home = nfelo_snapshot[['team', 'nfelo', 'qb_adj']].copy()
nfelo_home.columns = ['home_team', 'home_nfelo', 'home_qb_adj']

nfelo_away = nfelo_snapshot[['team', 'nfelo', 'qb_adj']].copy()
nfelo_away.columns = ['away_team', 'away_nfelo', 'away_qb_adj']

matchups = matchups.merge(nfelo_home, on='home_team', how='left')
matchups = matchups.merge(nfelo_away, on='away_team', how='left')

# Merge EPA data
epa_home = recent_epa_df.copy()
epa_home.columns = ['home_team'] + [f'home_{col}' for col in recent_epa_df.columns if col != 'team']

epa_away = recent_epa_df.copy()
epa_away.columns = ['away_team'] + [f'away_{col}' for col in recent_epa_df.columns if col != 'team']

matchups = matchups.merge(epa_home, on='home_team', how='left')
matchups = matchups.merge(epa_away, on='away_team', how='left')

# Check EPA coverage
epa_coverage = matchups[['home_off_epa_per_play', 'away_off_epa_per_play']].notna().all(axis=1).sum()
print(f"  EPA coverage: {epa_coverage}/{len(matchups)} games")

# Engineer features (same as v1.3 training)
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']

matchups['rest_advantage'] = (matchups['home_rest'] - matchups['away_rest'])
matchups['div_game'] = matchups['div_game'].astype(int)

# Use defaults for missing situational features (not available in current week data)
matchups['surface_mod'] = 0.0
matchups['time_advantage'] = 0.0

matchups['qb_diff'] = matchups['home_qb_adj'].fillna(0) - matchups['away_qb_adj'].fillna(0)

# EPA differentials
matchups['epa_off_diff'] = matchups['home_off_epa_per_play'] - matchups['away_off_epa_per_play']
matchups['epa_def_diff'] = matchups['home_def_epa_per_play'] - matchups['away_def_epa_per_play']
matchups['success_rate_off_diff'] = matchups['home_off_success_rate'] - matchups['away_off_success_rate']
matchups['success_rate_def_diff'] = matchups['home_def_success_rate'] - matchups['away_def_success_rate']

# Fill any remaining NaNs with 0 (teams without EPA data)
epa_features = ['epa_off_diff', 'epa_def_diff', 'success_rate_off_diff', 'success_rate_def_diff']
for feat in epa_features:
    matchups[feat] = matchups[feat].fillna(0)

print(f"  ✓ Prepared features for {len(matchups)} games")

# Generate predictions
feature_cols = model_params['features']
coefs = model_params['coefficients']
intercept = model_params['intercept']

matchups['bk_v1_3_spread'] = intercept
for feat in feature_cols:
    matchups['bk_v1_3_spread'] += matchups[feat] * coefs[feat]

# Calculate edge
matchups['edge'] = matchups['bk_v1_3_spread'] - matchups['spread_line']
matchups['abs_edge'] = matchups['edge'].abs()

print("  ✓ Predictions generated")

# ============================================================================
# BETTING ANALYTICS
# ============================================================================

print("\n[4.5/5] Calculating betting metrics...")

# Use v1.3 residual std for probability calibration
residual_std = model_params['test_rmse']

# Win probabilities
matchups['home_win_prob'] = matchups['bk_v1_3_spread'].apply(
    lambda x: betting_utils.spread_to_win_probability(x, residual_std=residual_std)
)
matchups['away_win_prob'] = 1 - matchups['home_win_prob']

# Determine bet side
matchups['bet_side'] = matchups['edge'].apply(lambda x: 'HOME' if x < 0 else 'AWAY')
matchups['bet_prob'] = matchups.apply(
    lambda row: row['home_win_prob'] if row['bet_side'] == 'HOME' else row['away_win_prob'],
    axis=1
)

standard_odds = -110

# Kelly sizing
matchups['kelly_full'] = matchups.apply(
    lambda row: betting_utils.kelly_criterion(
        model_prob=row['bet_prob'],
        odds=standard_odds
    ),
    axis=1
)
matchups['kelly_quarter'] = matchups['kelly_full'] * 0.25

# Expected value
matchups['ev_per_100'] = matchups.apply(
    lambda row: betting_utils.calculate_ev(
        model_prob=row['bet_prob'],
        market_prob=0.5,
        odds=standard_odds,
        stake=100
    ),
    axis=1
)

print(f"  ✓ Added betting analytics")

# ============================================================================
# VALUE BETS
# ============================================================================

print("\n[5/5] Identifying value bets...")

print("\n" + "="*80)
print("ALL WEEK 11 PREDICTIONS (v1.3)")
print("="*80)

display_cols = ['away_team', 'home_team', 'spread_line', 'bk_v1_3_spread',
                'edge', 'home_win_prob', 'kelly_quarter', 'ev_per_100']

all_preds = matchups[display_cols].copy()
all_preds = all_preds.sort_values('edge', ascending=False)

# Format for display
all_preds_display = all_preds.copy()
all_preds_display['home_win_prob'] = (all_preds_display['home_win_prob'] * 100).round(1)
all_preds_display['kelly_quarter'] = (all_preds_display['kelly_quarter'] * 100).round(2)
all_preds_display[['spread_line', 'bk_v1_3_spread', 'edge', 'ev_per_100']] = \
    all_preds_display[['spread_line', 'bk_v1_3_spread', 'edge', 'ev_per_100']].round(2)

all_preds_display.columns = ['Away', 'Home', 'Vegas', 'BK_v1.3', 'Edge',
                              'Home_Win_%', '1/4_Kelly_%', 'EV_per_$100']

print("\n" + all_preds_display.to_string(index=False))

# Value bet analysis
value_threshold = 2.0
value_bets = matchups[matchups['abs_edge'] >= value_threshold].copy()

print("\n" + "="*80)
print(f"VALUE BETS ({value_threshold}+ POINT EDGE)")
print("="*80)

if len(value_bets) > 0:
    value_bets['bet_team'] = value_bets.apply(
        lambda row: row['home_team'] if row['bet_side'] == 'HOME' else row['away_team'],
        axis=1
    )

    value_bets['recommendation'] = value_bets.apply(
        lambda row: f"Bet {row['bet_team']} {row['spread_line']:.1f} (edge: {abs(row['edge']):.1f})",
        axis=1
    )

    bet_display = value_bets[[
        'away_team', 'home_team', 'spread_line', 'bk_v1_3_spread',
        'edge', 'bet_side', 'kelly_quarter', 'ev_per_100', 'recommendation'
    ]].copy()

    bet_display = bet_display.sort_values('edge', key=abs, ascending=False)

    bet_display_formatted = bet_display.copy()
    bet_display_formatted['kelly_pct'] = (bet_display_formatted['kelly_quarter'] * 100).round(2)
    bet_display_formatted = bet_display_formatted.drop(columns=['kelly_quarter'])
    bet_display_formatted = bet_display_formatted.round(2)

    print(f"\n{len(value_bets)} recommended bets:\n")
    print(bet_display_formatted.to_string(index=False))

    # Professional analysis
    print("\n" + "="*80)
    print("PROFESSIONAL ANALYSIS")
    print("="*80)

    print(f"\nTotal recommended bets: {len(value_bets)}\n")

    for bet_idx, bet in bet_display.iterrows():
        edge_mag = abs(bet['edge'])
        kelly_pct = bet['kelly_quarter'] * 100

        if edge_mag >= 3.0:
            confidence = "HIGH"
        elif edge_mag >= 2.5:
            confidence = "MODERATE-HIGH"
        elif edge_mag >= 2.0:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        print(f"  {bet['away_team']} @ {bet['home_team']}")
        print(f"    {bet['recommendation']}")
        print(f"    Confidence: {confidence} | 1/4 Kelly: {kelly_pct:.2f}% | EV: ${bet['ev_per_100']:.2f}/bet")
        print()

else:
    print(f"\nNo games meet the {value_threshold}+ point threshold this week.")
    print(f"Largest edge: {matchups['abs_edge'].max():.2f} points")

# Save predictions
output_file = config.OUTPUT_DIR / 'week_11_value_bets_v1_3.csv'
matchups.to_csv(output_file, index=False)
print(f"\n✓ Predictions saved to: {output_file}")

print("\n" + "="*80)
print("v1.3 vs v1.2 PREDICTION COMPARISON")
print("="*80)

# Load v1.2 predictions if available
v1_2_file = config.OUTPUT_DIR / 'week_11_value_bets_v1_2.csv'
if v1_2_file.exists():
    v1_2_preds = pd.read_csv(v1_2_file)

    # Merge for comparison
    comparison = matchups[['away_team', 'home_team', 'spread_line', 'bk_v1_3_spread', 'edge']].copy()
    comparison = comparison.merge(
        v1_2_preds[['away_team', 'home_team', 'bk_v1_2_spread', 'edge']],
        on=['away_team', 'home_team'],
        how='left',
        suffixes=('_v1_3', '_v1_2')
    )

    comparison['edge_v1_2'] = comparison['bk_v1_2_spread'] - comparison['spread_line']
    comparison['edge_diff'] = comparison['edge_v1_3'] - comparison['edge_v1_2']

    print("\nHow EPA changed the predictions:\n")
    print(comparison[['away_team', 'home_team', 'spread_line', 'bk_v1_2_spread',
                      'bk_v1_3_spread', 'edge_v1_2', 'edge_v1_3', 'edge_diff']].round(2).to_string(index=False))

    print(f"\nMean absolute edge change: {comparison['edge_diff'].abs().mean():.2f} points")

else:
    print("\nv1.2 predictions not available for comparison")

print("\n" + "="*80 + "\n")
