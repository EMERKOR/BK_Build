"""
Backtest Ball Knower v1.0 Against-The-Spread Performance

Uses nfelo historical games with actual outcomes to evaluate:
- ATS record (wins, losses, pushes)
- Win rate and ROI
- Performance by edge threshold
- Calibration quality

Model: predicted_margin = 2.0295 + 0.0449 × nfelo_diff
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Model parameters (from v1.0 calibration)
NFELO_COEF = 0.0449
INTERCEPT = 2.0295

print("\n" + "="*80)
print("BALL KNOWER v1.0 - ATS BACKTEST WITH ACTUAL OUTCOMES")
print("="*80)
print(f"\nModel: predicted_margin = {INTERCEPT} + ({NFELO_COEF} × nfelo_diff)")

# ============================================================================
# LOAD HISTORICAL DATA WITH ACTUAL OUTCOMES
# ============================================================================

print("\n[1/6] Loading historical data with actual game outcomes...")

# Load nfelo games with ELO ratings and Vegas lines
print("  - Loading nfelo ELO ratings...")
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
nfelo = pd.read_csv(nfelo_url)
print(f"    ✓ Loaded {len(nfelo):,} games from nfelo")

# Load nflverse games for actual scores
print("  - Loading nflverse game scores...")
nflverse_url = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
scores = pd.read_csv(nflverse_url)
print(f"    ✓ Loaded {len(scores):,} games from nflverse")

# Merge on game_id
print("  - Merging datasets...")
df = nfelo.merge(scores[['game_id', 'home_score', 'away_score']], on='game_id', how='inner')
print(f"    ✓ Merged to {len(df):,} games with both ELO and scores")

# Extract season/week/teams from game_id (format: YYYY_WW_AWAY_HOME)
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Filter to games with all required data
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()
df = df[df['home_score'].notna()].copy()
df = df[df['away_score'].notna()].copy()

# Restrict to backtest period (2018-2025)
df = df[df['season'] >= 2018].copy()

print(f"\n✓ Final dataset: {len(df):,} games with complete data (2018-2025)")
print(f"  Coverage: {df['season'].min()}-{df['season'].max()}")

# ============================================================================
# CALCULATE KEY METRICS
# ============================================================================

print("\n[2/6] Calculating predictions and ATS outcomes...")

# Calculate actual margin (home perspective)
df['home_score'] = df['home_score'].astype(float)
df['away_score'] = df['away_score'].astype(float)
df['actual_margin'] = df['home_score'] - df['away_score']

# Calculate ELO differential (home perspective)
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# Ball Knower v1.0 prediction (home perspective, predicted margin)
# Predicted margin: positive = home expected to win by X points
df['bk_predicted_margin'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)

# Vegas line from nfelo uses BETTING CONVENTION:
#   - Negative line (e.g., -7) = home FAVORED by 7 points
#   - Positive line (e.g., +7) = home UNDERDOG by 7 points
#
# Convert to PREDICTED MARGIN convention for consistent comparison:
#   - Positive margin = home expected to win by X points
#   - Negative margin = home expected to lose by X points
#
# Conversion: predicted_margin = -1 × betting_line
df['vegas_betting_line'] = df['home_line_close']
df['vegas_predicted_margin'] = -1 * df['vegas_betting_line']

# Edge calculation (in predicted margin space)
# Positive edge = we think home will do better than Vegas thinks
# Negative edge = we think home will do worse than Vegas thinks
df['edge'] = df['bk_predicted_margin'] - df['vegas_predicted_margin']
df['abs_edge'] = df['edge'].abs()

# ============================================================================
# ATS LOGIC - PREDICTED MARGIN PERSPECTIVE
# ============================================================================
#
# ALL VALUES NOW IN PREDICTED MARGIN CONVENTION:
#   - actual_margin: positive = home won, negative = home lost
#   - vegas_predicted_margin: positive = Vegas expects home to win
#   - bk_predicted_margin: positive = we expect home to win
#   - edge: positive = we're more bullish on home than Vegas
#
# COVERING THE SPREAD (in predicted margin space):
#   - Home covers if: actual_margin > vegas_predicted_margin
#   - Away covers if: actual_margin < vegas_predicted_margin
#   - Push if: actual_margin == vegas_predicted_margin
#
# BETTING RULES:
#   - If edge >= threshold: bet home (we think home will do better)
#   - If edge <= -threshold: bet away (we think away will do better)
#   - Otherwise: no bet
#
# OUTCOMES:
#   - If we bet home: win if actual_margin > vegas_predicted_margin
#   - If we bet away: win if actual_margin < vegas_predicted_margin
#   - Push if actual_margin == vegas_predicted_margin
# ============================================================================

EDGE_THRESHOLD = 2.0

def determine_bet_and_outcome(row):
    """Determine bet side and outcome based on edge and actual result."""
    edge = row['edge']
    actual_margin = row['actual_margin']
    vegas_predicted_margin = row['vegas_predicted_margin']

    # Determine bet side based on edge
    if edge >= EDGE_THRESHOLD:
        bet_side = 'home'
    elif edge <= -EDGE_THRESHOLD:
        bet_side = 'away'
    else:
        bet_side = 'no_bet'

    # Determine outcome
    if bet_side == 'no_bet':
        bet_result = 'no_bet'
    elif actual_margin == vegas_predicted_margin:
        bet_result = 'push'
    elif bet_side == 'home':
        # We bet home, win if home covers (does better than Vegas expected)
        bet_result = 'win' if actual_margin > vegas_predicted_margin else 'loss'
    else:  # bet_side == 'away'
        # We bet away, win if away covers (home does worse than Vegas expected)
        bet_result = 'win' if actual_margin < vegas_predicted_margin else 'loss'

    return pd.Series({'bet_side': bet_side, 'bet_result': bet_result})

df[['bet_side', 'bet_result']] = df.apply(determine_bet_and_outcome, axis=1)

print(f"✓ Calculated {len(df):,} predictions and outcomes")

# ============================================================================
# DEBUG SAMPLE - INSPECT 10 RANDOM GAMES
# ============================================================================

print("\n[3/6] Debug sample - examining 10 random games...")
print("\n" + "="*80)
print("DEBUG SAMPLE (10 random games)")
print("="*80)

debug_cols = [
    'season', 'week', 'home_team', 'away_team',
    'home_score', 'away_score', 'actual_margin',
    'vegas_betting_line', 'vegas_predicted_margin',
    'bk_predicted_margin', 'edge',
    'bet_side', 'bet_result'
]

debug_sample = df[debug_cols].sample(10, random_state=42)
print("\n" + debug_sample.to_string(index=False))

# ============================================================================
# ATS PERFORMANCE SUMMARY
# ============================================================================

print("\n\n[4/6] Calculating ATS performance...")
print("\n" + "="*80)
print("ATS PERFORMANCE SUMMARY")
print("="*80)

# Overall ATS record
bets = df[df['bet_side'] != 'no_bet']
wins = len(bets[bets['bet_result'] == 'win'])
losses = len(bets[bets['bet_result'] == 'loss'])
pushes = len(bets[bets['bet_result'] == 'push'])
total_bets = len(bets)
win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

print(f"\nEdge Threshold: {EDGE_THRESHOLD} points")
print(f"\nATS Record: {wins}-{losses}-{pushes}")
print(f"Win Rate: {win_rate:.1f}% (excluding pushes)")
print(f"Total Bets: {total_bets:,} ({total_bets/len(df)*100:.1f}% of games)")

# Calculate units and ROI (assuming -110 odds)
# Win: +1 unit, Loss: -1.1 units, Push: 0 units
units = wins * 1.0 - losses * 1.1
roi = (units / (total_bets * 1.1)) * 100 if total_bets > 0 else 0

print(f"\nUnits Won: {units:+.1f} units (at -110 odds)")
print(f"ROI: {roi:+.1f}%")

# Prediction accuracy (overall, not just bets)
mae = np.abs(df['actual_margin'] - df['bk_predicted_margin']).mean()
rmse = np.sqrt(((df['actual_margin'] - df['bk_predicted_margin']) ** 2).mean())

print(f"\nPrediction Accuracy (all games):")
print(f"  MAE: {mae:.2f} points")
print(f"  RMSE: {rmse:.2f} points")

# ============================================================================
# EDGE DISTRIBUTION AND WIN RATE BY EDGE BUCKET
# ============================================================================

print("\n\n[5/6] Analyzing edge distribution and performance...")
print("\n" + "="*80)
print("EDGE DISTRIBUTION & PERFORMANCE BY BUCKET")
print("="*80)

# Basic edge stats
print(f"\nEdge Statistics (all games):")
print(f"  Mean: {df['edge'].mean():.2f}")
print(f"  Median: {df['edge'].median():.2f}")
print(f"  Std: {df['edge'].std():.2f}")
print(f"  Min: {df['edge'].min():.2f}")
print(f"  Max: {df['edge'].max():.2f}")

# Count games by absolute edge threshold
print(f"\nGames by absolute edge threshold:")
for threshold in [1, 2, 4, 6]:
    count = len(df[df['abs_edge'] >= threshold])
    pct = count / len(df) * 100
    print(f"  >= {threshold} points: {count:,} ({pct:.1f}%)")

# Performance by edge bucket
edge_buckets = [
    (0, 2, '0-2'),
    (2, 4, '2-4'),
    (4, 6, '4-6'),
    (6, 100, '6+')
]

print(f"\nATS Performance by Edge Bucket:")
print(f"{'Edge Range':<12} {'Bets':<8} {'W-L-P':<15} {'Win%':<8}")
print("-" * 50)

for low, high, label in edge_buckets:
    bucket_bets = bets[(bets['abs_edge'] >= low) & (bets['abs_edge'] < high)]
    if len(bucket_bets) == 0:
        continue

    b_wins = len(bucket_bets[bucket_bets['bet_result'] == 'win'])
    b_losses = len(bucket_bets[bucket_bets['bet_result'] == 'loss'])
    b_pushes = len(bucket_bets[bucket_bets['bet_result'] == 'push'])
    b_win_pct = b_wins / (b_wins + b_losses) * 100 if (b_wins + b_losses) > 0 else 0

    print(f"{label:<12} {len(bucket_bets):<8} {b_wins}-{b_losses}-{b_pushes:<10} {b_win_pct:.1f}%")

# ============================================================================
# PERFORMANCE BY SEASON
# ============================================================================

print("\n\n[6/6] Calculating performance by season...")
print("\n" + "="*80)
print("PERFORMANCE BY SEASON")
print("="*80)

season_results = []
for season in sorted(df['season'].unique()):
    season_df = df[df['season'] == season]
    season_bets = season_df[season_df['bet_side'] != 'no_bet']

    if len(season_bets) == 0:
        continue

    s_wins = len(season_bets[season_bets['bet_result'] == 'win'])
    s_losses = len(season_bets[season_bets['bet_result'] == 'loss'])
    s_pushes = len(season_bets[season_bets['bet_result'] == 'push'])
    s_win_pct = s_wins / (s_wins + s_losses) * 100 if (s_wins + s_losses) > 0 else 0
    s_units = s_wins * 1.0 - s_losses * 1.1

    season_results.append({
        'Season': season,
        'Bets': len(season_bets),
        'W-L-P': f"{s_wins}-{s_losses}-{s_pushes}",
        'Win%': f"{s_win_pct:.1f}%",
        'Units': f"{s_units:+.1f}"
    })

season_df_display = pd.DataFrame(season_results)
print("\n" + season_df_display.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

backtest_file = output_dir / 'backtest_v1_0_ats_results.csv'
df.to_csv(backtest_file, index=False)

print(f"\n\nBacktest results saved to: {backtest_file}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80 + "\n")
