"""
Ball Knower v1.0 - Actual Margin Backtest

Evaluates v1.0 model performance against:
1. Actual game outcomes (calibration check)
2. Vegas closing lines (edge-based betting simulation)

This backtest answers the key question:
"Does modeling actual margins (instead of Vegas lines) produce better edges?"
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ball_knower.datasets import v1_0 as ds_v1_0

print("\n" + "="*80)
print("BALL KNOWER v1.0 - ACTUAL MARGIN BACKTEST")
print("="*80)

# ============================================================================
# LOAD MODEL PARAMETERS
# ============================================================================

print("\n[1/5] Loading v1.0 model parameters...")

params_file = Path(project_root) / 'output' / 'v1_0_model_params.json'

if not params_file.exists():
    print(f"\n❌ Error: Model parameters not found at {params_file}")
    print("   Please run src/rebuild_v1_0_model.py first to train the model.")
    sys.exit(1)

with open(params_file, 'r') as f:
    model_params = json.load(f)

intercept = model_params['intercept']
coef_nfelo_diff = model_params['coef_nfelo_diff']

print(f"✓ Loaded v1.0 model:")
print(f"  Intercept:   {intercept:.4f}")
print(f"  nfelo_diff:  {coef_nfelo_diff:.4f}")
print(f"  Trained on:  {model_params['training_period']['min_season']}-{model_params['training_period']['max_season']}")

# ============================================================================
# LOAD BACKTEST DATA
# ============================================================================

print("\n[2/5] Loading backtest data (2018-2024)...")

# Use a recent period for backtest (after the training period ends at 2023)
# But also include some overlap to see in-sample vs out-of-sample
df = ds_v1_0.build_v1_0_training_frame(min_season=2018, max_season=None)

print(f"✓ Loaded {len(df):,} games from {df['season'].min()}-{df['season'].max()}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n[3/5] Generating v1.0 predictions...")

# Apply the v1.0 model: predicted_margin = intercept + coef * nfelo_diff
df['bk_line_v1_0'] = intercept + coef_nfelo_diff * df['nfelo_diff']

# Calculate edge vs Vegas (home perspective)
# Positive edge = we predict home team to do better than Vegas expects
df['edge'] = df['bk_line_v1_0'] - df['vegas_line']
df['abs_edge'] = df['edge'].abs()

print(f"✓ Generated predictions for {len(df):,} games")

# ============================================================================
# EVALUATE CALIBRATION VS ACTUAL MARGINS
# ============================================================================

print("\n[4/5] Evaluating calibration vs actual game margins...")

# Error vs actual outcomes
df['error_vs_actual'] = df['actual_margin'] - df['bk_line_v1_0']

mae_actual = df['error_vs_actual'].abs().mean()
rmse_actual = np.sqrt((df['error_vs_actual'] ** 2).mean())
mean_error = df['error_vs_actual'].mean()

print(f"\nCalibration vs Actual Margins:")
print(f"  MAE:         {mae_actual:.2f} points")
print(f"  RMSE:        {rmse_actual:.2f} points")
print(f"  Mean error:  {mean_error:.4f} points (should be ~0)")

# ============================================================================
# EVALUATE ATS PERFORMANCE VS VEGAS
# ============================================================================

print("\n[5/5] Evaluating ATS performance vs Vegas...")

# Define edge threshold for betting
EDGE_THRESHOLD = 2.0  # Bet when abs(edge) >= 2.0 points

print(f"\nUsing edge threshold: {EDGE_THRESHOLD} points")

# Identify betting opportunities
bet_mask = df['abs_edge'] >= EDGE_THRESHOLD
df_bets = df[bet_mask].copy()

print(f"Games meeting threshold: {len(df_bets):,} ({len(df_bets)/len(df)*100:.1f}%)")

if len(df_bets) > 0:
    # For each bet, determine:
    # - Which side we're betting (home or away)
    # - Whether we won vs the spread

    # Determine bet side based on edge direction
    # If edge > 0: we think home team outperforms Vegas → bet on home
    # If edge < 0: we think away team outperforms Vegas → bet on away
    df_bets['bet_side'] = df_bets['edge'].apply(lambda x: 'home' if x > 0 else 'away')

    # Calculate ATS result
    # ATS margin = actual_margin - vegas_line (from home perspective)
    df_bets['ats_margin'] = df_bets['actual_margin'] - df_bets['vegas_line']

    # Determine bet outcome
    # If we bet home and ATS margin > 0 → win (home covered)
    # If we bet away and ATS margin < 0 → win (away covered)
    # If ATS margin = 0 → push
    def determine_outcome(row):
        if row['ats_margin'] == 0:
            return 'push'
        elif row['bet_side'] == 'home':
            return 'win' if row['ats_margin'] > 0 else 'loss'
        else:  # bet_side == 'away'
            return 'win' if row['ats_margin'] < 0 else 'loss'

    df_bets['outcome'] = df_bets.apply(determine_outcome, axis=1)

    # Calculate record
    wins = (df_bets['outcome'] == 'win').sum()
    losses = (df_bets['outcome'] == 'loss').sum()
    pushes = (df_bets['outcome'] == 'push').sum()
    total_decided = wins + losses
    win_pct = wins / total_decided * 100 if total_decided > 0 else 0

    # Calculate units won/lost (assuming -110 vig, 1 unit per bet)
    # Win = +0.909 units (risk 1.1 to win 1.0)
    # Loss = -1.0 units
    # Push = 0 units
    units_won = wins * 0.909 - losses * 1.0
    roi = units_won / len(df_bets) * 100 if len(df_bets) > 0 else 0

    print(f"\nATS Performance (threshold >= {EDGE_THRESHOLD} points):")
    print(f"  Bets placed:  {len(df_bets):,}")
    print(f"  Wins:         {wins:,}")
    print(f"  Losses:       {losses:,}")
    print(f"  Pushes:       {pushes:,}")
    print(f"  Win %:        {win_pct:.1f}%")
    print(f"  Units won:    {units_won:+.2f}")
    print(f"  ROI:          {roi:+.2f}%")

    # Breakeven at -110 is 52.4%
    breakeven = 52.4
    print(f"\n  Breakeven (at -110): {breakeven:.1f}%")
    if win_pct >= breakeven:
        print(f"  ✓ Profitable! ({win_pct - breakeven:+.1f}% above breakeven)")
    else:
        print(f"  ⚠ Below breakeven ({win_pct - breakeven:.1f}%)")

    # Breakdown by bet side
    print(f"\nBreakdown by side:")
    for side in ['home', 'away']:
        side_bets = df_bets[df_bets['bet_side'] == side]
        if len(side_bets) > 0:
            side_wins = (side_bets['outcome'] == 'win').sum()
            side_losses = (side_bets['outcome'] == 'loss').sum()
            side_total = side_wins + side_losses
            side_win_pct = side_wins / side_total * 100 if side_total > 0 else 0
            print(f"  {side.capitalize():5s}: {len(side_bets):3,} bets, {side_wins:3,}-{side_losses:3,} ({side_win_pct:.1f}%)")

    # Breakdown by season
    print(f"\nPerformance by season:")
    season_summary = df_bets.groupby('season').agg({
        'outcome': lambda x: (
            f"{(x == 'win').sum()}-{(x == 'loss').sum()}-{(x == 'push').sum()}"
        ),
        'edge': lambda x: f"{x.abs().mean():.2f}"
    })
    season_summary.columns = ['Record (W-L-P)', 'Avg Edge']

    # Add win percentage
    season_win_pct = df_bets.groupby('season').apply(
        lambda x: (x['outcome'] == 'win').sum() / ((x['outcome'] == 'win').sum() + (x['outcome'] == 'loss').sum()) * 100
        if ((x['outcome'] == 'win').sum() + (x['outcome'] == 'loss').sum()) > 0 else 0
    )
    season_summary['Win %'] = season_win_pct.apply(lambda x: f"{x:.1f}%")

    print(season_summary.to_string())

else:
    print("\n⚠ No games meet the edge threshold")

# ============================================================================
# EDGE DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EDGE DISTRIBUTION")
print("="*80)

edge_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

print(f"\nGames by edge threshold:")
for threshold in edge_bins:
    count = len(df[df['abs_edge'] >= threshold])
    pct = count / len(df) * 100
    avg_edge = df[df['abs_edge'] >= threshold]['abs_edge'].mean() if count > 0 else 0
    print(f"  {threshold:>3.1f}+ points: {count:4,} games ({pct:5.1f}%) - avg edge: {avg_edge:.2f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BACKTEST SUMMARY")
print("="*80)

print(f"""
Period: {df['season'].min()}-{df['season'].max()} ({len(df):,} games)

Calibration (vs Actual Margins):
- MAE:  {mae_actual:.2f} points
- RMSE: {rmse_actual:.2f} points
- Model is {'well-calibrated' if abs(mean_error) < 0.5 else 'slightly biased'} (mean error: {mean_error:+.2f})

ATS Performance (vs Vegas, threshold = {EDGE_THRESHOLD}+ points):
- Bets:    {len(df_bets):,} ({len(df_bets)/len(df)*100:.1f}% of games)
- Record:  {wins}-{losses}-{pushes}
- Win %:   {win_pct:.1f}%
- Units:   {units_won:+.2f}
- ROI:     {roi:+.2f}%

Edge Opportunities:
- Games with 1.5+ edge: {len(df[df['abs_edge'] >= 1.5]):,} ({len(df[df['abs_edge'] >= 1.5])/len(df)*100:.1f}%)
- Games with 2.0+ edge: {len(df[df['abs_edge'] >= 2.0]):,} ({len(df[df['abs_edge'] >= 2.0])/len(df)*100:.1f}%)
- Games with 3.0+ edge: {len(df[df['abs_edge'] >= 3.0]):,} ({len(df[df['abs_edge'] >= 3.0])/len(df)*100:.1f}%)

Key Insight:
{'✓ v1.0 shows promise! Modeling actual margins produces profitable edges.' if win_pct >= breakeven else '⚠ v1.0 needs refinement. Consider adding features or adjusting threshold.'}
""")

print("="*80 + "\n")
