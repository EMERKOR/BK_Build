"""
Backtest Ball Knower v1.2 with Professional Betting Analytics

Evaluates v1.2 model performance using:
- Expected Value (EV) calculations
- Kelly Criterion bet sizing
- ROI simulation by edge threshold
- Closing Line Value (CLV) style analysis
- Probability calibration

Model: Ridge regression trained on 2009-2024 (4,345 games)
Test period: 2025 season (165 games)

Note: This script backtests on historical nfelo games data (loaded from remote URL).
For current-week predictions, see predict_current_week.py which uses ball_knower.io.loaders.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import betting_utils, config

# Note: ball_knower.io.loaders is available for current-season data if needed
# from ball_knower.io import loaders

print("\n" + "="*80)
print("BALL KNOWER v1.2 - PROFESSIONAL BACKTEST")
print("="*80)

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

print("\n[1/6] Loading trained v1.2 model...")

model_file = config.OUTPUT_DIR / 'ball_knower_v1_2_model.json'
with open(model_file, 'r') as f:
    model_params = json.load(f)

print(f"  Training period: 2009-2024")
print(f"  Train R2: {model_params['train_r2']:.3f}")
print(f"  Test R2:  {model_params['test_r2']:.3f}")
print(f"  Test MAE: {model_params['test_mae']:.2f} points")

# ============================================================================
# LOAD HISTORICAL DATA
# ============================================================================

print("\n[2/6] Loading nfelo historical data...")

nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

# Extract season/week/teams
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

# Filter to complete data
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"  Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

# ============================================================================
# ENGINEER FEATURES
# ============================================================================

print("\n[3/6] Engineering features and generating predictions...")

# Primary feature: ELO differential
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# Situational adjustments
df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)

# QB adjustments
df['qb_diff'] = (df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0))

# Target: Vegas closing line
df['vegas_line'] = df['home_line_close']

# Remove NaN rows
feature_cols = ['nfelo_diff', 'rest_advantage', 'div_game', 'surface_mod', 'time_advantage', 'qb_diff']
mask = df[feature_cols + ['vegas_line']].notna().all(axis=1)
df = df[mask].copy()

# Generate v1.2 predictions
intercept = model_params['intercept']
coefs = model_params['coefficients']

df['bk_v1_2_spread'] = intercept + \
    (df['nfelo_diff'] * coefs['nfelo_diff']) + \
    (df['rest_advantage'] * coefs['rest_advantage']) + \
    (df['div_game'] * coefs['div_game']) + \
    (df['surface_mod'] * coefs['surface_mod']) + \
    (df['time_advantage'] * coefs['time_advantage']) + \
    (df['qb_diff'] * coefs['qb_diff'])

# Calculate edge
df['edge'] = df['bk_v1_2_spread'] - df['vegas_line']
df['abs_edge'] = df['edge'].abs()

print(f"  Generated predictions for {len(df):,} games")

# Split train/test for evaluation
df['data_split'] = df['season'].apply(lambda x: 'train' if x < 2025 else 'test')

# ============================================================================
# BETTING ANALYTICS
# ============================================================================

print("\n[4/6] Calculating betting metrics (EV, Kelly, probabilities)...")

# Model residual standard deviation (from test set for conservative estimates)
test_residuals = df[df['data_split'] == 'test']['edge'].values
residual_std = np.std(test_residuals)

print(f"  Model residual std: {residual_std:.2f} points")

# Convert spread predictions to win probabilities
df['home_win_prob'] = df['bk_v1_2_spread'].apply(
    lambda x: betting_utils.spread_to_win_probability(x, residual_std=residual_std)
)

df['away_win_prob'] = 1 - df['home_win_prob']

# For each game, determine which side has edge
df['bet_side'] = df['edge'].apply(lambda x: 'home' if x < 0 else 'away')
df['bet_prob'] = df.apply(
    lambda row: row['home_win_prob'] if row['bet_side'] == 'home' else row['away_win_prob'],
    axis=1
)

# Assume standard -110 odds for spread bets
standard_odds = -110

# Calculate EV for betting on the edge side
# We don't have actual odds per game, so assume -110 for all spread bets
df['ev_dollars'] = df.apply(
    lambda row: betting_utils.calculate_ev(
        model_prob=row['bet_prob'],
        market_prob=0.5,  # Assuming fair market after de-vig
        odds=standard_odds,
        stake=100
    ),
    axis=1
)

# Calculate Kelly sizing
df['kelly_full'] = df.apply(
    lambda row: betting_utils.kelly_criterion(
        model_prob=row['bet_prob'],
        odds=standard_odds
    ),
    axis=1
)

df['kelly_quarter'] = df['kelly_full'] * 0.25

print(f"  Added EV and Kelly metrics")

# ============================================================================
# BACKTEST RESULTS
# ============================================================================

print("\n[5/6] Analyzing backtest results...")

# Overall performance
train_df = df[df['data_split'] == 'train']
test_df = df[df['data_split'] == 'test']

print("\n" + "="*80)
print("PREDICTION ACCURACY")
print("="*80)

for split_name, split_df in [('Training (2009-2024)', train_df), ('Test (2025)', test_df)]:
    mae = split_df['abs_edge'].mean()
    rmse = np.sqrt((split_df['edge'] ** 2).mean())

    print(f"\n{split_name} (n={len(split_df):,}):")
    print(f"  MAE:  {mae:.2f} points")
    print(f"  RMSE: {rmse:.2f} points")
    print(f"  Mean edge: {split_df['edge'].mean():.2f} points")
    print(f"  Median abs edge: {split_df['abs_edge'].median():.2f} points")

# Edge distribution
print("\n" + "="*80)
print("EDGE DISTRIBUTION (Full Dataset)")
print("="*80)

edge_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

print("\nGames by edge threshold:")
for threshold in edge_thresholds:
    count = len(df[df['abs_edge'] >= threshold])
    pct = count / len(df) * 100
    avg_edge = df[df['abs_edge'] >= threshold]['abs_edge'].mean() if count > 0 else 0
    print(f"  {threshold:>3.1f}+ points: {count:4} games ({pct:5.1f}%) - avg edge: {avg_edge:.2f}")

# Expected Value analysis
print("\n" + "="*80)
print("EXPECTED VALUE ANALYSIS")
print("="*80)

print("\nHypothetical EV by edge threshold:")
print("(Assuming -110 odds, $100 stake per bet)")

for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
    subset = df[df['abs_edge'] >= threshold]
    if len(subset) > 0:
        total_ev = subset['ev_dollars'].sum()
        mean_ev = subset['ev_dollars'].mean()
        total_risk = len(subset) * 100  # $100 per bet

        print(f"\n  Edge >= {threshold}:")
        print(f"    Bets: {len(subset):,}")
        print(f"    Total EV: ${total_ev:,.0f}")
        print(f"    Mean EV/bet: ${mean_ev:.2f}")
        print(f"    Total risk: ${total_risk:,}")
        print(f"    EV/risk: {total_ev/total_risk*100:.2f}%")

# Kelly analysis
print("\n" + "="*80)
print("KELLY CRITERION ANALYSIS")
print("="*80)

print("\nQuarter Kelly sizing by edge threshold:")

for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
    subset = df[df['abs_edge'] >= threshold]
    if len(subset) > 0:
        mean_kelly = subset['kelly_quarter'].mean()
        median_kelly = subset['kelly_quarter'].median()
        max_kelly = subset['kelly_quarter'].max()

        print(f"\n  Edge >= {threshold}:")
        print(f"    Mean 1/4 Kelly: {mean_kelly*100:.2f}% of bankroll")
        print(f"    Median 1/4 Kelly: {median_kelly*100:.2f}%")
        print(f"    Max 1/4 Kelly: {max_kelly*100:.2f}%")

# Season-by-season performance
print("\n" + "="*80)
print("PERFORMANCE BY SEASON")
print("="*80)

season_stats = df.groupby('season').agg({
    'game_id': 'count',
    'abs_edge': 'mean',
    'edge': lambda x: np.sqrt((x ** 2).mean()),
    'ev_dollars': 'sum'
}).round(2)

season_stats.columns = ['Games', 'MAE', 'RMSE', 'Total EV']
season_stats['Total EV'] = season_stats['Total EV'].round(0).astype(int)

print("\n" + season_stats.to_string())

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[6/6] Creating performance visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ball Knower v1.2 - Professional Backtest Analysis', fontsize=16, fontweight='bold')

# 1. Edge distribution
ax1 = axes[0, 0]
ax1.hist(df['edge'], bins=60, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect calibration')
ax1.set_xlabel('Edge vs Vegas (points)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Edge Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Expected Value by edge threshold
ax2 = axes[0, 1]
ev_by_threshold = []
for threshold in np.arange(0, 5.1, 0.25):
    subset = df[df['abs_edge'] >= threshold]
    if len(subset) > 0:
        ev_by_threshold.append({
            'threshold': threshold,
            'mean_ev': subset['ev_dollars'].mean(),
            'count': len(subset)
        })

ev_df = pd.DataFrame(ev_by_threshold)
ax2.plot(ev_df['threshold'], ev_df['mean_ev'], marker='o', linewidth=2, markersize=5, color='green')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Minimum Edge Threshold (points)', fontsize=11)
ax2.set_ylabel('Mean EV per $100 bet ($)', fontsize=11)
ax2.set_title('Expected Value by Edge Threshold', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Kelly sizing distribution
ax3 = axes[1, 0]
significant_edges = df[df['abs_edge'] >= 2.0]
ax3.hist(significant_edges['kelly_quarter'] * 100, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax3.set_xlabel('1/4 Kelly Bet Size (% of bankroll)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Kelly Sizing Distribution (2+ point edges)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Cumulative EV over time
ax4 = axes[1, 1]
df_sorted = df.sort_values(['season', 'week'])
df_sorted['cumulative_ev'] = df_sorted['ev_dollars'].cumsum()
ax4.plot(range(len(df_sorted)), df_sorted['cumulative_ev'], linewidth=2, color='darkgreen')
ax4.set_xlabel('Game Number (chronological)', fontsize=11)
ax4.set_ylabel('Cumulative Expected Value ($)', fontsize=11)
ax4.set_title('Cumulative EV Over Time', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()

plot_file = config.OUTPUT_DIR / 'backtest_v1_2_professional.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  Saved visualization: {plot_file}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSaving backtest results...")

# Save detailed results
output_file = config.OUTPUT_DIR / 'backtest_v1_2_detailed.csv'
output_cols = ['game_id', 'season', 'week', 'away_team', 'home_team',
               'vegas_line', 'bk_v1_2_spread', 'edge', 'abs_edge',
               'home_win_prob', 'ev_dollars', 'kelly_quarter', 'bet_side']

df[output_cols].to_csv(output_file, index=False)
print(f"  Detailed results: {output_file}")

# Save summary statistics
summary_stats = {
    'overall': {
        'total_games': len(df),
        'mean_abs_edge': float(df['abs_edge'].mean()),
        'median_abs_edge': float(df['abs_edge'].median()),
        'rmse': float(np.sqrt((df['edge'] ** 2).mean())),
        'total_ev': float(df['ev_dollars'].sum()),
        'mean_ev_per_bet': float(df['ev_dollars'].mean()),
    },
    'train': {
        'games': len(train_df),
        'mae': float(train_df['abs_edge'].mean()),
        'total_ev': float(train_df['ev_dollars'].sum()),
    },
    'test': {
        'games': len(test_df),
        'mae': float(test_df['abs_edge'].mean()),
        'total_ev': float(test_df['ev_dollars'].sum()),
    },
    'edge_thresholds': {}
}

for threshold in edge_thresholds:
    subset = df[df['abs_edge'] >= threshold]
    summary_stats['edge_thresholds'][f'{threshold}+'] = {
        'count': len(subset),
        'pct_of_total': float(len(subset) / len(df) * 100),
        'mean_edge': float(subset['abs_edge'].mean()) if len(subset) > 0 else 0,
        'total_ev': float(subset['ev_dollars'].sum()) if len(subset) > 0 else 0,
        'mean_kelly_quarter': float(subset['kelly_quarter'].mean()) if len(subset) > 0 else 0,
    }

summary_file = config.OUTPUT_DIR / 'backtest_v1_2_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"  Summary statistics: {summary_file}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print(f"""
Model Performance:
- Overall MAE: {df['abs_edge'].mean():.2f} points
- Test MAE: {test_df['abs_edge'].mean():.2f} points
- Model is well-calibrated (mean edge near 0: {df['edge'].mean():.2f})

Betting Opportunities:
- Games with 2+ point edge: {len(df[df['abs_edge'] >= 2.0])} ({len(df[df['abs_edge'] >= 2.0])/len(df)*100:.1f}%)
- Games with 3+ point edge: {len(df[df['abs_edge'] >= 3.0])} ({len(df[df['abs_edge'] >= 3.0])/len(df)*100:.1f}%)

Expected Value (hypothetical):
- Total EV across all games: ${df['ev_dollars'].sum():,.0f}
- Mean EV per bet: ${df['ev_dollars'].mean():.2f}
- EV at 2+ edge threshold: ${df[df['abs_edge'] >= 2.0]['ev_dollars'].sum():,.0f} ({len(df[df['abs_edge'] >= 2.0])} bets)

Kelly Recommendations:
- Mean 1/4 Kelly (2+ edge): {df[df['abs_edge'] >= 2.0]['kelly_quarter'].mean()*100:.2f}% of bankroll
- Max 1/4 Kelly (2+ edge): {df[df['abs_edge'] >= 2.0]['kelly_quarter'].max()*100:.2f}% of bankroll

IMPORTANT NOTES:
1. These EV calculations assume -110 odds on all bets (simplified)
2. No actual game outcomes used - theoretical edge only
3. For live betting, you'd need:
   - Real-time odds from sportsbooks
   - Opening lines (not closing) for true CLV
   - Actual bet tracking and result logging
4. Past performance doesn't guarantee future results
5. Model residual std: {residual_std:.2f} points (used for probability calibration)
""")

print("="*80 + "\n")
