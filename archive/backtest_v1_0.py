"""
Backtest Ball Knower v1.0 Against Historical Data

Uses nfelo historical games (2009-2025, 4,510 games) to evaluate:
- Prediction accuracy vs Vegas lines
- ROI by edge threshold
- Performance across seasons
- Calibration quality

Model: spread = 2.67 + (nfelo_diff × 0.0447)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Model parameters (calibrated from Week 11 2025)
NFELO_COEF = 0.0447
INTERCEPT = 2.67

print("\n" + "="*80)
print("BALL KNOWER v1.0 - HISTORICAL BACKTEST")
print("="*80)
print(f"\nModel: spread = {INTERCEPT} + ({NFELO_COEF} × nfelo_diff)")

# ============================================================================
# LOAD HISTORICAL DATA
# ============================================================================

print("\n[1/5] Loading nfelo historical data...")

# Load nfelo games with ELO ratings and Vegas lines
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

print(f"✓ Loaded {len(df):,} games from nfelo database")

# Extract season/week/teams from game_id (format: YYYY_WW_AWAY_HOME)
df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

print(f"Coverage: {df['season'].min()}-{df['season'].max()}")
print(f"Games with closing lines: {df['home_line_close'].notna().sum():,}")

# Filter to games with both ELO ratings and Vegas lines
df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()

print(f"Games with complete data: {len(df):,}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n[2/5] Generating Ball Knower v1.0 predictions...")

# Calculate ELO differential (home perspective)
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']

# Ball Knower v1.0 prediction
df['bk_v1_0_spread'] = INTERCEPT + (df['nfelo_diff'] * NFELO_COEF)

# Edge vs Vegas closing line (negative = model favors home more than Vegas)
df['edge'] = df['bk_v1_0_spread'] - df['home_line_close']
df['abs_edge'] = df['edge'].abs()

# Compare to nfelo's own predictions
df['nfelo_prediction'] = df['nfelo_home_line_close']
df['nfelo_edge'] = df['nfelo_prediction'] - df['home_line_close']

print(f"✓ Generated {len(df):,} predictions")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print("\n[3/5] Calculating performance metrics...")

# Overall accuracy
mae = df['abs_edge'].mean()
rmse = np.sqrt((df['edge'] ** 2).mean())
r_squared = 1 - (df['edge'] ** 2).sum() / ((df['home_line_close'] - df['home_line_close'].mean()) ** 2).sum()

print("\n" + "="*80)
print("OVERALL PERFORMANCE (2009-2025)")
print("="*80)

print(f"\nBall Knower v1.0:")
print(f"  MAE:         {mae:.2f} points")
print(f"  RMSE:        {rmse:.2f} points")
print(f"  R²:          {r_squared:.3f}")
print(f"  Mean Edge:   {df['edge'].mean():.2f} points")

# Compare to nfelo
nfelo_mae = df['nfelo_edge'].abs().mean()
nfelo_rmse = np.sqrt((df['nfelo_edge'] ** 2).mean())

print(f"\nnfelo (for comparison):")
print(f"  MAE:         {nfelo_mae:.2f} points")
print(f"  RMSE:        {nfelo_rmse:.2f} points")

# Performance by season
print("\n" + "="*80)
print("PERFORMANCE BY SEASON")
print("="*80)

season_stats = df.groupby('season').agg({
    'game_id': 'count',
    'abs_edge': 'mean',
    'edge': lambda x: np.sqrt((x ** 2).mean())  # RMSE
}).round(2)

season_stats.columns = ['Games', 'MAE', 'RMSE']
print("\n" + season_stats.to_string())

# ============================================================================
# EDGE ANALYSIS
# ============================================================================

print("\n[4/5] Analyzing edge distribution...")

print("\n" + "="*80)
print("EDGE DISTRIBUTION")
print("="*80)

# Create edge bins
bins = [-np.inf, -5, -3, -2, -1, 0, 1, 2, 3, 5, np.inf]
labels = ['<-5', '-5 to -3', '-3 to -2', '-2 to -1', '-1 to 0',
          '0 to 1', '1 to 2', '2 to 3', '3 to 5', '>5']

df['edge_bin'] = pd.cut(df['edge'], bins=bins, labels=labels)

edge_dist = df['edge_bin'].value_counts().sort_index()
print("\nEdge distribution (points):")
print(edge_dist.to_string())

# Games with significant edges (2+ points)
sig_edges = df[df['abs_edge'] >= 2.0]
print(f"\n\nGames with 2+ point edge: {len(sig_edges):,} ({len(sig_edges)/len(df)*100:.1f}%)")
print(f"  Mean edge magnitude: {sig_edges['abs_edge'].mean():.2f} points")
print(f"  Max edge: {df['abs_edge'].max():.2f} points")

# ============================================================================
# HYPOTHETICAL BETTING ANALYSIS
# ============================================================================

print("\n[5/5] Calculating hypothetical betting ROI...")

print("\n" + "="*80)
print("HYPOTHETICAL BETTING ANALYSIS")
print("="*80)

print("\nNOTE: This analysis assumes:")
print("  - Betting spreads at -110 odds (risk $110 to win $100)")
print("  - Perfect discipline (only bet when edge >= threshold)")
print("  - Closing line value (bet at closing line)")
print("  - NO ACTUAL OUTCOMES - showing theoretical edge realization")

# Analyze different edge thresholds
thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

threshold_results = []

for threshold in thresholds:
    bets = df[df['abs_edge'] >= threshold].copy()

    if len(bets) > 0:
        # Mean edge at this threshold
        mean_edge = bets['abs_edge'].mean()

        # Games per season at this threshold
        games_per_season = len(bets) / df['season'].nunique()

        threshold_results.append({
            'Threshold': f'{threshold}+',
            'Bets': len(bets),
            'Per Season': round(games_per_season, 1),
            'Mean Edge': round(mean_edge, 2),
            'Pct of Games': f"{len(bets)/len(df)*100:.1f}%"
        })

threshold_df = pd.DataFrame(threshold_results)
print("\n" + threshold_df.to_string(index=False))

print("\n" + "="*80)
print("IMPORTANT DISCLAIMERS")
print("="*80)

print("""
1. LOOK-AHEAD BIAS: This model was calibrated on 2025 Week 11 data and applied
   to historical data. In reality, we wouldn't have known these coefficients
   in the past. True out-of-sample testing requires time-series cross-validation.

2. NO ACTUAL RESULTS: This backtest only measures edge vs Vegas, not actual
   game outcomes. To calculate real ROI, we'd need to know which side covered.

3. CLOSING LINE ASSUMPTION: Assumes betting at closing lines. Real betting
   requires finding edges at open/mid-week lines before the market adjusts.

4. SIMPLIFIED MODEL: v1.0 uses only ELO ratings. Missing situational factors
   (injuries, weather, rest, etc.) that affect real games.

5. SMALL CALIBRATION SAMPLE: Model was calibrated on just 14 Week 11 games.
   Coefficients may not generalize well across 16 years of data.

For proper backtesting, see v1.2 which will use historical training/test splits.
""")

# Save backtest results
output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

backtest_file = output_dir / 'backtest_v1_0_results.csv'
df.to_csv(backtest_file, index=False)

print(f"\nBacktest results saved to: {backtest_file}")

# Generate summary plot
print("\n[Bonus] Creating performance visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ball Knower v1.0 - Historical Backtest (2009-2025)', fontsize=16, fontweight='bold')

# 1. MAE by season
ax1 = axes[0, 0]
season_mae = df.groupby('season')['abs_edge'].mean()
ax1.plot(season_mae.index, season_mae.values, marker='o', linewidth=2, markersize=6)
ax1.axhline(y=mae, color='r', linestyle='--', label=f'Overall MAE: {mae:.2f}')
ax1.set_xlabel('Season', fontsize=11)
ax1.set_ylabel('Mean Absolute Edge (points)', fontsize=11)
ax1.set_title('Prediction Error by Season', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Edge distribution
ax2 = axes[0, 1]
ax2.hist(df['edge'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Calibration')
ax2.set_xlabel('Edge vs Vegas (points)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Edge Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Games by edge threshold
ax3 = axes[1, 0]
threshold_counts = [len(df[df['abs_edge'] >= t]) for t in thresholds]
ax3.bar(range(len(thresholds)), threshold_counts, color='steelblue', edgecolor='black')
ax3.set_xticks(range(len(thresholds)))
ax3.set_xticklabels([f'{t}+' for t in thresholds])
ax3.set_xlabel('Edge Threshold (points)', fontsize=11)
ax3.set_ylabel('Number of Bets', fontsize=11)
ax3.set_title('Betting Opportunities by Threshold', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Predicted vs Vegas
ax4 = axes[1, 1]
sample = df.sample(min(500, len(df)))  # Sample for readability
ax4.scatter(sample['home_line_close'], sample['bk_v1_0_spread'], alpha=0.5, s=20)
ax4.plot([-20, 20], [-20, 20], 'r--', linewidth=2, label='Perfect Agreement')
ax4.set_xlabel('Vegas Closing Line', fontsize=11)
ax4.set_ylabel('Ball Knower v1.0 Prediction', fontsize=11)
ax4.set_title(f'Predictions vs Vegas (R² = {r_squared:.3f})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-20, 20)
ax4.set_ylim(-20, 20)

plt.tight_layout()

plot_file = output_dir / 'backtest_v1_0_performance.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Performance visualization saved to: {plot_file}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80 + "\n")
