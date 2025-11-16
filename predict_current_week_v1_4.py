"""
Generate Week 11 predictions using Ball Knower v1.4

v1.4 uses enhanced EPA features with rolling averages and momentum indicators
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.stats import norm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse

print("\n" + "="*80)
print("BALL KNOWER v1.4 - WEEK 11 PREDICTIONS")
print("="*80)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n[1/7] Loading v1.4 model...")

model_path = project_root / 'output' / 'ball_knower_v1_4_model.json'

if not model_path.exists():
    print(f"  ✗ Model not found: {model_path}")
    print("  Run ball_knower_v1_4.py first to train the model")
    sys.exit(1)

with open(model_path, 'r') as f:
    model_data = json.load(f)

coefficients = pd.Series(model_data['coefficients'])
intercept = model_data['intercept']
residual_std = model_data.get('residual_std', 2.0)

print(f"  ✓ Loaded v1.4 model")
print(f"  Features: {len(coefficients)}")
print(f"  Intercept: {intercept:.4f}")
print(f"  Residual std: {residual_std:.4f}")

# ============================================================================
# LOAD CURRENT WEEK GAMES
# ============================================================================

print("\n[2/7] Loading Week 11 games...")

try:
    games = nflverse.games(season=2025, week=11)
    games = games[games['spread_line'].notna()].copy()

    # CRITICAL: Convert spread_line from away to home perspective
    games['spread_line'] = -1 * games['spread_line']

    print(f"  ✓ Loaded {len(games)} games with spreads")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# LOAD CURRENT NFELO + QB RATINGS
# ============================================================================

print("\n[3/7] Loading current nfelo ratings...")

try:
    nfelo_snapshot_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/elo_snapshot.csv'
    nfelo_snapshot = pd.read_csv(nfelo_snapshot_url)

    # Handle duplicate teams (take first occurrence)
    nfelo_snapshot = nfelo_snapshot.drop_duplicates(subset=['team'], keep='first')

    nfelo_home = nfelo_snapshot[['team', 'nfelo', 'qb_adj']].copy()
    nfelo_home.columns = ['home_team', 'home_nfelo', 'home_qb_adj']

    nfelo_away = nfelo_snapshot[['team', 'nfelo', 'qb_adj']].copy()
    nfelo_away.columns = ['away_team', 'away_nfelo', 'away_qb_adj']

    print(f"  ✓ Loaded nfelo for {len(nfelo_snapshot)} teams")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# ============================================================================
# LOAD EPA DATA AND CALCULATE ROLLING AVERAGES
# ============================================================================

print("\n[4/7] Loading EPA data and calculating rolling features...")

epa_path = project_root / 'data' / 'team_week_epa_2013_2024.csv'

if not epa_path.exists():
    print(f"  ✗ EPA data not found: {epa_path}")
    sys.exit(1)

epa_df = pd.read_csv(epa_path)

print(f"  ✓ Loaded {len(epa_df):,} team-week records")
print("  Calculating rolling averages...")

# Sort by team and time
epa_df = epa_df.sort_values(['team', 'season', 'week'])

# Calculate rolling averages for each team
enhanced_epa = []

for team in epa_df['team'].unique():
    team_data = epa_df[epa_df['team'] == team].copy()
    team_data = team_data.sort_values(['season', 'week'])

    # Rolling averages (3-game and 5-game)
    team_data['off_epa_roll3'] = team_data['off_epa_per_play'].rolling(window=3, min_periods=1).mean()
    team_data['def_epa_roll3'] = team_data['def_epa_per_play'].rolling(window=3, min_periods=1).mean()
    team_data['success_off_roll3'] = team_data['off_success_rate'].rolling(window=3, min_periods=1).mean()
    team_data['success_def_roll3'] = team_data['def_success_rate'].rolling(window=3, min_periods=1).mean()

    team_data['off_epa_roll5'] = team_data['off_epa_per_play'].rolling(window=5, min_periods=1).mean()
    team_data['def_epa_roll5'] = team_data['def_epa_per_play'].rolling(window=5, min_periods=1).mean()

    # Momentum (last 3 games vs previous 3 games)
    team_data['off_epa_momentum'] = (
        team_data['off_epa_per_play'].rolling(window=3, min_periods=1).mean() -
        team_data['off_epa_per_play'].shift(3).rolling(window=3, min_periods=1).mean()
    )

    enhanced_epa.append(team_data)

epa_enhanced = pd.concat(enhanced_epa, ignore_index=True)

print(f"  ✓ Calculated rolling features")

# Get most recent stats for each team (latest week in 2024)
latest_week = epa_enhanced[epa_enhanced['season'] == 2024]['week'].max()
current_epa = epa_enhanced[
    (epa_enhanced['season'] == 2024) &
    (epa_enhanced['week'] == latest_week)
].copy()

print(f"  Using 2024 Week {latest_week} as baseline")
print(f"  Coverage: {len(current_epa)} teams")

# ============================================================================
# BUILD MATCHUP FEATURES
# ============================================================================

print("\n[5/7] Building v1.4 features for Week 11 matchups...")

# Merge nfelo
matchups = games.merge(
    nfelo_home,
    on='home_team',
    how='left'
)
matchups = matchups.merge(
    nfelo_away,
    on='away_team',
    how='left'
)

# Merge EPA (home)
epa_home = current_epa[[
    'team', 'off_epa_roll3', 'def_epa_roll3',
    'success_off_roll3', 'success_def_roll3',
    'off_epa_roll5', 'def_epa_roll5', 'off_epa_momentum'
]].copy()
epa_home.columns = [
    'home_team', 'home_off_epa_roll3', 'home_def_epa_roll3',
    'home_success_off_roll3', 'home_success_def_roll3',
    'home_off_epa_roll5', 'home_def_epa_roll5', 'home_momentum'
]

matchups = matchups.merge(epa_home, on='home_team', how='left')

# Merge EPA (away)
epa_away = current_epa[[
    'team', 'off_epa_roll3', 'def_epa_roll3',
    'success_off_roll3', 'success_def_roll3',
    'off_epa_roll5', 'def_epa_roll5', 'off_epa_momentum'
]].copy()
epa_away.columns = [
    'away_team', 'away_off_epa_roll3', 'away_def_epa_roll3',
    'away_success_off_roll3', 'away_success_def_roll3',
    'away_off_epa_roll5', 'away_def_epa_roll5', 'away_momentum'
]

matchups = matchups.merge(epa_away, on='away_team', how='left')

# Calculate differentials
matchups['nfelo_diff'] = matchups['home_nfelo'] - matchups['away_nfelo']
matchups['qb_diff'] = matchups['home_qb_adj'] - matchups['away_qb_adj']

# EPA differentials (rolling averages)
matchups['epa_off_diff_roll3'] = matchups['home_off_epa_roll3'] - matchups['away_off_epa_roll3']
matchups['epa_def_diff_roll3'] = matchups['home_def_epa_roll3'] - matchups['away_def_epa_roll3']
matchups['success_off_diff_roll3'] = matchups['home_success_off_roll3'] - matchups['away_success_off_roll3']
matchups['success_def_diff_roll3'] = matchups['home_success_def_roll3'] - matchups['away_success_def_roll3']

matchups['epa_off_diff_roll5'] = matchups['home_off_epa_roll5'] - matchups['away_off_epa_roll5']
matchups['epa_def_diff_roll5'] = matchups['home_def_epa_roll5'] - matchups['away_def_epa_roll5']

# Momentum differential
matchups['momentum_diff'] = matchups['home_momentum'] - matchups['away_momentum']

# Interaction term
matchups['epa_elo_interaction'] = matchups['epa_off_diff_roll3'] * matchups['nfelo_diff']

# Baseline features (same as v1.2)
matchups['rest_advantage'] = 0  # Unknown for upcoming games
matchups['div_game'] = 0        # Unknown for upcoming games
matchups['surface_mod'] = 0     # Unknown
matchups['time_advantage'] = 0  # Unknown

print(f"  ✓ Created 14 v1.4 features")

# Check for missing values
missing = matchups[coefficients.index].isna().sum().sum()
if missing > 0:
    print(f"  ⚠ Warning: {missing} missing values - filling with 0")
    matchups[coefficients.index] = matchups[coefficients.index].fillna(0)

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n[6/7] Generating v1.4 predictions...")

X = matchups[coefficients.index]
matchups['predicted_spread'] = X @ coefficients + intercept

# Convert to probabilities
matchups['home_win_prob'] = norm.cdf(-matchups['predicted_spread'] / residual_std)

# Calculate edge vs spread_line
matchups['edge'] = matchups['predicted_spread'] - matchups['spread_line']

print(f"  ✓ Generated predictions for {len(matchups)} games")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n[7/7] Week 11 Predictions (v1.4)...")
print("\n" + "="*80)

results = matchups[[
    'home_team', 'away_team', 'spread_line',
    'predicted_spread', 'edge', 'home_win_prob'
]].copy()

results = results.sort_values('edge', ascending=False)

print("\nAll Week 11 Games:")
print("-" * 80)
for _, row in results.iterrows():
    home = row['home_team']
    away = row['away_team']
    line = row['spread_line']
    pred = row['predicted_spread']
    edge = row['edge']
    prob = row['home_win_prob'] * 100

    print(f"{home:3s} vs {away:3s} | Line: {line:+6.1f} | Pred: {pred:+6.2f} | Edge: {edge:+5.2f} | Home Win: {prob:4.1f}%")

# Value bets (2+ point edge)
print("\n" + "="*80)
print("VALUE BETS (2+ point edge):")
print("="*80)

value_bets = results[abs(results['edge']) >= 2.0]

if len(value_bets) == 0:
    print("No value bets identified this week")
else:
    for _, row in value_bets.iterrows():
        home = row['home_team']
        away = row['away_team']
        line = row['spread_line']
        pred = row['predicted_spread']
        edge = row['edge']
        prob = row['home_win_prob'] * 100

        if edge > 0:
            rec = f"BET {home} (Home)"
        else:
            rec = f"BET {away} (Away)"

        print(f"\n{rec}")
        print(f"  Matchup:    {home} vs {away}")
        print(f"  Line:       {line:+.1f}")
        print(f"  Prediction: {pred:+.2f}")
        print(f"  Edge:       {edge:+.2f} points")
        print(f"  Home Win %: {prob:.1f}%")

# ============================================================================
# COMPARE TO v1.2 AND v1.3
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON (v1.2 vs v1.3 vs v1.4)")
print("="*80)

# Load v1.2 predictions
v1_2_path = project_root / 'output' / 'week_11_value_bets_v1_2.csv'
if v1_2_path.exists():
    v1_2_preds = pd.read_csv(v1_2_path)
    v1_2_preds = v1_2_preds[['home_team', 'away_team', 'bk_v1_2_spread', 'edge']].copy()
    v1_2_preds.columns = ['home_team', 'away_team', 'pred_v1_2', 'edge_v1_2']
else:
    print("\n⚠ v1.2 predictions not found")
    v1_2_preds = None

# Load v1.3 predictions
v1_3_path = project_root / 'output' / 'week_11_value_bets_v1_3.csv'
if v1_3_path.exists():
    v1_3_preds = pd.read_csv(v1_3_path)
    v1_3_preds = v1_3_preds[['home_team', 'away_team', 'bk_v1_3_spread', 'edge']].copy()
    v1_3_preds.columns = ['home_team', 'away_team', 'pred_v1_3', 'edge_v1_3']
else:
    print("\n⚠ v1.3 predictions not found")
    v1_3_preds = None

# Merge all predictions
comparison = results[['home_team', 'away_team', 'spread_line', 'predicted_spread', 'edge']].copy()
comparison.columns = ['home_team', 'away_team', 'spread_line', 'pred_v1_4', 'edge_v1_4']

if v1_2_preds is not None:
    comparison = comparison.merge(v1_2_preds, on=['home_team', 'away_team'], how='left')

if v1_3_preds is not None:
    comparison = comparison.merge(v1_3_preds, on=['home_team', 'away_team'], how='left')

# Calculate differences
if v1_2_preds is not None:
    comparison['edge_diff_v1_4_vs_v1_2'] = comparison['edge_v1_4'] - comparison['edge_v1_2']

if v1_3_preds is not None:
    comparison['edge_diff_v1_4_vs_v1_3'] = comparison['edge_v1_4'] - comparison['edge_v1_3']

print("\nEdge Comparison Across Models:")
print("-" * 80)

if v1_2_preds is not None and v1_3_preds is not None:
    for _, row in comparison.iterrows():
        print(f"\n{row['home_team']} vs {row['away_team']} (Line: {row['spread_line']:+.1f})")
        print(f"  v1.2 Edge: {row['edge_v1_2']:+5.2f}")
        print(f"  v1.3 Edge: {row['edge_v1_3']:+5.2f}")
        print(f"  v1.4 Edge: {row['edge_v1_4']:+5.2f}")
        print(f"  Δ v1.4-v1.2: {row['edge_diff_v1_4_vs_v1_2']:+5.2f}")
        print(f"  Δ v1.4-v1.3: {row['edge_diff_v1_4_vs_v1_3']:+5.2f}")
elif v1_2_preds is not None:
    for _, row in comparison.iterrows():
        print(f"\n{row['home_team']} vs {row['away_team']} (Line: {row['spread_line']:+.1f})")
        print(f"  v1.2 Edge: {row['edge_v1_2']:+5.2f}")
        print(f"  v1.4 Edge: {row['edge_v1_4']:+5.2f}")
        print(f"  Δ v1.4-v1.2: {row['edge_diff_v1_4_vs_v1_2']:+5.2f}")

# Summary statistics
print("\n" + "="*80)
print("EDGE STATISTICS")
print("="*80)

if v1_2_preds is not None and v1_3_preds is not None:
    print(f"\nMean Absolute Edge:")
    print(f"  v1.2: {comparison['edge_v1_2'].abs().mean():.2f} points")
    print(f"  v1.3: {comparison['edge_v1_3'].abs().mean():.2f} points")
    print(f"  v1.4: {comparison['edge_v1_4'].abs().mean():.2f} points")

    print(f"\nMean Edge Change:")
    print(f"  v1.4 vs v1.2: {comparison['edge_diff_v1_4_vs_v1_2'].abs().mean():.2f} points")
    print(f"  v1.4 vs v1.3: {comparison['edge_diff_v1_4_vs_v1_3'].abs().mean():.2f} points")

# Save output
output_file = project_root / 'output' / 'week_11_predictions_v1_4.csv'
results.to_csv(output_file, index=False)

comparison_file = project_root / 'output' / 'week_11_model_comparison.csv'
comparison.to_csv(comparison_file, index=False)

print(f"\n✓ Saved predictions to: {output_file}")
print(f"✓ Saved comparison to: {comparison_file}")

print("\n" + "="*80)
