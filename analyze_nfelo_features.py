"""
Analyze available features in greerreNFL/nfelo data

Investigates what advanced metrics are already available
that we can use for v1.3 without needing separate EPA data
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("NFELO DATA FEATURE ANALYSIS")
print("="*80)

# Load nfelo historical data
print("\nLoading nfelo games data...")
nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

print(f"  Loaded {len(df):,} games")
print(f"  Columns: {len(df.columns)}")

print("\n" + "="*80)
print("ALL AVAILABLE COLUMNS")
print("="*80)

# Categorize columns
basic_cols = []
team_strength_cols = []
situational_cols = []
outcome_cols = []
betting_cols = []
other_cols = []

for col in df.columns:
    col_lower = col.lower()

    if any(x in col_lower for x in ['game_id', 'season', 'week', 'team', 'date']):
        basic_cols.append(col)
    elif any(x in col_lower for x in ['elo', 'rating', 'strength', 'qb', 'value_adj']):
        team_strength_cols.append(col)
    elif any(x in col_lower for x in ['rest', 'bye', 'div', 'surface', 'time', 'travel']):
        situational_cols.append(col)
    elif any(x in col_lower for x in ['score', 'result', 'win', 'margin']):
        outcome_cols.append(col)
    elif any(x in col_lower for x in ['line', 'spread', 'total', 'odds']):
        betting_cols.append(col)
    else:
        other_cols.append(col)

print("\nBasic Info:")
for col in basic_cols:
    print(f"  - {col}")

print("\nTeam Strength/Rating:")
for col in team_strength_cols:
    print(f"  - {col}")

print("\nSituational Factors:")
for col in situational_cols:
    print(f"  - {col}")

print("\nGame Outcomes:")
for col in outcome_cols:
    print(f"  - {col}")

print("\nBetting Lines:")
for col in betting_cols:
    print(f"  - {col}")

print("\nOther:")
for col in other_cols:
    print(f"  - {col}")

# ============================================================================
# VALUE-ADJUSTED ELO
# ============================================================================

print("\n" + "="*80)
print("VALUE-ADJUSTED ELO ANALYSIS")
print("="*80)

# Check if value_adj columns exist
value_adj_cols = [col for col in df.columns if 'value_adj' in col.lower()]

if value_adj_cols:
    print(f"\nFound {len(value_adj_cols)} value-adjusted columns:")
    for col in value_adj_cols:
        print(f"  - {col}")

    # Sample data
    print("\nSample game with value adjustments:")
    sample = df[df[value_adj_cols].notna().all(axis=1)].iloc[0]

    print(f"  Game: {sample['game_id']}")
    for col in value_adj_cols:
        print(f"    {col}: {sample[col]}")
else:
    print("\nNo value-adjusted ELO columns found")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE CORRELATION WITH SPREAD")
print("="*80)

# Calculate correlations with actual margin
df_complete = df[df['home_line_close'].notna() & df['home_result_spread'].notna()].copy()

if len(df_complete) > 0:
    # Calculate ELO differential
    df_complete['nfelo_diff'] = (df_complete['starting_nfelo_home'] -
                                   df_complete['starting_nfelo_away'])

    # Features to test
    feature_candidates = {
        'nfelo_diff': df_complete['nfelo_diff'],
        'home_line_close': df_complete['home_line_close'],
    }

    # Add any value_adj differentials
    if 'starting_value_adj_home' in df.columns and 'starting_value_adj_away' in df.columns:
        df_complete['value_adj_diff'] = (df_complete['starting_value_adj_home'] -
                                          df_complete['starting_value_adj_away'])
        feature_candidates['value_adj_diff'] = df_complete['value_adj_diff']

    # Add situational features
    situational_features = ['home_bye_mod', 'away_bye_mod', 'div_game_mod',
                           'dif_surface_mod', 'home_time_advantage_mod']

    for feat in situational_features:
        if feat in df.columns:
            feature_candidates[feat] = df_complete[feat].fillna(0)

    # Calculate correlations
    print("\nCorrelation with actual result (home_result_spread):")
    print(f"{'Feature':<30s} {'Correlation':>12s} {'Abs Corr':>12s}")
    print("-" * 56)

    correlations = []
    for feat_name, feat_values in feature_candidates.items():
        corr = feat_values.corr(df_complete['home_result_spread'])
        correlations.append({
            'feature': feat_name,
            'corr': corr,
            'abs_corr': abs(corr)
        })

    # Sort by absolute correlation
    correlations = sorted(correlations, key=lambda x: x['abs_corr'], reverse=True)

    for c in correlations:
        print(f"{c['feature']:<30s} {c['corr']:>12.4f} {c['abs_corr']:>12.4f}")

# ============================================================================
# MISSING FEATURES ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("MISSING FEATURES (vs Professional Research)")
print("="*80)

print("""
What nfelo HAS:
  ✓ Team strength (ELO ratings)
  ✓ QB adjustments (538 QB value)
  ✓ Rest/bye week effects
  ✓ Divisional game indicators
  ✓ Surface differences
  ✓ Travel/timezone effects
  ✓ Vegas closing lines (for training targets)

What nfelo LACKS (from professional research):
  ✗ EPA (Expected Points Added) per play
  ✗ Success rate metrics
  ✗ Explosive play rates
  ✗ Recent form (rolling averages)
  ✗ Schedule-adjusted performance (DVOA-like)
  ✗ Weather data (temperature, wind)
  ✗ Pass vs run efficiency splits
  ✗ Situational efficiency (3rd down, red zone)

Implications for v1.3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION A: Work with Available Data
  - v1.3 focuses on feature engineering from nfelo
  - Build rolling averages of ELO changes
  - Create interaction features (e.g., rest * ELO diff)
  - Add polynomial features for nonlinear relationships
  - Expected improvement: Modest (5-10%)

OPTION B: Use Alternative EPA Source
  - Download pre-computed EPA data manually
  - Store locally in data/ directory
  - Build EPA features as originally planned
  - Expected improvement: Significant (15-25%)

OPTION C: Skip to STEP 3 (Score Prediction)
  - Build score-per-team model with current features
  - Focus on architecture improvement over feature improvement
  - Can still provide CLV and EV analysis
  - Expected improvement: Significant (20-30%)

RECOMMENDATION: OPTION C
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why skip to STEP 3:
1. Network restrictions prevent EPA data access
2. Current nfelo features are already strong (MAE 1.57)
3. Bigger gains from better architecture than more features
4. Score prediction unlocks:
   - Moneyline betting
   - Over/under betting
   - True win probabilities (not just spread)
   - Full game simulations

We can always revisit EPA features later when network access improves.
""")

print("="*80 + "\n")
