"""
BALL KNOWER v2.1 - HYBRID MATCHUP MODEL

Combines the best of both worlds:
- v1.4 features (baseline, EPA, NGS) - which worked great (1.42 MAE)
- v2.0 matchup features (scheme vs scheme) - new insights

This predicts SPREADS (not raw point differentials) but with richer matchup context.
We then compare our spread to Vegas spread to find value.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import json

print("\n" + "="*80)
print("BALL KNOWER v2.1 - HYBRID MATCHUP MODEL")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/7] Loading all data sources...")

# Schedules
schedules = pd.read_parquet('/home/user/BK_Build/schedules.parquet')
schedules = schedules[schedules['game_type'] == 'REG'].copy()
schedules = schedules[schedules['spread_line'].notna()].copy()

# ELO
nfelo = pd.read_csv('/home/user/BK_Build/nfelo_2002_2025.csv')

# QB values
qb_vals = pd.read_csv('/home/user/BK_Build/qb_values_2002_2025.csv')

# EPA data
epa_data = pd.read_csv('/home/user/BK_Build/team_week_epa_2013_2025.csv')

# NGS data
ngs_data = pd.read_csv('/home/user/BK_Build/team_week_ngs_2016_2025.csv')

# Team profiles (NEW)
team_profiles = pd.read_csv('/home/user/BK_Build/team_profiles_advanced_2018_2025.csv')

print(f"✓ Loaded all data sources")
print(f"  Schedules: {len(schedules):,} games")
print(f"  Team profiles: {len(team_profiles):,} team-weeks")

# ============================================================================
# CREATE ROLLING FEATURES
# ============================================================================

print("\n[2/7] Creating rolling features...")

# Sort for rolling calculations
epa_data = epa_data.sort_values(['team', 'season', 'week'])
ngs_data = ngs_data.sort_values(['team', 'season', 'week'])
team_profiles = team_profiles.sort_values(['team', 'season', 'week'])

# EPA rolling windows
windows = [3, 5, 10]
for window in windows:
    epa_data[f'off_epa_L{window}'] = (
        epa_data.groupby('team')['off_epa_per_play']
        .shift(1).rolling(window, min_periods=1).mean()
    )
    epa_data[f'def_epa_L{window}'] = (
        epa_data.groupby('team')['def_epa_per_play']
        .shift(1).rolling(window, min_periods=1).mean()
    )
    epa_data[f'epa_margin_L{window}'] = (
        epa_data[f'off_epa_L{window}'] - epa_data[f'def_epa_L{window}']
    )

# NGS rolling windows
ngs_metrics = ['cpoe', 'avg_time_to_throw', 'aggressiveness',
               'rush_efficiency', 'avg_separation']
for window in [3, 5]:
    for metric in ngs_metrics:
        if metric in ngs_data.columns:
            ngs_data[f'{metric}_L{window}'] = (
                ngs_data.groupby('team')[metric]
                .shift(1).rolling(window, min_periods=1).mean()
            )

# Team profile rolling windows (NEW)
profile_metrics = [
    'def_coverage_quality', 'def_pressure_rate', 'def_tackle_efficiency',
    'off_oline_quality', 'off_passing_accuracy', 'off_wr_quality'
]
for window in [3, 5, 10]:
    for metric in profile_metrics:
        if metric in team_profiles.columns:
            team_profiles[f'{metric}_L{window}'] = (
                team_profiles.groupby('team')[metric]
                .shift(1).rolling(window, min_periods=1).mean()
            )

print(f"✓ Created rolling features for EPA, NGS, and team profiles")

# ============================================================================
# BUILD FEATURE SET
# ============================================================================

print("\n[3/7] Building comprehensive feature set...")

# Merge all data to schedules
df = schedules.copy()

# Add ELO differentials
df = df.merge(
    nfelo[['season', 'week', 'team', 'nfelo_pre']],
    left_on=['season', 'week', 'home_team'],
    right_on=['season', 'week', 'team'],
    how='left'
).drop('team', axis=1).rename(columns={'nfelo_pre': 'home_nfelo'})

df = df.merge(
    nfelo[['season', 'week', 'team', 'nfelo_pre']],
    left_on=['season', 'week', 'away_team'],
    right_on=['season', 'week', 'team'],
    how='left'
).drop('team', axis=1).rename(columns={'nfelo_pre': 'away_nfelo'})

df['nfelo_diff'] = df['home_nfelo'] - df['away_nfelo']

# Add QB values
df = df.merge(
    qb_vals[['season', 'week', 'team', 'qb_value_pre']],
    left_on=['season', 'week', 'home_team'],
    right_on=['season', 'week', 'team'],
    how='left'
).drop('team', axis=1).rename(columns={'qb_value_pre': 'home_qb_value'})

df = df.merge(
    qb_vals[['season', 'week', 'team', 'qb_value_pre']],
    left_on=['season', 'week', 'away_team'],
    right_on=['season', 'week', 'team'],
    how='left'
).drop('team', axis=1).rename(columns={'qb_value_pre': 'away_qb_value'})

df['qb_diff'] = df['home_qb_value'].fillna(0) - df['away_qb_value'].fillna(0)

# Context features
df['rest_advantage'] = df['home_rest'] - df['away_rest']
df['div_game'] = (df['div_game'] == True).astype(int)

# Surface advantage (home team perspective)
df['surface_mod'] = 0
df.loc[(df['surface'] == 'grass') & (df['roof'] == 'outdoors'), 'surface_mod'] = 1
df.loc[df['roof'] == 'dome', 'surface_mod'] = -1

# Timezone advantage
df['time_advantage'] = 0
for _, row in df.iterrows():
    if pd.notna(row['gametime']) and ':' in str(row['gametime']):
        hour = int(str(row['gametime']).split(':')[0])
        # West coast home team in early game
        if row['home_team'] in ['SF', 'LAC', 'LAR', 'SEA', 'LV'] and hour < 13:
            df.loc[df.index == row.name, 'time_advantage'] = -1
        # East coast away team in late game
        elif row['away_team'] in ['BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE',
                                   'PIT', 'NYG', 'PHI', 'WAS'] and hour >= 16:
            df.loc[df.index == row.name, 'time_advantage'] = -1

print(f"✓ Added v1.4 baseline features (ELO, QB, context)")

# ============================================================================
# ADD EPA DIFFERENTIALS
# ============================================================================

print("\n[4/7] Adding EPA differentials...")

for window in windows:
    # Home team EPA
    df = df.merge(
        epa_data[['season', 'week', 'team', f'off_epa_L{window}',
                  f'def_epa_L{window}', f'epa_margin_L{window}']],
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_drop')
    )
    df = df.rename(columns={
        f'off_epa_L{window}': f'home_off_epa_L{window}',
        f'def_epa_L{window}': f'home_def_epa_L{window}',
        f'epa_margin_L{window}': f'home_epa_margin_L{window}'
    })
    df = df.drop([col for col in df.columns if 'drop' in col], axis=1)

    # Away team EPA
    df = df.merge(
        epa_data[['season', 'week', 'team', f'off_epa_L{window}',
                  f'def_epa_L{window}', f'epa_margin_L{window}']],
        left_on=['season', 'week', 'away_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_drop')
    )
    df = df.rename(columns={
        f'off_epa_L{window}': f'away_off_epa_L{window}',
        f'def_epa_L{window}': f'away_def_epa_L{window}',
        f'epa_margin_L{window}': f'away_epa_margin_L{window}'
    })
    df = df.drop([col for col in df.columns if 'drop' in col], axis=1)

    # Differentials
    df[f'epa_margin_diff_L{window}'] = (
        df[f'home_epa_margin_L{window}'] - df[f'away_epa_margin_L{window}']
    )
    df[f'epa_off_diff_L{window}'] = (
        df[f'home_off_epa_L{window}'] - df[f'away_off_epa_L{window}']
    )
    df[f'epa_def_diff_L{window}'] = (
        df[f'home_def_epa_L{window}'] - df[f'away_def_epa_L{window}']
    )

print(f"✓ Added {len(windows) * 3} EPA differential features")

# ============================================================================
# ADD NGS DIFFERENTIALS
# ============================================================================

print("\n[5/7] Adding NGS differentials...")

for window in [3, 5]:
    for metric in ngs_metrics:
        col = f'{metric}_L{window}'
        if col not in ngs_data.columns:
            continue

        # Home team
        df = df.merge(
            ngs_data[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'home_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        # Away team
        df = df.merge(
            ngs_data[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'away_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        # Differential
        df[f'{metric}_diff_L{window}'] = df[f'home_{col}'] - df[f'away_{col}']

print(f"✓ Added {2 * len(ngs_metrics)} NGS differential features")

# ============================================================================
# ADD MATCHUP FEATURES (NEW!)
# ============================================================================

print("\n[6/7] Adding matchup-specific features...")

matchup_count = 0

for window in windows:
    # Get home and away profiles
    for metric in profile_metrics:
        col = f'{metric}_L{window}'
        if col not in team_profiles.columns:
            continue

        # Home team profile
        df = df.merge(
            team_profiles[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'home_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        # Away team profile
        df = df.merge(
            team_profiles[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'away_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

    # Create matchup features
    # Home offense vs Away defense
    if f'home_off_passing_accuracy_L{window}' in df.columns and f'away_def_coverage_quality_L{window}' in df.columns:
        df[f'pass_matchup_home_L{window}'] = (
            df[f'home_off_passing_accuracy_L{window}'] - df[f'away_def_coverage_quality_L{window}']
        )
        matchup_count += 1

    if f'home_off_oline_quality_L{window}' in df.columns and f'away_def_pressure_rate_L{window}' in df.columns:
        df[f'oline_matchup_home_L{window}'] = (
            df[f'home_off_oline_quality_L{window}'] - df[f'away_def_pressure_rate_L{window}']
        )
        matchup_count += 1

    if f'home_off_wr_quality_L{window}' in df.columns and f'away_def_coverage_quality_L{window}' in df.columns:
        df[f'wr_matchup_home_L{window}'] = (
            df[f'home_off_wr_quality_L{window}'] - df[f'away_def_coverage_quality_L{window}']
        )
        matchup_count += 1

    # Away offense vs Home defense
    if f'away_off_passing_accuracy_L{window}' in df.columns and f'home_def_coverage_quality_L{window}' in df.columns:
        df[f'pass_matchup_away_L{window}'] = (
            df[f'away_off_passing_accuracy_L{window}'] - df[f'home_def_coverage_quality_L{window}']
        )
        matchup_count += 1

    if f'away_off_oline_quality_L{window}' in df.columns and f'home_def_pressure_rate_L{window}' in df.columns:
        df[f'oline_matchup_away_L{window}'] = (
            df[f'away_off_oline_quality_L{window}'] - df[f'home_def_pressure_rate_L{window}']
        )
        matchup_count += 1

    if f'away_off_wr_quality_L{window}' in df.columns and f'home_def_coverage_quality_L{window}' in df.columns:
        df[f'wr_matchup_away_L{window}'] = (
            df[f'away_off_wr_quality_L{window}'] - df[f'home_def_coverage_quality_L{window}']
        )
        matchup_count += 1

print(f"✓ Added {matchup_count} matchup features")

# ============================================================================
# PREPARE FOR MODELING
# ============================================================================

print("\n[7/7] Training hybrid matchup model...")

# Select feature columns
feature_cols = [
    # Baseline (v1.2)
    'nfelo_diff', 'rest_advantage', 'div_game',
    'surface_mod', 'time_advantage', 'qb_diff'
]

# EPA features (v1.3)
for window in windows:
    feature_cols.extend([
        f'epa_margin_diff_L{window}',
        f'epa_off_diff_L{window}',
        f'epa_def_diff_L{window}'
    ])

# NGS features (v1.4)
for window in [3, 5]:
    for metric in ngs_metrics:
        feature_cols.append(f'{metric}_diff_L{window}')

# Matchup features (v2.1 - NEW!)
for window in windows:
    feature_cols.extend([
        f'pass_matchup_home_L{window}',
        f'oline_matchup_home_L{window}',
        f'wr_matchup_home_L{window}',
        f'pass_matchup_away_L{window}',
        f'oline_matchup_away_L{window}',
        f'wr_matchup_away_L{window}'
    ])

# Keep only columns that exist
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]
y = df['spread_line']

# Fill NaN
X = X.fillna(0)

# Filter complete cases
mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

# Filter to 2018+ (when team profiles available)
mask_2018 = df['season'] >= 2018
X = X[mask_2018].reset_index(drop=True)
y = y[mask_2018].reset_index(drop=True)
df = df[mask_2018].reset_index(drop=True)

print(f"\n✓ Feature set ready:")
print(f"  Total features: {len(X.columns)}")
print(f"  Total games: {len(X):,}")
print(f"  Seasons: {df['season'].min()}-{df['season'].max()}")

# Train/test split
train_mask = df['season'] < 2025
test_mask = df['season'] == 2025

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n  Train: {len(X_train):,} games (2018-2024)")
print(f"  Test:  {len(X_test):,} games (2025)")

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n{'='*80}")
print(f"MODEL PERFORMANCE")
print(f"{'='*80}")

print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")
print(f"  R²:   {train_r2:.3f}")

print(f"\nTest Set (2025):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  RMSE: {test_rmse:.2f} points")
print(f"  R²:   {test_r2:.3f}")

# Compare to v1.4
v1_4_mae = 1.42
improvement = v1_4_mae - test_mae
pct_change = (improvement / v1_4_mae) * 100

print(f"\nComparison to v1.4:")
print(f"  v1.4 MAE: {v1_4_mae:.2f}")
print(f"  v2.1 MAE: {test_mae:.2f}")
if improvement > 0:
    print(f"  Improvement: {improvement:.2f} points ({pct_change:+.1f}%)")
else:
    print(f"  Change: {improvement:.2f} points ({pct_change:+.1f}%)")

# ============================================================================
# VALUE IDENTIFICATION
# ============================================================================

print(f"\n{'='*80}")
print(f"VALUE BET ANALYSIS")
print(f"{'='*80}")

test_df = df[test_mask].copy()
test_df['our_spread'] = y_test_pred
test_df['vegas_spread'] = y_test
test_df['edge'] = test_df['our_spread'] - test_df['vegas_spread']
test_df['abs_edge'] = test_df['edge'].abs()

# Value threshold
value_threshold = 2.0
test_df['is_value'] = test_df['abs_edge'] >= value_threshold

print(f"\nEdge Distribution:")
print(f"  Mean absolute edge: {test_df['abs_edge'].mean():.2f} points")
print(f"  Median absolute edge: {test_df['abs_edge'].median():.2f} points")
print(f"  Max edge: {test_df['abs_edge'].max():.2f} points")

print(f"\nValue Bets (|edge| >= {value_threshold}):")
n_value = test_df['is_value'].sum()
print(f"  Count: {n_value} / {len(test_df)} games ({test_df['is_value'].mean()*100:.1f}%)")

if n_value > 0:
    value_bets = test_df[test_df['is_value']].copy()
    print(f"\nTop 10 Value Bets:")
    print(f"\n{'Week':<6} {'Matchup':<22} {'Vegas':<8} {'Our Line':<10} {'Edge':<8}")
    print("-" * 60)

    top_value = value_bets.nlargest(10, 'abs_edge')
    for _, row in top_value.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"{row['week']:<6} {matchup:<22} {row['vegas_spread']:>7.1f} {row['our_spread']:>9.1f} {row['edge']:>7.1f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print(f"\n{'='*80}")
print(f"TOP 20 MOST IMPORTANT FEATURES")
print(f"{'='*80}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

feature_importance['abs_coef'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<40} {'Coefficient':<12}")
print("-" * 60)
for idx, row in feature_importance.head(20).iterrows():
    rank = list(feature_importance.index).index(idx) + 1
    print(f"{rank:<6} {row['Feature']:<40} {row['Coefficient']:>11.3f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = Path('/home/user/BK_Build/output')

model_metadata = {
    'model_version': 'v2.1',
    'model_type': 'spread_prediction_with_matchups',
    'n_features': len(X.columns),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'train_mae': train_mae,
    'train_r2': train_r2,
    'train_rmse': train_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'improvement_vs_v1_4_pct': float(pct_change),
    'n_value_bets': int(n_value),
    'value_threshold': value_threshold
}

with open(output_dir / 'ball_knower_v2_1_model.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

feature_importance.to_csv(output_dir / 'ball_knower_v2_1_feature_importance.csv', index=False)

if n_value > 0:
    value_bets.to_csv(output_dir / 'ball_knower_v2_1_value_bets_2025.csv', index=False)

test_df.to_csv(output_dir / 'ball_knower_v2_1_predictions_2025.csv', index=False)

print(f"\n{'='*80}")
print(f"RESULTS SAVED")
print(f"{'='*80}")
print(f"\n✓ Model metadata: {output_dir / 'ball_knower_v2_1_model.json'}")
print(f"✓ Feature importance: {output_dir / 'ball_knower_v2_1_feature_importance.csv'}")
if n_value > 0:
    print(f"✓ Value bets: {output_dir / 'ball_knower_v2_1_value_bets_2025.csv'}")
print(f"✓ All predictions: {output_dir / 'ball_knower_v2_1_predictions_2025.csv'}")

print(f"\n{'='*80}")
print(f"SUMMARY - BALL KNOWER v2.1")
print(f"{'='*80}")

print(f"""
Model Evolution:
  v1.2: Baseline (6 features, 1.57 MAE)
  v1.3: + Rolling EPA (15 features, 1.49 MAE)
  v1.4: + Next Gen Stats (25 features, 1.42 MAE)
  v2.1: + Matchup Features ({len(X.columns)} features, {test_mae:.2f} MAE)

v2.1 Enhancements:
  • Team profile rolling windows (3/5/10 games)
  • Matchup-specific features:
    - Home offense vs away defense
    - Away offense vs home defense
    - OL quality vs pass rush
    - WR quality vs coverage
    - Passing accuracy vs coverage quality

Performance:
  • Test MAE: {test_mae:.2f} points
  • Test R²: {test_r2:.3f}
  • {pct_change:+.1f}% vs v1.4

Value Bets:
  • {n_value} games with |edge| >= {value_threshold} points
  • Ready for systematic betting analysis

Next Steps:
  1. Backtest betting performance (ROI, Sharpe ratio)
  2. Add more scheme features (play-action%, blitz%, etc.)
  3. Implement Kelly criterion position sizing
  4. Build weekly prediction pipeline
""")

print("="*80 + "\n")
