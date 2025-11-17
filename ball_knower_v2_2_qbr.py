"""
BALL KNOWER v2.2 - Adding ESPN QBR Features

Extends v1.4 with ESPN QBR weekly data:
- QBR total (comprehensive QB rating)
- Points added (QB contribution to scoring)
- EPA from QB plays
- Pass/Run/Sack components

Expected impact: HIGH (QB is most important position)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import json

print("\n" + "="*80)
print("BALL KNOWER v2.2 - ESPN QBR ENHANCED MODEL")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/7] Loading data from nfelo repository...")

# Load base dataset from nfelo
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

# Filter to 2016+ (when NGS data starts)
df = df[df['season'] >= 2016].copy()

print(f"✓ Loaded {len(df):,} games (2016-{df['season'].max()})")

# Load ESPN QBR data
qbr_week = pd.read_parquet('/home/user/BK_Build/espn_qbr_week.parquet')
qbr_week = qbr_week[qbr_week['season_type'] == 'Regular'].copy()

# Drop duplicate 'team' column, keep 'team_abb'
if 'team' in qbr_week.columns:
    qbr_week = qbr_week.drop('team', axis=1)

print(f"✓ Loaded ESPN QBR: {len(qbr_week):,} QB-weeks (2006-{qbr_week['season'].max()})")

# ============================================================================
# AGGREGATE QBR TO TEAM-WEEK LEVEL
# ============================================================================

print("\n[2/7] Aggregating QBR to team-week level...")

# Group by team-week and take primary QB (highest qb_plays)
qbr_team = qbr_week.sort_values('qb_plays', ascending=False).groupby(
    ['season', 'week_num', 'team_abb']
).first().reset_index()

qbr_team = qbr_team.rename(columns={'week_num': 'week', 'team_abb': 'team'})

# Select key QBR features
qbr_features = ['qbr_total', 'pts_added', 'epa_total', 'pass', 'run', 'sack']
qbr_team = qbr_team[['season', 'week', 'team'] + qbr_features]

print(f"✓ Aggregated QBR to {len(qbr_team):,} team-weeks")
print(f"  Features: {', '.join(qbr_features)}")

# ============================================================================
# CREATE ROLLING QBR FEATURES
# ============================================================================

print("\n[3/7] Creating rolling QBR features...")

qbr_team = qbr_team.sort_values(['team', 'season', 'week'])

windows = [3, 5, 10]
for window in windows:
    for feature in qbr_features:
        qbr_team[f'{feature}_L{window}'] = (
            qbr_team.groupby('team')[feature]
            .shift(1)  # Exclude current game
            .rolling(window, min_periods=1)
            .mean()
        )

print(f"✓ Created {len(windows) * len(qbr_features)} rolling QBR features")

# ============================================================================
# LOAD OTHER FEATURES (v1.4)
# ============================================================================

print("\n[4/7] Loading EPA and NGS features...")

# Load EPA
epa_data = pd.read_csv('/home/user/BK_Build/team_week_epa_2013_2025.csv')
epa_data = epa_data.sort_values(['team', 'season', 'week'])

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

# Load NGS
ngs_data = pd.read_csv('/home/user/BK_Build/team_week_ngs_2016_2025.csv')
ngs_data = ngs_data.sort_values(['team', 'season', 'week'])

ngs_metrics = ['cpoe', 'avg_time_to_throw', 'aggressiveness',
               'rush_efficiency', 'avg_separation']

for window in [3, 5]:
    for metric in ngs_metrics:
        if metric in ngs_data.columns:
            ngs_data[f'{metric}_L{window}'] = (
                ngs_data.groupby('team')[metric]
                .shift(1).rolling(window, min_periods=1).mean()
            )

print(f"✓ Loaded and created rolling features for EPA and NGS")

# ============================================================================
# BUILD FEATURES
# ============================================================================

print("\n[5/7] Building comprehensive feature set...")

# v1.2 baseline features
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
df['rest_advantage'] = df['home_bye_mod'].fillna(0) + df['away_bye_mod'].fillna(0)
df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
df['qb_diff'] = df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0)

# Target
df['vegas_line'] = df['home_line_close']

# Add EPA differentials
for window in windows:
    df = df.merge(
        epa_data[['season', 'week', 'team', f'epa_margin_L{window}',
                  f'off_epa_L{window}', f'def_epa_L{window}']],
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_drop')
    )
    df = df.rename(columns={
        f'epa_margin_L{window}': f'home_epa_margin_L{window}',
        f'off_epa_L{window}': f'home_off_epa_L{window}',
        f'def_epa_L{window}': f'home_def_epa_L{window}'
    })
    df = df.drop([col for col in df.columns if 'drop' in col], axis=1)

    df = df.merge(
        epa_data[['season', 'week', 'team', f'epa_margin_L{window}',
                  f'off_epa_L{window}', f'def_epa_L{window}']],
        left_on=['season', 'week', 'away_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_drop')
    )
    df = df.rename(columns={
        f'epa_margin_L{window}': f'away_epa_margin_L{window}',
        f'off_epa_L{window}': f'away_off_epa_L{window}',
        f'def_epa_L{window}': f'away_def_epa_L{window}'
    })
    df = df.drop([col for col in df.columns if 'drop' in col], axis=1)

    df[f'epa_margin_diff_L{window}'] = (
        df[f'home_epa_margin_L{window}'] - df[f'away_epa_margin_L{window}']
    )
    df[f'epa_off_diff_L{window}'] = (
        df[f'home_off_epa_L{window}'] - df[f'away_off_epa_L{window}']
    )
    df[f'epa_def_diff_L{window}'] = (
        df[f'home_def_epa_L{window}'] - df[f'away_def_epa_L{window}']
    )

# Add NGS differentials
for window in [3, 5]:
    for metric in ngs_metrics:
        col = f'{metric}_L{window}'
        if col not in ngs_data.columns:
            continue

        df = df.merge(
            ngs_data[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'home_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df = df.merge(
            ngs_data[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'away_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df[f'{metric}_diff_L{window}'] = df[f'home_{col}'] - df[f'away_{col}']

# Add QBR differentials (NEW in v2.2!)
for window in windows:
    for feature in qbr_features:
        col = f'{feature}_L{window}'

        df = df.merge(
            qbr_team[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'home_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df = df.merge(
            qbr_team[['season', 'week', 'team', col]],
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_drop')
        )
        df = df.rename(columns={col: f'away_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df[f'{feature}_diff_L{window}'] = df[f'home_{col}'] - df[f'away_{col}']

print(f"✓ Added QBR differentials")

# ============================================================================
# PREPARE FOR MODELING
# ============================================================================

print("\n[6/7] Preparing for modeling...")

# Select features
feature_cols = [
    # v1.2 baseline
    'nfelo_diff', 'rest_advantage', 'div_game',
    'surface_mod', 'time_advantage', 'qb_diff'
]

# v1.3 EPA
for window in windows:
    feature_cols.extend([
        f'epa_margin_diff_L{window}',
        f'epa_off_diff_L{window}',
        f'epa_def_diff_L{window}'
    ])

# v1.4 NGS
for window in [3, 5]:
    for metric in ngs_metrics:
        feature_cols.append(f'{metric}_diff_L{window}')

# v2.2 QBR (NEW!)
for window in windows:
    for feature in qbr_features:
        feature_cols.append(f'{feature}_diff_L{window}')

# Keep only existing columns
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols]
y = df['vegas_line']

# Fill NaN
X = X.fillna(0)

# Filter complete cases
mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

print(f"\n✓ Feature set ready:")
print(f"  Total features: {len(X.columns)}")
print(f"    v1.2 baseline: 6")
print(f"    v1.3 EPA: 9")
print(f"    v1.4 NGS: 10")
print(f"    v2.2 QBR: {len(windows) * len(qbr_features)} (NEW!)")
print(f"  Total games: {len(X):,}")
print(f"  Seasons: {df['season'].min()}-{df['season'].max()}")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n[7/7] Training v2.2 model...")

# Train/test split
train_mask = df['season'] < 2025
test_mask = df['season'] == 2025

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"  Train: {len(X_train):,} games ({df[train_mask]['season'].min()}-{df[train_mask]['season'].max()})")
print(f"  Test:  {len(X_test):,} games (2025)")

# Train
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predict
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
print(f"MODEL PERFORMANCE - v2.2 vs v1.4")
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
print(f"  v2.2 MAE: {test_mae:.2f}")
if improvement > 0:
    print(f"  ✓ IMPROVEMENT: {improvement:.2f} points ({pct_change:+.1f}%)")
else:
    print(f"  ✗ REGRESSION: {improvement:.2f} points ({pct_change:+.1f}%)")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print(f"\n{'='*80}")
print(f"TOP 20 FEATURES")
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
    is_qbr = any(qbr_feat in row['Feature'] for qbr_feat in ['qbr_total', 'pts_added', 'epa_total', 'pass', 'run', 'sack'])
    marker = " ◄ QBR" if is_qbr else ""
    print(f"{rank:<6} {row['Feature']:<40} {row['Coefficient']:>11.3f}{marker}")

# Count QBR features in top 20
qbr_in_top20 = sum(1 for idx, row in feature_importance.head(20).iterrows()
                   if any(qbr_feat in row['Feature'] for qbr_feat in qbr_features))
print(f"\n✓ QBR features in top 20: {qbr_in_top20}/20")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

model_metadata = {
    'model_version': 'v2.2',
    'features_added': 'ESPN QBR (qbr_total, pts_added, epa, pass, run, sack)',
    'n_features': len(X.columns),
    'n_qbr_features': len(windows) * len(qbr_features),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'train_mae': train_mae,
    'train_r2': train_r2,
    'train_rmse': train_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'improvement_vs_v1_4': float(improvement),
    'improvement_vs_v1_4_pct': float(pct_change),
    'qbr_features_in_top20': int(qbr_in_top20)
}

with open(output_dir / 'ball_knower_v2_2_model.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

feature_importance.to_csv(output_dir / 'ball_knower_v2_2_feature_importance.csv', index=False)

print(f"\n{'='*80}")
print(f"RESULTS SAVED")
print(f"{'='*80}")
print(f"\n✓ Model metadata: {output_dir / 'ball_knower_v2_2_model.json'}")
print(f"✓ Feature importance: {output_dir / 'ball_knower_v2_2_feature_importance.csv'}")

print(f"\n{'='*80}")
print(f"SUMMARY - v2.2 QBR ENHANCED")
print(f"{'='*80}")

if improvement > 0:
    verdict = f"✓ SUCCESS - QBR features improved prediction by {pct_change:.1f}%"
else:
    verdict = f"✗ NO IMPROVEMENT - QBR features added {abs(pct_change):.1f}% noise"

print(f"""
{verdict}

v2.2 Additions:
  • ESPN QBR Total (comprehensive QB rating)
  • Points Added (QB contribution to scoring)
  • EPA from QB plays
  • Pass/Run/Sack components
  • {len(windows)} rolling windows (L3, L5, L10)
  • = {len(windows) * len(qbr_features)} new features

Performance:
  • v1.4: {v1_4_mae:.2f} MAE
  • v2.2: {test_mae:.2f} MAE
  • Change: {improvement:+.2f} points ({pct_change:+.1f}%)

QBR Feature Impact:
  • {qbr_in_top20}/20 top features are QBR-based
  • {'High' if qbr_in_top20 >= 5 else 'Medium' if qbr_in_top20 >= 2 else 'Low'} importance

Next: {'Keep QBR features for v2.3+' if improvement > 0 else 'Consider dropping QBR features'}
""")

print("="*80 + "\n")
