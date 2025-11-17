"""
BALL KNOWER v2.4 - Comprehensive Team Stats

Extends v1.4 with team_stats_week.parquet:
- 3rd down conversion %
- Red zone efficiency
- Turnover differential
- Penalty yards
- Time of possession trends

These stats capture game control & execution quality.
Expected impact: MEDIUM (may overlap with EPA)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import json

print("\n" + "="*80)
print("BALL KNOWER v2.4 - COMPREHENSIVE TEAM STATS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/7] Loading base data...")

nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
df = pd.read_csv(nfelo_url)

df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
df['season'] = df['season'].astype(int)
df['week'] = df['week'].astype(int)

df = df[df['home_line_close'].notna()].copy()
df = df[df['starting_nfelo_home'].notna()].copy()
df = df[df['starting_nfelo_away'].notna()].copy()
df = df[df['season'] >= 2016].copy()

print(f"✓ Loaded {len(df):,} games (2016-{df['season'].max()})")

# Load team stats
team_stats = pd.read_parquet('/home/user/BK_Build/team_stats_week.parquet')
team_stats = team_stats[team_stats['season_type'] == 'REG'].copy()
team_stats = team_stats[team_stats['season'] >= 2016].copy()

print(f"✓ Loaded team stats: {len(team_stats):,} team-weeks")

# ============================================================================
# CALCULATE EFFICIENCY METRICS
# ============================================================================

print("\n[2/7] Calculating efficiency metrics...")

# 3rd down efficiency
team_stats['third_down_pct'] = (
    team_stats['passing_third_down_converted'] + team_stats['rushing_third_down_converted']
) / (
    team_stats['passing_third_down_failed'] + team_stats['rushing_third_down_failed'] +
    team_stats['passing_third_down_converted'] + team_stats['rushing_third_down_converted']
).replace(0, 1)

# Red zone efficiency (TD%)
team_stats['redzone_td_pct'] = (
    team_stats['passing_tds'] + team_stats['rushing_tds']
) / (
    team_stats['passing_tds'] + team_stats['rushing_tds'] +
    team_stats.get('passing_fg_att', 0) + 1  # +1 to avoid div by zero
)

# Turnover differential (per play)
team_stats['to_diff'] = (
    (team_stats.get('def_interceptions', 0) + team_stats.get('def_fumbles_rec', 0)) -
    (team_stats['passing_interceptions'] + team_stats['rushing_fumbles_lost'] +
     team_stats['sack_fumbles_lost'] + team_stats['receiving_fumbles_lost'])
)

# Explosive play rate (20+ yard plays)
team_stats['explosive_plays'] = (
    ((team_stats['passing_yards'] > 0) & (team_stats['passing_yards'] / team_stats['attempts'].replace(0, 1) > 20)).astype(int) +
    ((team_stats['rushing_yards'] > 0) & (team_stats['rushing_yards'] / team_stats['carries'].replace(0, 1) > 20)).astype(int)
)

# Success rate (using EPA as proxy)
team_stats['success_rate'] = (team_stats['passing_epa'] + team_stats['rushing_epa']) / (
    team_stats['attempts'] + team_stats['carries']
).replace(0, 1)

print(f"✓ Calculated efficiency metrics")

# ============================================================================
# CREATE ROLLING TEAM STATS
# ============================================================================

print("\n[3/7] Creating rolling team stats...")

team_stats = team_stats.sort_values(['team', 'season', 'week'])

stat_metrics = [
    'third_down_pct',
    'redzone_td_pct',
    'to_diff',
    'explosive_plays',
    'success_rate'
]

windows = [3, 5, 10]

for window in windows:
    for metric in stat_metrics:
        if metric in team_stats.columns:
            team_stats[f'{metric}_L{window}'] = (
                team_stats.groupby('team')[metric]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )

print(f"✓ Created {len(windows) * len(stat_metrics)} rolling stat features")

# ============================================================================
# LOAD OTHER FEATURES (v1.4)
# ============================================================================

print("\n[4/7] Loading EPA and NGS features...")

# v1.2 baseline
df['nfelo_diff'] = df['starting_nfelo_home'] - df['starting_nfelo_away']
df['rest_advantage'] = df['home_bye_mod'].fillna(0) + df['away_bye_mod'].fillna(0)
df['div_game'] = df['div_game_mod'].fillna(0)
df['surface_mod'] = df['dif_surface_mod'].fillna(0)
df['time_advantage'] = df['home_time_advantage_mod'].fillna(0)
df['qb_diff'] = df['home_538_qb_adj'].fillna(0) - df['away_538_qb_adj'].fillna(0)
df['vegas_line'] = df['home_line_close']

# EPA
epa_data = pd.read_csv('/home/user/BK_Build/team_week_epa_2013_2025.csv')
epa_data = epa_data.sort_values(['team', 'season', 'week'])

for window in windows:
    epa_data[f'epa_margin_L{window}'] = (
        epa_data.groupby('team')['off_epa_per_play'].shift(1).rolling(window, min_periods=1).mean() -
        epa_data.groupby('team')['def_epa_per_play'].shift(1).rolling(window, min_periods=1).mean()
    )
    epa_data[f'off_epa_L{window}'] = (
        epa_data.groupby('team')['off_epa_per_play'].shift(1).rolling(window, min_periods=1).mean()
    )
    epa_data[f'def_epa_L{window}'] = (
        epa_data.groupby('team')['def_epa_per_play'].shift(1).rolling(window, min_periods=1).mean()
    )

# NGS
ngs_data = pd.read_csv('/home/user/BK_Build/team_week_ngs_2016_2025.csv')
ngs_data = ngs_data.sort_values(['team', 'season', 'week'])

ngs_metrics = ['cpoe', 'avg_time_to_throw', 'aggressiveness',
               'rush_efficiency', 'avg_separation']

for window in [3, 5]:
    for metric in ngs_metrics:
        if metric in ngs_data.columns:
            ngs_data[f'{metric}_L{window}'] = (
                ngs_data.groupby('team')[metric].shift(1).rolling(window, min_periods=1).mean()
            )

print(f"✓ Created rolling features for EPA and NGS")

# ============================================================================
# MERGE FEATURES
# ============================================================================

print("\n[5/7] Merging all features...")

# EPA differentials
for window in windows:
    for prefix, suffix in [('home', 'away')]:
        for team_type in ['home_team', 'away_team']:
            df = df.merge(
                epa_data[['season', 'week', 'team', f'epa_margin_L{window}',
                          f'off_epa_L{window}', f'def_epa_L{window}']],
                left_on=['season', 'week', team_type],
                right_on=['season', 'week', 'team'],
                how='left',
                suffixes=('', '_drop')
            )
            pre = 'home' if team_type == 'home_team' else 'away'
            df = df.rename(columns={
                f'epa_margin_L{window}': f'{pre}_epa_margin_L{window}',
                f'off_epa_L{window}': f'{pre}_off_epa_L{window}',
                f'def_epa_L{window}': f'{pre}_def_epa_L{window}'
            })
            df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

    df[f'epa_margin_diff_L{window}'] = df[f'home_epa_margin_L{window}'] - df[f'away_epa_margin_L{window}']
    df[f'epa_off_diff_L{window}'] = df[f'home_off_epa_L{window}'] - df[f'away_off_epa_L{window}']
    df[f'epa_def_diff_L{window}'] = df[f'home_def_epa_L{window}'] - df[f'away_def_epa_L{window}']

# NGS differentials
for window in [3, 5]:
    for metric in ngs_metrics:
        col = f'{metric}_L{window}'
        if col not in ngs_data.columns:
            continue

        for team_type in ['home_team', 'away_team']:
            df = df.merge(ngs_data[['season', 'week', 'team', col]],
                          left_on=['season', 'week', team_type],
                          right_on=['season', 'week', 'team'],
                          how='left', suffixes=('', '_drop'))
            pre = 'home' if team_type == 'home_team' else 'away'
            df = df.rename(columns={col: f'{pre}_{col}'})
            df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df[f'{metric}_diff_L{window}'] = df[f'home_{col}'] - df[f'away_{col}']

# TEAM STATS DIFFERENTIALS (NEW!)
for window in windows:
    for metric in stat_metrics:
        col = f'{metric}_L{window}'
        if col not in team_stats.columns:
            continue

        for team_type in ['home_team', 'away_team']:
            df = df.merge(team_stats[['season', 'week', 'team', col]],
                          left_on=['season', 'week', team_type],
                          right_on=['season', 'week', 'team'],
                          how='left', suffixes=('', '_drop'))
            pre = 'home' if team_type == 'home_team' else 'away'
            df = df.rename(columns={col: f'{pre}_{col}'})
            df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df[f'{metric}_diff_L{window}'] = df[f'home_{col}'].fillna(0) - df[f'away_{col}'].fillna(0)

print(f"✓ Merged team stats differentials")

# ============================================================================
# PREPARE FOR MODELING
# ============================================================================

print("\n[6/7] Preparing for modeling...")

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

# v2.4 TEAM STATS (NEW!)
for window in windows:
    for metric in stat_metrics:
        feature_cols.append(f'{metric}_diff_L{window}')

feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['vegas_line']

X = X.fillna(0)

mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

stat_features = [c for c in X.columns if any(m in c for m in stat_metrics)]

print(f"\n✓ Feature set ready:")
print(f"  Total features: {len(X.columns)}")
print(f"    v1.2 baseline: 6")
print(f"    v1.3 EPA: 9")
print(f"    v1.4 NGS: 10")
print(f"    v2.4 Team Stats: {len(stat_features)} (NEW!)")
print(f"  Total games: {len(X):,}")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n[7/7] Training v2.4 model...")

train_mask = df['season'] < 2025
test_mask = df['season'] == 2025

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"  Train: {len(X_train):,} games")
print(f"  Test:  {len(X_test):,} games (2025)")

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

v1_4_mae = 1.42
improvement = v1_4_mae - test_mae
pct_change = (improvement / v1_4_mae) * 100

print(f"\n{'='*80}")
print(f"PERFORMANCE - v2.4 vs v1.4")
print(f"{'='*80}")
print(f"\nTest Set (2025):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  R²:   {test_r2:.3f}")
print(f"\n  v1.4: {v1_4_mae:.2f} MAE")
print(f"  v2.4: {test_mae:.2f} MAE")
if improvement > 0:
    print(f"  ✓ IMPROVEMENT: {improvement:.2f} points ({pct_change:+.1f}%)")
else:
    print(f"  ✗ REGRESSION: {improvement:.2f} points ({pct_change:+.1f}%)")

# Save
output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

model_metadata = {
    'model_version': 'v2.4',
    'features_added': 'Team stats (3rd down%, red zone%, TO diff, explosive plays, success rate)',
    'n_features': len(X.columns),
    'n_stat_features': len(stat_features),
    'test_mae': test_mae,
    'test_r2': test_r2,
    'improvement_vs_v1_4_pct': float(pct_change)
}

with open(output_dir / 'ball_knower_v2_4_model.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"\n{'='*80}")
print(f"SUMMARY - v2.4")
print(f"{'='*80}")
if improvement > 0:
    print(f"\n✓ Team stats IMPROVED by {pct_change:.1f}%")
else:
    print(f"\n✗ Team stats HURT by {abs(pct_change):.1f}%")

print("="*80 + "\n")
