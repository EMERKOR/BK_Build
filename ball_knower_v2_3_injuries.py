"""
BALL KNOWER v2.3 - Injury Impact Features

Extends v1.4 with injury data:
- QB injuries (most impactful)
- Top WR/TE injuries
- OL injuries (pass protection impact)
- Key defensive injuries (CB, EDGE)

Injury impact quantification:
- Position-weighted injury score
- Key player out indicators
- Cumulative injury burden

Expected impact: HIGH (injuries are orthogonal to existing features)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import json

print("\n" + "="*80)
print("BALL KNOWER v2.3 - INJURY IMPACT MODEL")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/8] Loading base data...")

# Load from nfelo
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

# Load injuries
injuries = pd.read_parquet('/home/user/BK_Build/injuries.parquet')
print(f"✓ Loaded injuries: {len(injuries):,} injury reports ({injuries['season'].min()}-{injuries['season'].max()})")

# ============================================================================
# QUANTIFY INJURY IMPACT
# ============================================================================

print("\n[2/8] Quantifying injury impact by position...")

# Position impact weights (relative importance)
position_weights = {
    'QB': 10.0,    # Most important
    'WR': 2.0,
    'TE': 1.5,
    'RB': 1.5,
    'T': 2.5,      # Tackles (OL)
    'G': 1.5,      # Guards
    'C': 2.0,      # Center
    'EDGE': 2.5,   # Edge rusher
    'DE': 2.5,
    'CB': 2.0,
    'S': 1.0,
    'LB': 1.5,
    'DT': 1.5
}

# Injury severity by status
status_severity = {
    'Out': 1.0,              # Definitely not playing
    'Doubtful': 0.8,         # Probably not playing
    'Questionable': 0.4,     # 50/50
    'Probable': 0.1,         # Probably playing
    'IR': 1.0,               # Out for season
    'PUP': 1.0,              # Out
    'Suspended': 1.0         # Out
}

# Calculate injury impact score
injuries['position_weight'] = injuries['position'].map(position_weights).fillna(0.5)
injuries['severity'] = injuries['report_status'].map(status_severity).fillna(0.0)
injuries['injury_impact'] = injuries['position_weight'] * injuries['severity']

print(f"✓ Calculated injury impact scores")
print(f"  Position weights: QB={position_weights['QB']}, WR={position_weights['WR']}, OL={position_weights['T']}")
print(f"  Status severity: Out={status_severity['Out']}, Questionable={status_severity['Questionable']}")

# ============================================================================
# AGGREGATE INJURIES TO TEAM-WEEK LEVEL
# ============================================================================

print("\n[3/8] Aggregating injuries to team-week level...")

# Total injury burden
team_injuries = injuries.groupby(['season', 'week', 'team']).agg({
    'injury_impact': 'sum',  # Total weighted injury score
    'full_name': 'count'     # Number of injured players
}).reset_index()

team_injuries = team_injuries.rename(columns={
    'injury_impact': 'total_injury_impact',
    'full_name': 'num_injuries'
})

# Key position-specific injuries
position_injuries = injuries.pivot_table(
    index=['season', 'week', 'team'],
    columns='position',
    values='injury_impact',
    aggfunc='sum',
    fill_value=0
).reset_index()

# Rename position columns
for pos in position_weights.keys():
    if pos in position_injuries.columns:
        position_injuries = position_injuries.rename(columns={pos: f'{pos.lower()}_injury'})

# Merge
team_injuries = team_injuries.merge(
    position_injuries,
    on=['season', 'week', 'team'],
    how='left'
)

# Fill missing
for col in team_injuries.columns:
    if '_injury' in col or 'injury_impact' in col:
        team_injuries[col] = team_injuries[col].fillna(0)

print(f"✓ Created team-week injury profiles: {len(team_injuries):,} team-weeks")
print(f"  Features: total impact, num injuries, {len([c for c in team_injuries.columns if '_injury' in c])} position-specific")

# ============================================================================
# CREATE ROLLING INJURY FEATURES
# ============================================================================

print("\n[4/8] Creating rolling injury features...")

team_injuries = team_injuries.sort_values(['team', 'season', 'week'])

# Rolling windows for injury trends
windows = [2, 4]  # Short windows (injuries are acute, not chronic)

for window in windows:
    team_injuries[f'injury_impact_L{window}'] = (
        team_injuries.groupby('team')['total_injury_impact']
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    # QB injuries specifically (most important)
    if 'qb_injury' in team_injuries.columns:
        team_injuries[f'qb_injury_L{window}'] = (
            team_injuries.groupby('team')['qb_injury']
            .shift(1)
            .rolling(window, min_periods=1)
            .max()  # Max (not mean) - QB injury is binary
        )

print(f"✓ Created {len(windows) * 2} rolling injury features")

# ============================================================================
# LOAD OTHER FEATURES (v1.4)
# ============================================================================

print("\n[5/8] Loading EPA and NGS features...")

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

epa_windows = [3, 5, 10]
for window in epa_windows:
    epa_data[f'epa_margin_L{window}'] = (
        epa_data.groupby('team')['off_epa_per_play']
        .shift(1).rolling(window, min_periods=1).mean() -
        epa_data.groupby('team')['def_epa_per_play']
        .shift(1).rolling(window, min_periods=1).mean()
    )
    epa_data[f'off_epa_L{window}'] = (
        epa_data.groupby('team')['off_epa_per_play']
        .shift(1).rolling(window, min_periods=1).mean()
    )
    epa_data[f'def_epa_L{window}'] = (
        epa_data.groupby('team')['def_epa_per_play']
        .shift(1).rolling(window, min_periods=1).mean()
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
                ngs_data.groupby('team')[metric]
                .shift(1).rolling(window, min_periods=1).mean()
            )

print(f"✓ Created rolling features for EPA and NGS")

# ============================================================================
# MERGE FEATURES
# ============================================================================

print("\n[6/8] Merging all features...")

# EPA differentials
for window in epa_windows:
    df = df.merge(
        epa_data[['season', 'week', 'team', f'epa_margin_L{window}',
                  f'off_epa_L{window}', f'def_epa_L{window}']],
        left_on=['season', 'week', 'home_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_drop')
    )
    df = df.rename(columns={f'epa_margin_L{window}': f'home_epa_margin_L{window}',
                             f'off_epa_L{window}': f'home_off_epa_L{window}',
                             f'def_epa_L{window}': f'home_def_epa_L{window}'})
    df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

    df = df.merge(
        epa_data[['season', 'week', 'team', f'epa_margin_L{window}',
                  f'off_epa_L{window}', f'def_epa_L{window}']],
        left_on=['season', 'week', 'away_team'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_drop')
    )
    df = df.rename(columns={f'epa_margin_L{window}': f'away_epa_margin_L{window}',
                             f'off_epa_L{window}': f'away_off_epa_L{window}',
                             f'def_epa_L{window}': f'away_def_epa_L{window}'})
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

        df = df.merge(ngs_data[['season', 'week', 'team', col]],
                      left_on=['season', 'week', 'home_team'],
                      right_on=['season', 'week', 'team'],
                      how='left', suffixes=('', '_drop'))
        df = df.rename(columns={col: f'home_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df = df.merge(ngs_data[['season', 'week', 'team', col]],
                      left_on=['season', 'week', 'away_team'],
                      right_on=['season', 'week', 'team'],
                      how='left', suffixes=('', '_drop'))
        df = df.rename(columns={col: f'away_{col}'})
        df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

        df[f'{metric}_diff_L{window}'] = df[f'home_{col}'] - df[f'away_{col}']

# INJURY DIFFERENTIALS (NEW in v2.3!)
injury_cols = ['total_injury_impact', 'injury_impact_L2', 'injury_impact_L4']
if 'qb_injury' in team_injuries.columns:
    injury_cols.extend(['qb_injury_L2', 'qb_injury_L4'])

for col in injury_cols:
    if col not in team_injuries.columns:
        continue

    df = df.merge(team_injuries[['season', 'week', 'team', col]],
                  left_on=['season', 'week', 'home_team'],
                  right_on=['season', 'week', 'team'],
                  how='left', suffixes=('', '_drop'))
    df = df.rename(columns={col: f'home_{col}'})
    df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

    df = df.merge(team_injuries[['season', 'week', 'team', col]],
                  left_on=['season', 'week', 'away_team'],
                  right_on=['season', 'week', 'team'],
                  how='left', suffixes=('', '_drop'))
    df = df.rename(columns={col: f'away_{col}'})
    df = df.drop([c for c in df.columns if 'drop' in c], axis=1)

    df[f'{col}_diff'] = df[f'home_{col}'].fillna(0) - df[f'away_{col}'].fillna(0)

print(f"✓ Merged injury differentials")

# ============================================================================
# PREPARE FOR MODELING
# ============================================================================

print("\n[7/8] Preparing for modeling...")

# Select features
feature_cols = [
    # v1.2 baseline
    'nfelo_diff', 'rest_advantage', 'div_game',
    'surface_mod', 'time_advantage', 'qb_diff'
]

# v1.3 EPA
for window in epa_windows:
    feature_cols.extend([
        f'epa_margin_diff_L{window}',
        f'epa_off_diff_L{window}',
        f'epa_def_diff_L{window}'
    ])

# v1.4 NGS
for window in [3, 5]:
    for metric in ngs_metrics:
        feature_cols.append(f'{metric}_diff_L{window}')

# v2.3 INJURIES (NEW!)
for col in injury_cols:
    feature_cols.append(f'{col}_diff')

# Keep only existing
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['vegas_line']

# Fill NaN
X = X.fillna(0)

# Filter complete cases
mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

injury_features = [c for c in X.columns if 'injury' in c]

print(f"\n✓ Feature set ready:")
print(f"  Total features: {len(X.columns)}")
print(f"    v1.2 baseline: 6")
print(f"    v1.3 EPA: 9")
print(f"    v1.4 NGS: 10")
print(f"    v2.3 Injuries: {len(injury_features)} (NEW!)")
print(f"  Total games: {len(X):,}")
print(f"  Seasons: {df['season'].min()}-{df['season'].max()}")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n[8/8] Training v2.3 model...")

train_mask = df['season'] < 2025
test_mask = df['season'] == 2025

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"  Train: {len(X_train):,} games ({df[train_mask]['season'].min()}-{df[train_mask]['season'].max()})")
print(f"  Test:  {len(X_test):,} games (2025)")

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n{'='*80}")
print(f"MODEL PERFORMANCE - v2.3 vs v1.4")
print(f"{'='*80}")

print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")
print(f"  R²:   {train_r2:.3f}")

print(f"\nTest Set (2025):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  RMSE: {test_rmse:.2f} points")
print(f"  R²:   {test_r2:.3f}")

v1_4_mae = 1.42
improvement = v1_4_mae - test_mae
pct_change = (improvement / v1_4_mae) * 100

print(f"\nComparison to v1.4:")
print(f"  v1.4 MAE: {v1_4_mae:.2f}")
print(f"  v2.3 MAE: {test_mae:.2f}")
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
    is_injury = 'injury' in row['Feature']
    marker = " ◄ INJURY" if is_injury else ""
    print(f"{rank:<6} {row['Feature']:<40} {row['Coefficient']:>11.3f}{marker}")

injury_in_top20 = sum(1 for idx, row in feature_importance.head(20).iterrows()
                      if 'injury' in row['Feature'])
print(f"\n✓ Injury features in top 20: {injury_in_top20}/20")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

model_metadata = {
    'model_version': 'v2.3',
    'features_added': 'Injury impact (position-weighted, QB-specific, rolling)',
    'n_features': len(X.columns),
    'n_injury_features': len(injury_features),
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
    'injury_features_in_top20': int(injury_in_top20)
}

with open(output_dir / 'ball_knower_v2_3_model.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

feature_importance.to_csv(output_dir / 'ball_knower_v2_3_feature_importance.csv', index=False)

print(f"\n{'='*80}")
print(f"RESULTS SAVED")
print(f"{'='*80}")
print(f"\n✓ Model metadata: {output_dir / 'ball_knower_v2_3_model.json'}")
print(f"✓ Feature importance: {output_dir / 'ball_knower_v2_3_feature_importance.csv'}")

print(f"\n{'='*80}")
print(f"SUMMARY - v2.3 INJURY IMPACT")
print(f"{'='*80}")

if improvement > 0:
    verdict = f"✓ SUCCESS - Injury features improved prediction by {pct_change:.1f}%"
else:
    verdict = f"✗ NO IMPROVEMENT - Injury features added {abs(pct_change):.1f}% noise"

print(f"""
{verdict}

v2.3 Additions:
  • Position-weighted injury scores (QB=10x, WR=2x, etc.)
  • Injury status severity (Out=1.0, Questionable=0.4, etc.)
  • Rolling injury burden (L2, L4 windows)
  • QB-specific injury indicators
  • = {len(injury_features)} new features

Performance:
  • v1.4: {v1_4_mae:.2f} MAE
  • v2.3: {test_mae:.2f} MAE
  • Change: {improvement:+.2f} points ({pct_change:+.1f}%)

Injury Feature Impact:
  • {injury_in_top20}/20 top features are injury-based
  • {'High' if injury_in_top20 >= 3 else 'Medium' if injury_in_top20 >= 1 else 'Low'} importance

Next: {'Keep injury features for final model' if improvement > 0 else 'Consider dropping injury features'}
""")

print("="*80 + "\n")
