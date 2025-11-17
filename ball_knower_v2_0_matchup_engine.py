"""
BALL KNOWER v2.0 - MATCHUP ENGINE & POINT DIFFERENTIAL MODEL

This is a fundamental shift from v1.x models:
- v1.x: Predicted Vegas spreads directly
- v2.0: Predicts ACTUAL point differentials based on matchup analysis
        Then compares to Vegas to find value

Key Features:
1. Rolling team profiles (scheme tendencies, execution quality)
2. Matchup-specific features (offense vs defense)
3. Point differential prediction (our own lines)
4. Value identification (our line vs Vegas line)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path
import json

print("\n" + "="*80)
print("BALL KNOWER v2.0 - MATCHUP ENGINE")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/8] Loading data sources...")

# Load team profiles
team_profiles = pd.read_csv('/home/user/BK_Build/team_profiles_advanced_2018_2025.csv')
print(f"✓ Team profiles: {len(team_profiles):,} team-weeks (2018-2025)")

# Load schedules (for game outcomes and Vegas lines)
schedules = pd.read_parquet('/home/user/BK_Build/schedules.parquet')
schedules = schedules[schedules['game_type'] == 'REG'].copy()
print(f"✓ Schedules: {len(schedules):,} games")

# Load EPA data
epa_data = pd.read_csv('/home/user/BK_Build/team_week_epa_2013_2025.csv')
print(f"✓ EPA data: {len(epa_data):,} team-weeks")

# Load NGS data
ngs_data = pd.read_csv('/home/user/BK_Build/team_week_ngs_2016_2025.csv')
print(f"✓ NGS data: {len(ngs_data):,} team-weeks")

# ============================================================================
# CREATE ROLLING TEAM PROFILES
# ============================================================================

print("\n[2/8] Creating rolling team profiles (3/5/10 game windows)...")

# Metrics to roll
profile_metrics = [
    'def_completion_pct', 'def_yards_allowed_per_tgt', 'def_pressures',
    'def_sacks', 'def_missed_tackle_pct', 'times_pressured_pct',
    'passing_bad_throw_pct', 'passing_drop_pct',
    'def_coverage_quality', 'def_pressure_rate', 'def_tackle_efficiency',
    'off_oline_quality', 'off_passing_accuracy', 'off_wr_quality'
]

# Sort by team and time
team_profiles = team_profiles.sort_values(['team', 'season', 'week'])

# Create rolling features
windows = [3, 5, 10]

for window in windows:
    for metric in profile_metrics:
        if metric in team_profiles.columns:
            team_profiles[f'{metric}_L{window}'] = (
                team_profiles.groupby('team')[metric]
                .shift(1)  # Exclude current game
                .rolling(window, min_periods=1)
                .mean()
            )

print(f"✓ Created {len(windows) * len(profile_metrics)} rolling profile features")

# ============================================================================
# CREATE ROLLING EPA FEATURES
# ============================================================================

print("\n[3/8] Adding rolling EPA features...")

epa_data = epa_data.sort_values(['team', 'season', 'week'])

for window in windows:
    epa_data[f'off_epa_L{window}'] = (
        epa_data.groupby('team')['off_epa_per_play']
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

    epa_data[f'def_epa_L{window}'] = (
        epa_data.groupby('team')['def_epa_per_play']
        .shift(1)
        .rolling(window, min_periods=1)
        .mean()
    )

print(f"✓ Created {len(windows) * 2} rolling EPA features")

# ============================================================================
# CREATE ROLLING NGS FEATURES
# ============================================================================

print("\n[4/8] Adding rolling NGS features...")

ngs_data = ngs_data.sort_values(['team', 'season', 'week'])

ngs_metrics = ['cpoe', 'avg_time_to_throw', 'aggressiveness',
               'rush_efficiency', 'avg_separation']

for window in [3, 5]:  # NGS only for 3,5 (more volatile)
    for metric in ngs_metrics:
        if metric in ngs_data.columns:
            ngs_data[f'{metric}_L{window}'] = (
                ngs_data.groupby('team')[metric]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
            )

print(f"✓ Created {2 * len(ngs_metrics)} rolling NGS features")

# ============================================================================
# MERGE ALL FEATURES TO GAME LEVEL
# ============================================================================

print("\n[5/8] Building matchup-specific features...")

# Filter to games with spreads
schedules = schedules[schedules['spread_line'].notna()].copy()

# Create point differential (TARGET)
schedules['point_differential'] = schedules['home_score'] - schedules['away_score']

# Initialize feature DataFrame
features_list = []

for idx, game in schedules.iterrows():
    season = game['season']
    week = game['week']
    home_team = game['home_team']
    away_team = game['away_team']

    # Get home team profiles
    home_profiles = team_profiles[
        (team_profiles['season'] == season) &
        (team_profiles['week'] == week) &
        (team_profiles['team'] == home_team)
    ]

    # Get away team profiles
    away_profiles = team_profiles[
        (team_profiles['season'] == season) &
        (team_profiles['week'] == week) &
        (team_profiles['team'] == away_team)
    ]

    # Get EPA
    home_epa = epa_data[
        (epa_data['season'] == season) &
        (epa_data['week'] == week) &
        (epa_data['team'] == home_team)
    ]

    away_epa = epa_data[
        (epa_data['season'] == season) &
        (epa_data['week'] == week) &
        (epa_data['team'] == away_team)
    ]

    # Get NGS
    home_ngs = ngs_data[
        (ngs_data['season'] == season) &
        (ngs_data['week'] == week) &
        (ngs_data['team'] == home_team)
    ]

    away_ngs = ngs_data[
        (ngs_data['season'] == season) &
        (ngs_data['week'] == week) &
        (ngs_data['team'] == away_team)
    ]

    # Skip if missing data
    if len(home_profiles) == 0 or len(away_profiles) == 0:
        continue

    # Build feature dict
    game_features = {
        'season': season,
        'week': week,
        'home_team': home_team,
        'away_team': away_team,
        'point_differential': game['point_differential'],
        'spread_line': game['spread_line'],
        'total_line': game.get('total_line', np.nan)
    }

    # Add home field advantage (constant)
    game_features['home_field'] = 1

    # MATCHUP FEATURES: Home Offense vs Away Defense
    for window in windows:
        # Home passing attack vs Away pass coverage
        if f'off_passing_accuracy_L{window}' in home_profiles.columns:
            home_pass_acc = home_profiles[f'off_passing_accuracy_L{window}'].values[0] if len(home_profiles) > 0 else 0
            away_coverage = away_profiles[f'def_coverage_quality_L{window}'].values[0] if len(away_profiles) > 0 else 0
            game_features[f'pass_matchup_home_L{window}'] = home_pass_acc - away_coverage

        # Home OL vs Away pass rush
        if f'off_oline_quality_L{window}' in home_profiles.columns:
            home_ol = home_profiles[f'off_oline_quality_L{window}'].values[0] if len(home_profiles) > 0 else 0
            away_rush = away_profiles[f'def_pressure_rate_L{window}'].values[0] if len(away_profiles) > 0 else 0
            game_features[f'oline_matchup_home_L{window}'] = home_ol - away_rush

        # Home WR vs Away coverage
        if f'off_wr_quality_L{window}' in home_profiles.columns:
            home_wr = home_profiles[f'off_wr_quality_L{window}'].values[0] if len(home_profiles) > 0 else 0
            away_cov = away_profiles[f'def_coverage_quality_L{window}'].values[0] if len(away_profiles) > 0 else 0
            game_features[f'wr_matchup_home_L{window}'] = home_wr - away_cov

    # MATCHUP FEATURES: Away Offense vs Home Defense
    for window in windows:
        # Away passing attack vs Home pass coverage
        if f'off_passing_accuracy_L{window}' in away_profiles.columns:
            away_pass_acc = away_profiles[f'off_passing_accuracy_L{window}'].values[0] if len(away_profiles) > 0 else 0
            home_coverage = home_profiles[f'def_coverage_quality_L{window}'].values[0] if len(home_profiles) > 0 else 0
            game_features[f'pass_matchup_away_L{window}'] = away_pass_acc - home_coverage

        # Away OL vs Home pass rush
        if f'off_oline_quality_L{window}' in away_profiles.columns:
            away_ol = away_profiles[f'off_oline_quality_L{window}'].values[0] if len(away_profiles) > 0 else 0
            home_rush = home_profiles[f'def_pressure_rate_L{window}'].values[0] if len(home_profiles) > 0 else 0
            game_features[f'oline_matchup_away_L{window}'] = away_ol - home_rush

        # Away WR vs Home coverage
        if f'off_wr_quality_L{window}' in away_profiles.columns:
            away_wr = away_profiles[f'off_wr_quality_L{window}'].values[0] if len(away_profiles) > 0 else 0
            home_cov = home_profiles[f'def_coverage_quality_L{window}'].values[0] if len(home_profiles) > 0 else 0
            game_features[f'wr_matchup_away_L{window}'] = away_wr - home_cov

    # EPA matchup differentials
    for window in windows:
        if len(home_epa) > 0 and len(away_epa) > 0:
            home_off_epa = home_epa[f'off_epa_L{window}'].values[0] if f'off_epa_L{window}' in home_epa.columns else 0
            away_def_epa = away_epa[f'def_epa_L{window}'].values[0] if f'def_epa_L{window}' in away_epa.columns else 0
            away_off_epa = away_epa[f'off_epa_L{window}'].values[0] if f'off_epa_L{window}' in away_epa.columns else 0
            home_def_epa = home_epa[f'def_epa_L{window}'].values[0] if f'def_epa_L{window}' in home_epa.columns else 0

            game_features[f'epa_matchup_diff_L{window}'] = (home_off_epa - away_def_epa) - (away_off_epa - home_def_epa)

    # NGS matchup differentials
    for window in [3, 5]:
        if len(home_ngs) > 0 and len(away_ngs) > 0:
            for metric in ngs_metrics:
                col = f'{metric}_L{window}'
                if col in home_ngs.columns and col in away_ngs.columns:
                    home_val = home_ngs[col].values[0]
                    away_val = away_ngs[col].values[0]
                    game_features[f'{metric}_diff_L{window}'] = home_val - away_val

    features_list.append(game_features)

# Create DataFrame
df = pd.DataFrame(features_list)

print(f"✓ Built {len(df):,} games with matchup features")
print(f"  Total features: {len(df.columns) - 7}")  # -7 for metadata columns

# ============================================================================
# PREPARE FOR MODELING
# ============================================================================

print("\n[6/8] Preparing features for modeling...")

# Drop metadata columns
X_columns = [col for col in df.columns if col not in [
    'season', 'week', 'home_team', 'away_team',
    'point_differential', 'spread_line', 'total_line'
]]

X = df[X_columns]
y = df['point_differential']

# Fill NaN with 0 (missing data means no advantage)
X = X.fillna(0)

# Filter complete cases
mask = X.notna().all(axis=1) & y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

print(f"✓ Final dataset: {len(X):,} games with {len(X.columns)} features")
print(f"  Training period: {df['season'].min()}-{df['season'].max()}")

# ============================================================================
# TRAIN-TEST SPLIT (2025 = test)
# ============================================================================

print("\n[7/8] Training point differential model...")

train_mask = df['season'] < 2025
test_mask = df['season'] == 2025

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"  Train: {len(X_train):,} games (2018-2024)")
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
print(f"POINT DIFFERENTIAL MODEL PERFORMANCE")
print(f"{'='*80}")
print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")
print(f"  R²:   {train_r2:.3f}")

print(f"\nTest Set (2025):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  RMSE: {test_rmse:.2f} points")
print(f"  R²:   {test_r2:.3f}")

# ============================================================================
# VALUE IDENTIFICATION: OUR LINE vs VEGAS LINE
# ============================================================================

print("\n[8/8] Identifying value bets...")

# Add predictions to test set
test_df = df[test_mask].copy()
test_df['our_line'] = y_test_pred
test_df['vegas_line'] = test_df['spread_line']
test_df['actual_differential'] = test_df['point_differential']

# Calculate edges
test_df['edge'] = test_df['our_line'] - test_df['vegas_line']
test_df['abs_edge'] = test_df['edge'].abs()

# Identify value bets (|edge| >= 2.0 points)
value_threshold = 2.0
test_df['is_value'] = test_df['abs_edge'] >= value_threshold

print(f"\n{'='*80}")
print(f"VALUE BET ANALYSIS (2025 Season)")
print(f"{'='*80}")

print(f"\nEdge Distribution:")
print(f"  Mean absolute edge: {test_df['abs_edge'].mean():.2f} points")
print(f"  Median absolute edge: {test_df['abs_edge'].median():.2f} points")
print(f"  Max edge: {test_df['abs_edge'].max():.2f} points")

print(f"\nValue Bets (|edge| >= {value_threshold}):")
print(f"  Count: {test_df['is_value'].sum()} / {len(test_df)} games ({test_df['is_value'].mean()*100:.1f}%)")

if test_df['is_value'].sum() > 0:
    value_bets = test_df[test_df['is_value']].copy()

    # Calculate if our edge was correct
    value_bets['edge_correct'] = (
        ((value_bets['edge'] > 0) & (value_bets['actual_differential'] > value_bets['vegas_line'])) |
        ((value_bets['edge'] < 0) & (value_bets['actual_differential'] < value_bets['vegas_line']))
    )

    accuracy = value_bets['edge_correct'].mean()

    print(f"  Accuracy: {accuracy*100:.1f}% (when we disagreed with Vegas)")
    print(f"\nTop 5 Value Bets:")
    print(f"\n{'Week':<6} {'Matchup':<20} {'Vegas':<8} {'Our Line':<10} {'Edge':<8} {'Actual':<8}")
    print("-" * 70)

    top_value = value_bets.nlargest(5, 'abs_edge')
    for _, row in top_value.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"{row['week']:<6} {matchup:<20} {row['vegas_line']:>7.1f} {row['our_line']:>9.1f} {row['edge']:>7.1f} {row['actual_differential']:>7.1f}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print(f"\n{'='*80}")
print(f"TOP 15 MOST IMPORTANT FEATURES")
print(f"{'='*80}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

feature_importance['abs_coef'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<40} {'Coefficient':<12}")
print("-" * 60)
for idx, row in feature_importance.head(15).iterrows():
    print(f"{feature_importance.index.get_loc(idx)+1:<6} {row['Feature']:<40} {row['Coefficient']:>11.3f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = Path('/home/user/BK_Build/output')
output_dir.mkdir(exist_ok=True)

# Save model metadata
model_metadata = {
    'model_version': 'v2.0',
    'model_type': 'point_differential',
    'n_features': len(X.columns),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'train_mae': train_mae,
    'train_r2': train_r2,
    'train_rmse': train_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'value_threshold': value_threshold,
    'n_value_bets': int(test_df['is_value'].sum()),
    'value_bet_accuracy': float(accuracy) if test_df['is_value'].sum() > 0 else 0
}

with open(output_dir / 'ball_knower_v2_0_model.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

# Save feature importance
feature_importance.to_csv(output_dir / 'ball_knower_v2_0_feature_importance.csv', index=False)

# Save value bets
if test_df['is_value'].sum() > 0:
    value_bets.to_csv(output_dir / 'ball_knower_v2_0_value_bets_2025.csv', index=False)

# Save all test predictions
test_df.to_csv(output_dir / 'ball_knower_v2_0_predictions_2025.csv', index=False)

print(f"\n{'='*80}")
print(f"RESULTS SAVED")
print(f"{'='*80}")
print(f"\n✓ Model metadata: {output_dir / 'ball_knower_v2_0_model.json'}")
print(f"✓ Feature importance: {output_dir / 'ball_knower_v2_0_feature_importance.csv'}")
print(f"✓ Value bets: {output_dir / 'ball_knower_v2_0_value_bets_2025.csv'}")
print(f"✓ All predictions: {output_dir / 'ball_knower_v2_0_predictions_2025.csv'}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")

print(f"""
Ball Knower v2.0 represents a fundamental shift:

Previous Approach (v1.x):
  • Predicted Vegas spreads directly
  • Tried to replicate market consensus
  • Limited value identification

New Approach (v2.0):
  • Predicts ACTUAL point differentials based on matchups
  • Generates OUR OWN lines
  • Compares to Vegas to find value
  • Matchup-specific features (offense vs defense)

Performance:
  • Test MAE: {test_mae:.2f} points (predicting actual scores)
  • Test R²: {test_r2:.3f}
  • {test_df['is_value'].sum()} value bets identified in 2025
  • {accuracy*100:.1f}% accuracy when disagreeing with Vegas

Next Steps:
  1. Backtest value betting strategy (ROI analysis)
  2. Add more matchup features (coverage schemes, personnel)
  3. Implement Kelly criterion bet sizing
  4. Weekly predictions for remaining 2025 games

Model is ready for value bet identification!
""")

print("="*80 + "\n")
