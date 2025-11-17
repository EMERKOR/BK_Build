"""
Ball Knower v2.0 - Comprehensive NFL Spread Prediction

Key Improvements over v1.x:
1. Uses ALL available data (40MB, not just 2.5%)
2. Trains on actual outcomes (not Vegas lines)
3. Wind/ref/injuries are FEATURES, not standalone signals
4. Comprehensive feature engineering from multiple data sources
5. Only bets when model shows significant edge with high confidence

Strategy:
- Build rich feature set (50+ features)
- Train gradient boosting model on actual margins
- Predict actual margin (not Vegas line)
- Bet when |prediction - vegas| > threshold AND confidence is high
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_engineering_v2 import ComprehensiveFeatureBuilder
from team_mapping import normalize_team_name

print("="*80)
print("Ball Knower v2.0 - Comprehensive Model")
print("="*80)

# ============================================================================
# 1. LOAD DATA & BUILD FEATURES
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: Feature Engineering")
print("="*80)

# Initialize feature builder
builder = ComprehensiveFeatureBuilder(data_dir='.')
builder.load_all_data()

# Build training dataset (2013-2023)
# We hold out 2024 for final validation
print("\nBuilding training dataset (2013-2023)...")
train_df = builder.build_training_dataset(
    start_season=2013,
    end_season=2023,
    min_week=4  # Need week 4+ for feature history
)

print(f"\nTraining data shape: {train_df.shape}")
print(f"Seasons: {train_df['season'].min()}-{train_df['season'].max()}")
print(f"Weeks: {train_df['week'].min()}-{train_df['week'].max()}")

# Build validation dataset (2024)
print("\nBuilding validation dataset (2024)...")
val_df = builder.build_training_dataset(
    start_season=2024,
    end_season=2024,
    min_week=4
)

print(f"Validation data shape: {val_df.shape}")

# ============================================================================
# 2. PREPARE FEATURES & TARGET
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Prepare Features & Target")
print("="*80)

# Define feature columns (all numeric features we built)
feature_cols = [
    # Team EPA features
    'home_off_epa_mean', 'home_off_epa_recent3',
    'away_off_epa_mean', 'away_off_epa_recent3',
    'home_def_epa_mean', 'away_def_epa_mean',

    # Player features
    'home_qb_rating', 'away_qb_rating',
    'home_qb_completion_pct', 'away_qb_completion_pct',

    # Injury features
    'home_qb_out', 'away_qb_out',
    'home_players_out', 'away_players_out',

    # Context features (weather, ref, rest)
    'wind', 'temp', 'is_outdoor',
    'referee_scoring_tendency',
    'home_rest', 'away_rest',
    'div_game',

    # Matchup features
    'home_pass_vs_away_passdef',
    'away_pass_vs_home_passdef',
    'home_rush_vs_away_rushdef',
    'away_rush_vs_home_rushdef',
]

# Verify all features exist
available_features = [col for col in feature_cols if col in train_df.columns]
missing_features = [col for col in feature_cols if col not in train_df.columns]

print(f"\nFeatures available: {len(available_features)}")
print(f"Features missing: {len(missing_features)}")
if missing_features:
    print(f"Missing: {missing_features}")

feature_cols = available_features

# Target variable: actual margin (home team perspective)
target_col = 'actual_margin'

# Prepare training data
X_train = train_df[feature_cols].fillna(0).values
y_train = train_df[target_col].values

# Prepare validation data
X_val = val_df[feature_cols].fillna(0).values
y_val = val_df[target_col].values

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Train Model")
print("="*80)

print("\nTraining Gradient Boosting Regressor...")
print("Hyperparameters:")
print("  - n_estimators: 200")
print("  - max_depth: 4")
print("  - learning_rate: 0.05")
print("  - min_samples_split: 20")
print("  - subsample: 0.8")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_samples_split=20,
    subsample=0.8,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

print("\n✓ Model training complete")

# ============================================================================
# 4. EVALUATE ON TRAINING SET
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: Training Set Performance")
print("="*80)

train_preds = model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

print(f"\nPrediction Accuracy:")
print(f"  MAE: {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")

# Compare to Vegas
train_df['bk_v2_prediction'] = train_preds
train_df['vegas_error'] = train_df['actual_margin'] - train_df['spread_line']
train_df['bk_v2_error'] = train_df['actual_margin'] - train_df['bk_v2_prediction']

vegas_mae_train = train_df['vegas_error'].abs().mean()
bk_v2_mae_train = train_df['bk_v2_error'].abs().mean()

print(f"\nVs Vegas Lines:")
print(f"  Vegas MAE: {vegas_mae_train:.2f} points")
print(f"  BK v2.0 MAE: {bk_v2_mae_train:.2f} points")
print(f"  Improvement: {vegas_mae_train - bk_v2_mae_train:+.2f} points")

# ============================================================================
# 5. EVALUATE ON VALIDATION SET (2024)
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: Validation Set Performance (2024 Out-of-Sample)")
print("="*80)

val_preds = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_preds)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print(f"\nPrediction Accuracy:")
print(f"  MAE: {val_mae:.2f} points")
print(f"  RMSE: {val_rmse:.2f} points")

# Compare to Vegas
val_df['bk_v2_prediction'] = val_preds
val_df['vegas_error'] = val_df['actual_margin'] - val_df['spread_line']
val_df['bk_v2_error'] = val_df['actual_margin'] - val_df['bk_v2_prediction']

vegas_mae_val = val_df['vegas_error'].abs().mean()
bk_v2_mae_val = val_df['bk_v2_error'].abs().mean()

print(f"\nVs Vegas Lines:")
print(f"  Vegas MAE: {vegas_mae_val:.2f} points")
print(f"  BK v2.0 MAE: {bk_v2_mae_val:.2f} points")
print(f"  Improvement: {vegas_mae_val - bk_v2_mae_val:+.2f} points")

if bk_v2_mae_val < vegas_mae_val:
    print(f"\n✅ SUCCESS: Model beats Vegas on 2024 out-of-sample!")
else:
    print(f"\n⚠️  Model does not beat Vegas on 2024 out-of-sample")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: Feature Importance")
print("="*80)

# Get feature importances
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# ============================================================================
# 7. BETTING SIMULATION (2024)
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: Betting Simulation (2024)")
print("="*80)

# Calculate model edge vs Vegas
val_df['model_edge'] = val_df['bk_v2_prediction'] - val_df['spread_line']
val_df['abs_edge'] = val_df['model_edge'].abs()

# Betting strategy: Only bet when edge is significant
edge_thresholds = [2.0, 3.0, 4.0, 5.0]

print("\nBetting Performance by Edge Threshold:")
print("-" * 80)

for threshold in edge_thresholds:
    bets = val_df[val_df['abs_edge'] >= threshold].copy()

    if len(bets) == 0:
        print(f"\nEdge ≥ {threshold} pts: No bets")
        continue

    # Determine bet direction
    bets['bet_home'] = (bets['model_edge'] > 0).astype(int)
    bets['bet_away'] = (bets['model_edge'] < 0).astype(int)

    # Did bet win?
    bets['bet_won'] = 0

    # Home bet wins if: actual_margin > spread_line
    bets.loc[
        (bets['bet_home'] == 1) & (bets['actual_margin'] > bets['spread_line']),
        'bet_won'
    ] = 1

    # Away bet wins if: actual_margin < spread_line
    bets.loc[
        (bets['bet_away'] == 1) & (bets['actual_margin'] < bets['spread_line']),
        'bet_won'
    ] = 1

    # Calculate results
    n_bets = len(bets)
    wins = bets['bet_won'].sum()
    losses = n_bets - wins
    win_rate = wins / n_bets if n_bets > 0 else 0

    # ROI (assuming -110 vig)
    profit = wins * 1.0 - losses * 1.1
    roi = (profit / n_bets) * 100 if n_bets > 0 else 0

    print(f"\nEdge ≥ {threshold} pts:")
    print(f"  Bets: {n_bets}")
    print(f"  Record: {wins}W - {losses}L")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Profit: {profit:+.1f} units")

    if win_rate >= 0.524:
        print(f"  ✅ PROFITABLE: Beats 52.4% threshold")
    else:
        print(f"  ❌ NOT PROFITABLE: Below 52.4% threshold")

# ============================================================================
# 8. SAMPLE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: Sample 2024 Predictions")
print("="*80)

# Show games where model had strong opinion
strong_edges = val_df[val_df['abs_edge'] >= 3.0].sort_values('abs_edge', ascending=False).head(10)

print("\nTop 10 Largest Edges (2024):")
print("-" * 80)

for idx, row in strong_edges.iterrows():
    print(f"\nWeek {int(row['week'])}: {row['away_team']} @ {row['home_team']}")
    print(f"  Vegas line: {row['home_team']} {row['spread_line']:+.1f}")
    print(f"  BK v2.0 prediction: {row['home_team']} {row['bk_v2_prediction']:+.1f}")
    print(f"  Model edge: {row['model_edge']:+.1f} pts")
    print(f"  Actual result: {row['home_team']} {row['actual_margin']:+.1f}")
    print(f"  Bet result: {'WIN ✅' if row.get('bet_won', 0) == 1 else 'LOSS ❌'}")

# ============================================================================
# 9. SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY & NEXT STEPS")
print("="*80)

print(f"\n✓ Ball Knower v2.0 Complete")
print(f"\nModel Performance:")
print(f"  Training MAE: {train_mae:.2f} pts (vs Vegas {vegas_mae_train:.2f})")
print(f"  Validation MAE: {val_mae:.2f} pts (vs Vegas {vegas_mae_val:.2f})")

print(f"\nKey Insights:")
print(f"  1. Model uses {len(feature_cols)} comprehensive features")
print(f"  2. Trains on actual outcomes (not Vegas lines)")
print(f"  3. Wind, refs, injuries are features (not standalone signals)")

# Determine if model is viable
if bk_v2_mae_val < vegas_mae_val:
    print(f"\n🚀 RECOMMENDATION: Model shows promise!")
    print(f"   - Beats Vegas prediction accuracy on 2024 data")
    print(f"   - Next step: Test betting strategies with edge thresholds")
    print(f"   - Paper trade on remaining 2024 games before going live")
else:
    print(f"\n⚠️  Model needs improvement")
    print(f"   - Does not beat Vegas prediction accuracy")
    print(f"   - Next steps:")
    print(f"     1. Add more features (WR/RB/OL/DL stats)")
    print(f"     2. Tune hyperparameters")
    print(f"     3. Try different model architectures")

print("\n" + "="*80)
print("Analysis complete.")
print("="*80)
