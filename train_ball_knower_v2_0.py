"""
Ball Knower v2.0 - Advanced Model Training & Backtesting
=========================================================

This script builds and validates Ball Knower v2.0, which incorporates:
- QB rolling performance (QBR, EPA, CPOE)
- Team rolling EPA (offense/defense)
- Next Gen Stats (time to throw, aggressiveness, air yards)
- Momentum and recent form indicators

Training Strategy:
- Train on 2020-2023 seasons (4 years)
- Test on full 2024 season (held out)
- Use Ridge regression (handles feature correlations)
- Cross-validation for regularization strength
- Compare to v1.2 baseline and Vegas

Author: Ball Knower Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import advanced_features, config

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n" + "="*80)
print("BALL KNOWER v2.0 - MODEL TRAINING & BACKTESTING")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_SEASONS = [2020, 2021, 2022, 2023]
TEST_SEASON = 2024

# Feature engineering windows
QB_LOOKBACK = 3  # Last 3 games for QB stats
TEAM_LOOKBACK = 5  # Last 5 games for team stats

# Regularization alphas to test
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Minimum week for features (need prior games)
MIN_WEEK = max(QB_LOOKBACK, TEAM_LOOKBACK) + 1

print(f"\nConfiguration:")
print(f"  Training seasons: {TRAIN_SEASONS}")
print(f"  Test season: {TEST_SEASON}")
print(f"  QB lookback: {QB_LOOKBACK} games")
print(f"  Team lookback: {TEAM_LOOKBACK} games")
print(f"  Min week (for features): {MIN_WEEK}")

# ============================================================================
# STEP 1: LOAD HISTORICAL GAMES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING HISTORICAL GAMES")
print("="*80)

def load_historical_games(seasons):
    """Load all games with spreads from specified seasons."""
    all_games = []

    for season in seasons:
        print(f"\n  Loading {season} season...")
        season_games = nflverse.games(season=season)

        # Filter for regular season games with spreads
        season_games = season_games[
            (season_games['spread_line'].notna()) &
            (season_games['game_type'] == 'REG')
        ].copy()

        # Add actual margin
        season_games['actual_margin'] = season_games['home_score'] - season_games['away_score']

        # Convert spread to home perspective (nflverse uses away perspective)
        season_games['home_spread'] = -1 * season_games['spread_line']

        print(f"    ✓ Loaded {len(season_games)} games")
        all_games.append(season_games)

    combined = pd.concat(all_games, ignore_index=True)
    print(f"\n✓ Total games loaded: {len(combined)}")

    return combined

# Load training data
print("\nLoading training data...")
train_games = load_historical_games(TRAIN_SEASONS)

# Load test data
print("\nLoading test data...")
test_games = load_historical_games([TEST_SEASON])

# Filter for games with actual results (scores)
train_games = train_games[train_games['actual_margin'].notna()].copy()
test_games = test_games[test_games['actual_margin'].notna()].copy()

# Filter out early weeks (need prior games for rolling stats)
train_games = train_games[train_games['week'] >= MIN_WEEK].copy()
test_games = test_games[test_games['week'] >= MIN_WEEK].copy()

print(f"\nAfter filtering:")
print(f"  Training games: {len(train_games)} (weeks {train_games['week'].min()}-{train_games['week'].max()})")
print(f"  Test games: {len(test_games)} (weeks {test_games['week'].min()}-{test_games['week'].max()})")

# ============================================================================
# STEP 2: GENERATE ADVANCED FEATURES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: GENERATING ADVANCED FEATURES")
print("="*80)

print("\nThis may take a few minutes...")

# Load QB and EPA data once
print("\nLoading feature data sources...")
qbr_data, ngs_data, injuries = advanced_features.load_qb_data()
epa_data = advanced_features.load_team_epa()

print(f"✓ QBR data: {len(qbr_data)} rows")
print(f"✓ NGS data: {len(ngs_data)} rows")
print(f"✓ EPA data: {len(epa_data)} rows")

def generate_features_for_games(games_df, season_list, qbr, ngs, epa):
    """Generate advanced features for a set of games."""
    all_features = []

    for idx, game in games_df.iterrows():
        if idx % 100 == 0:
            print(f"    Processing game {idx+1}/{len(games_df)}...")

        features = advanced_features.calculate_matchup_features(
            home_team=game['home_team'],
            away_team=game['away_team'],
            season=game['season'],
            week=game['week'],
            qbr_data=qbr,
            ngs_data=ngs,
            epa_data=epa,
            lookback_qb=QB_LOOKBACK,
            lookback_team=TEAM_LOOKBACK
        )

        # Add nfelo diff (v1.2 baseline feature)
        # Note: We don't have historical nfelo snapshots, so we'll skip this for now
        # In production, would need to reconstruct historical nfelo ratings
        features['nfelo_diff'] = 0.0  # Placeholder

        all_features.append(features)

    return pd.DataFrame(all_features)

# Generate training features
print("\nGenerating training features...")
train_features = generate_features_for_games(
    train_games.reset_index(drop=True),
    TRAIN_SEASONS,
    qbr_data,
    ngs_data,
    epa_data
)

# Generate test features
print("\nGenerating test features...")
test_features = generate_features_for_games(
    test_games.reset_index(drop=True),
    [TEST_SEASON],
    qbr_data,
    ngs_data,
    epa_data
)

print(f"\n✓ Training features shape: {train_features.shape}")
print(f"✓ Test features shape: {test_features.shape}")

# Combine with target variable
X_train = train_features
y_train = train_games.reset_index(drop=True)['home_spread']

X_test = test_features
y_test = test_games.reset_index(drop=True)['home_spread']

print(f"\n✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# ============================================================================
# STEP 3: FEATURE SELECTION & PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("STEP 3: FEATURE SELECTION & PREPROCESSING")
print("="*80)

# Check for NaN values
print("\nChecking for missing values...")
train_nans = X_train.isna().sum()
test_nans = X_test.isna().sum()

if train_nans.sum() > 0:
    print(f"\n⚠️  Training data has {train_nans.sum()} NaN values:")
    print(train_nans[train_nans > 0])
    print("  Filling with 0...")
    X_train = X_train.fillna(0)

if test_nans.sum() > 0:
    print(f"\n⚠️  Test data has {test_nans.sum()} NaN values:")
    print(test_nans[test_nans > 0])
    print("  Filling with 0...")
    X_test = X_test.fillna(0)

# Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Scaled {X_train.shape[1]} features")

# ============================================================================
# STEP 4: TRAIN MODEL WITH CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TRAINING RIDGE REGRESSION MODEL")
print("="*80)

print(f"\nTesting regularization strengths: {ALPHAS}")

# Use cross-validation to select best alpha
ridge_cv = RidgeCV(alphas=ALPHAS, cv=5, scoring='neg_mean_absolute_error')
ridge_cv.fit(X_train_scaled, y_train)

best_alpha = ridge_cv.alpha_
print(f"\n✓ Best alpha (via CV): {best_alpha}")

# Train final model with best alpha
model = Ridge(alpha=best_alpha)
model.fit(X_train_scaled, y_train)

print(f"✓ Model trained on {len(X_train)} samples")

# ============================================================================
# STEP 5: EVALUATE ON TRAINING SET
# ============================================================================

print("\n" + "="*80)
print("STEP 5: TRAINING SET PERFORMANCE")
print("="*80)

y_train_pred = model.predict(X_train_scaled)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTraining metrics (predicting Vegas lines):")
print(f"  MAE:  {train_mae:.2f} points")
print(f"  RMSE: {train_rmse:.2f} points")
print(f"  R²:   {train_r2:.3f}")

# ============================================================================
# STEP 6: BACKTEST ON 2024 SEASON
# ============================================================================

print("\n" + "="*80)
print("STEP 6: BACKTEST ON 2024 SEASON")
print("="*80)

y_test_pred = model.predict(X_test_scaled)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTest metrics (predicting Vegas lines):")
print(f"  MAE:  {test_mae:.2f} points")
print(f"  RMSE: {test_rmse:.2f} points")
print(f"  R²:   {test_r2:.3f}")

# Add predictions to test dataframe
test_results = test_games.reset_index(drop=True).copy()
test_results['v2_predicted_spread'] = y_test_pred
test_results['v2_edge'] = y_test_pred - test_results['home_spread']
test_results['v2_prediction_error'] = y_test_pred - test_results['actual_margin']

# ============================================================================
# STEP 7: COMPARE TO VEGAS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: COMPARISON TO VEGAS")
print("="*80)

# Vegas error (how well Vegas predicted actual outcomes)
vegas_error = test_results['home_spread'] - test_results['actual_margin']
vegas_mae = vegas_error.abs().mean()
vegas_rmse = np.sqrt((vegas_error ** 2).mean())

print(f"\nVegas performance (predicting actual outcomes):")
print(f"  MAE:  {vegas_mae:.2f} points")
print(f"  RMSE: {vegas_rmse:.2f} points")

# v2.0 error (how well v2.0 predicted actual outcomes)
v2_actual_error = test_results['v2_predicted_spread'] - test_results['actual_margin']
v2_actual_mae = v2_actual_error.abs().mean()
v2_actual_rmse = np.sqrt((v2_actual_error ** 2).mean())

print(f"\nv2.0 performance (predicting actual outcomes):")
print(f"  MAE:  {v2_actual_mae:.2f} points")
print(f"  RMSE: {v2_actual_rmse:.2f} points")

print(f"\nComparison:")
print(f"  v2.0 vs Vegas MAE: {v2_actual_mae - vegas_mae:+.2f} points")

if v2_actual_mae < vegas_mae:
    print(f"  ✅ v2.0 BEAT Vegas by {vegas_mae - v2_actual_mae:.2f} points!")
else:
    print(f"  🔴 v2.0 WORSE than Vegas by {v2_actual_mae - vegas_mae:.2f} points")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("STEP 8: FEATURE IMPORTANCE")
print("="*80)

# Get feature coefficients
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 15 most important features:")
print(feature_importance.head(15)[['feature', 'coefficient']].to_string(index=False))

# ============================================================================
# STEP 9: BETTING SIMULATION
# ============================================================================

print("\n" + "="*80)
print("STEP 9: BETTING SIMULATION (2024 SEASON)")
print("="*80)

# Simulate betting on games where model has 2+ point edge
edge_threshold = 2.0

test_results['abs_edge'] = test_results['v2_edge'].abs()
value_bets = test_results[test_results['abs_edge'] >= edge_threshold].copy()

print(f"\nBets with {edge_threshold}+ point edge: {len(value_bets)}")

if len(value_bets) > 0:
    # Determine bet side
    value_bets['bet_side'] = value_bets['v2_edge'].apply(lambda x: 'AWAY' if x > 0 else 'HOME')

    # Evaluate bet outcomes
    def evaluate_bet(row):
        """Determine if bet won against the spread."""
        if row['bet_side'] == 'AWAY':
            # Betting away to cover (actual margin < spread)
            if row['actual_margin'] < row['home_spread']:
                return 'WIN'
            elif row['actual_margin'] == row['home_spread']:
                return 'PUSH'
            else:
                return 'LOSS'
        else:  # HOME
            # Betting home to cover (actual margin > spread)
            if row['actual_margin'] > row['home_spread']:
                return 'WIN'
            elif row['actual_margin'] == row['home_spread']:
                return 'PUSH'
            else:
                return 'LOSS'

    value_bets['outcome'] = value_bets.apply(evaluate_bet, axis=1)

    # Calculate results
    wins = len(value_bets[value_bets['outcome'] == 'WIN'])
    losses = len(value_bets[value_bets['outcome'] == 'LOSS'])
    pushes = len(value_bets[value_bets['outcome'] == 'PUSH'])

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    # ROI calculation (standard -110 odds)
    profit = (wins * 100) - (losses * 110)
    total_risked = len(value_bets) * 110
    roi = (profit / total_risked * 100) if total_risked > 0 else 0

    print(f"\nBetting results:")
    print(f"  Wins:    {wins}")
    print(f"  Losses:  {losses}")
    print(f"  Pushes:  {pushes}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Breakeven: 52.4%")

    print(f"\nFinancial results (at -110 odds):")
    print(f"  Total profit: ${profit:+.2f}")
    print(f"  Total risked: ${total_risked:.2f}")
    print(f"  ROI: {roi:+.1f}%")

    if roi > 0:
        print(f"\n  ✅ PROFITABLE - v2.0 generated positive ROI")
    else:
        print(f"\n  🔴 UNPROFITABLE - v2.0 lost money")

else:
    print(f"\nNo bets recommended at {edge_threshold}+ threshold")

# ============================================================================
# STEP 10: SAVE MODEL & RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 10: SAVING MODEL & RESULTS")
print("="*80)

# Save model parameters
model_params = {
    'model_version': '2.0',
    'train_seasons': TRAIN_SEASONS,
    'test_season': TEST_SEASON,
    'qb_lookback': QB_LOOKBACK,
    'team_lookback': TEAM_LOOKBACK,
    'best_alpha': float(best_alpha),
    'n_features': len(X_train.columns),
    'feature_names': list(X_train.columns),
    'intercept': float(model.intercept_),
    'coefficients': {feat: float(coef) for feat, coef in zip(X_train.columns, model.coef_)},
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'train_mae': float(train_mae),
    'train_r2': float(train_r2),
    'test_mae': float(test_mae),
    'test_r2': float(test_r2),
    'vegas_mae': float(vegas_mae),
    'v2_actual_mae': float(v2_actual_mae),
    'beats_vegas': bool(v2_actual_mae < vegas_mae)
}

model_file = config.OUTPUT_DIR / 'ball_knower_v2_0_model.json'
with open(model_file, 'w') as f:
    json.dump(model_params, f, indent=2)

print(f"\n✓ Model saved to: {model_file}")

# Save test results
results_file = config.OUTPUT_DIR / 'v2_0_backtest_2024.csv'
test_results.to_csv(results_file, index=False)
print(f"✓ Test results saved to: {results_file}")

# Save feature importance
importance_file = config.OUTPUT_DIR / 'v2_0_feature_importance.csv'
feature_importance.to_csv(importance_file, index=False)
print(f"✓ Feature importance saved to: {importance_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BALL KNOWER v2.0 - TRAINING COMPLETE")
print("="*80)

beats_vegas_str = '✅ BEATS VEGAS' if v2_actual_mae < vegas_mae else '🔴 WORSE THAN VEGAS'
bets_placed = len(value_bets) if len(value_bets) > 0 else 0
win_rate_str = f"{win_rate:.1f}%" if len(value_bets) > 0 else 'N/A'
roi_str = f"{roi:+.1f}%" if len(value_bets) > 0 else 'N/A'
profitable_str = '✅ PROFITABLE' if len(value_bets) > 0 and roi > 0 else ('🔴 UNPROFITABLE' if len(value_bets) > 0 else 'NO BETS')

print(f"""
Training Performance:
  MAE (vs Vegas lines): {train_mae:.2f} points
  R²: {train_r2:.3f}
  Samples: {len(X_train)}

Test Performance (2024 Season):
  MAE (vs Vegas lines): {test_mae:.2f} points
  MAE (vs actual outcomes): {v2_actual_mae:.2f} points
  R²: {test_r2:.3f}
  Samples: {len(X_test)}

Comparison:
  Vegas MAE: {vegas_mae:.2f} points
  v2.0 vs Vegas: {v2_actual_mae - vegas_mae:+.2f} points
  {beats_vegas_str}

Betting (2+ point edge):
  Bets placed: {bets_placed}
  Win rate: {win_rate_str}
  ROI: {roi_str}
  {profitable_str}

Next Steps:
  1. Review feature importance in {importance_file}
  2. Review test results in {results_file}
  3. Analyze which features are actually helping
  4. Consider feature selection to remove noise
  5. Test on Week 12 2025 for forward validation
""")

print("="*80 + "\n")
