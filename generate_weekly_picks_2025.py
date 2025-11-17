"""
Ball Knower - 2025 Weekly Picks Generator

Generates weekly picks for the "Favorites 5+ Edge" strategy.

Strategy:
- Only bet when model has ≥5 point edge vs Vegas
- AND model favors the FAVORITE (not underdog)
- Validated 54.6% win rate on 306 bets (2020-2024)

Usage:
    python generate_weekly_picks_2025.py --week 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'src'))
from feature_engineering_v2 import ComprehensiveFeatureBuilder
from sklearn.ensemble import GradientBoostingRegressor
import pickle

print("="*80)
print("Ball Knower - 2025 Weekly Picks Generator")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, required=True, help='NFL week number (1-18)')
parser.add_argument('--save-model', action='store_true', help='Save trained model to file')
parser.add_argument('--load-model', type=str, help='Load model from file instead of training')
args = parser.parse_args()

WEEK = args.week
SEASON = 2025

print(f"\n📅 Generating picks for 2025 Week {WEEK}")
print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 1. LOAD DATA & TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Load Data & Train Model")
print("="*80)

builder = ComprehensiveFeatureBuilder(data_dir='.')
builder.load_all_data()

# Feature columns
feature_cols = [
    'home_off_epa_mean', 'home_off_epa_recent3',
    'away_off_epa_mean', 'away_off_epa_recent3',
    'home_def_epa_mean', 'away_def_epa_mean',
    'home_qb_rating', 'away_qb_rating',
    'home_qb_completion_pct', 'away_qb_completion_pct',
    'home_qb_out', 'away_qb_out',
    'home_players_out', 'away_players_out',
    'wind', 'temp', 'is_outdoor',
    'referee_scoring_tendency',
    'home_rest', 'away_rest',
    'div_game',
    'home_pass_vs_away_passdef',
    'away_pass_vs_home_passdef',
    'home_rush_vs_away_rushdef',
    'away_rush_vs_home_rushdef',
]

# Load or train model
if args.load_model:
    print(f"\nLoading model from {args.load_model}...")
    with open(args.load_model, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
else:
    print("\nTraining model on 2013-2024...")
    train_df = builder.build_training_dataset(
        start_season=2013,
        end_season=2024,
        min_week=4
    )

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['actual_margin'].values

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_split=20,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    model.fit(X_train, y_train)
    print("✓ Model trained")

    if args.save_model:
        model_path = Path('models') / f'ball_knower_v2_trained_{datetime.now().strftime("%Y%m%d")}.pkl'
        model_path.parent.mkdir(exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {model_path}")

# ============================================================================
# 2. LOAD CURRENT WEEK GAMES
# ============================================================================

print("\n" + "="*80)
print(f"STEP 2: Load 2025 Week {WEEK} Games")
print("="*80)

# Load schedules
schedules = pd.read_parquet('schedules.parquet')

current_week_games = schedules[
    (schedules['season'] == SEASON) &
    (schedules['week'] == WEEK) &
    (schedules['game_type'] == 'REG')
].copy()

if len(current_week_games) == 0:
    print(f"\n⚠️  No games found for 2025 Week {WEEK}")
    print("Make sure you have updated schedules.parquet with 2025 data")
    sys.exit(1)

print(f"\n✓ Found {len(current_week_games)} games for Week {WEEK}")

# ============================================================================
# 3. BUILD FEATURES FOR EACH GAME
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Build Features & Generate Predictions")
print("="*80)

predictions = []

for idx, game in current_week_games.iterrows():
    # Build features
    features = builder.build_game_features(
        season=SEASON,
        week=WEEK,
        home_team=game['home_team'],
        away_team=game['away_team']
    )

    # Convert to array for prediction
    feature_array = np.array([[features[col] if col in features else 0.0 for col in feature_cols]])

    # Predict
    predicted_margin = model.predict(feature_array)[0]

    # Get Vegas line
    vegas_spread = game.get('spread_line', None)

    if vegas_spread is not None:
        # Calculate edge
        model_edge = predicted_margin - vegas_spread
        abs_edge = abs(model_edge)

        # Determine if this is a "Favorites 5+ Edge" bet
        is_home_favorite = vegas_spread < 0
        is_away_favorite = vegas_spread > 0

        # Model favors home if edge > 0, favors away if edge < 0
        model_favors_home = model_edge > 0
        model_favors_away = model_edge < 0

        # Check if this qualifies for our strategy
        is_bet = False
        bet_team = None
        bet_line = None

        if abs_edge >= 5.0:
            # Model has strong opinion
            if is_home_favorite and model_favors_home:
                # Betting on home favorite
                is_bet = True
                bet_team = game['home_team']
                bet_line = vegas_spread
            elif is_away_favorite and model_favors_away:
                # Betting on away favorite
                is_bet = True
                bet_team = game['away_team']
                bet_line = -vegas_spread  # Flip sign for away perspective

        predictions.append({
            'game_id': game['game_id'],
            'gameday': game.get('gameday', 'Unknown'),
            'gametime': game.get('gametime', 'Unknown'),
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'vegas_spread': vegas_spread,
            'model_prediction': predicted_margin,
            'model_edge': model_edge,
            'abs_edge': abs_edge,
            'is_bet': is_bet,
            'bet_team': bet_team,
            'bet_line': bet_line,
        })

pred_df = pd.DataFrame(predictions)

print(f"\n✓ Generated predictions for {len(pred_df)} games")

# ============================================================================
# 4. FILTER TO STRATEGY BETS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Identify Strategy Bets")
print("="*80)

bets = pred_df[pred_df['is_bet'] == True].copy()
no_bets = pred_df[pred_df['is_bet'] == False].copy()

print(f"\n📊 Strategy: Favorites 5+ Edge")
print(f"   Bets identified: {len(bets)}")
print(f"   Games passing: {len(no_bets)}")

if len(bets) > 0:
    print("\n" + "="*80)
    print("📋 BETS FOR WEEK " + str(WEEK))
    print("="*80)

    for idx, bet in bets.iterrows():
        print(f"\n🎯 BET #{bets.index.get_loc(idx) + 1}")
        print(f"   Game: {bet['away_team']} @ {bet['home_team']}")
        print(f"   Date: {bet['gameday']} {bet['gametime']}")
        print(f"   ")
        print(f"   Vegas Line: {bet['bet_team']} {bet['bet_line']:+.1f}")
        print(f"   Model Prediction: {bet['bet_team']} should win by {abs(bet['model_prediction']):+.1f}")
        print(f"   Model Edge: {bet['abs_edge']:.1f} points")
        print(f"   ")
        print(f"   → BET: {bet['bet_team']} {bet['bet_line']:+.1f}")
        print(f"   Confidence: HIGH (validated 54.6% win rate)")
else:
    print("\n✅ No bets meet strategy criteria this week")
    print("   (Need: ≥5 point edge AND model favors favorite)")

# ============================================================================
# 5. SAVE PICKS TO FILE
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Save Picks")
print("="*80)

# Create picks directory
picks_dir = Path('picks_2025')
picks_dir.mkdir(exist_ok=True)

# Save full predictions
full_path = picks_dir / f'week_{WEEK}_all_predictions.csv'
pred_df.to_csv(full_path, index=False)
print(f"\n✓ All predictions saved: {full_path}")

# Save bets only
if len(bets) > 0:
    bets_path = picks_dir / f'week_{WEEK}_bets.csv'
    bets.to_csv(bets_path, index=False)
    print(f"✓ Strategy bets saved: {bets_path}")

# Create human-readable summary
summary_path = picks_dir / f'week_{WEEK}_summary.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write(f"Ball Knower - 2025 Week {WEEK} Picks\n")
    f.write("="*80 + "\n")
    f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Strategy: Favorites 5+ Edge (54.6% validated win rate)\n")
    f.write(f"\n")
    f.write(f"Total games: {len(pred_df)}\n")
    f.write(f"Strategy bets: {len(bets)}\n")
    f.write(f"\n")

    if len(bets) > 0:
        f.write("="*80 + "\n")
        f.write("PICKS FOR WEEK " + str(WEEK) + "\n")
        f.write("="*80 + "\n")

        for idx, bet in bets.iterrows():
            f.write(f"\nBET #{bets.index.get_loc(idx) + 1}\n")
            f.write(f"  Game: {bet['away_team']} @ {bet['home_team']}\n")
            f.write(f"  Date: {bet['gameday']} {bet['gametime']}\n")
            f.write(f"  \n")
            f.write(f"  BET: {bet['bet_team']} {bet['bet_line']:+.1f}\n")
            f.write(f"  Model Edge: {bet['abs_edge']:.1f} points\n")
            f.write(f"  Confidence: HIGH\n")
            f.write(f"\n")
    else:
        f.write("\nNo bets meet strategy criteria this week.\n")

print(f"✓ Summary saved: {summary_path}")

print("\n" + "="*80)
print("✅ Pick generation complete!")
print("="*80)

if len(bets) > 0:
    print(f"\n📋 {len(bets)} bet(s) identified for Week {WEEK}")
    print(f"📁 Files saved to: picks_2025/")
    print(f"\n⚠️  PAPER TRADING ONLY - DO NOT BET REAL MONEY YET")
    print(f"   Track results for Weeks 1-4 before considering real bets")
else:
    print(f"\n✅ No qualifying bets this week - Week {WEEK} is a pass")
