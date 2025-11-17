"""
Path A Analysis: High-Confidence Strategy Deep Dive

Analyze the 68 bets from 2024 where model had ≥5 point edge vs Vegas
to understand WHAT makes these high-conviction opportunities special.

Questions to answer:
1. What patterns exist in these 68 bets?
2. Are certain teams/situations over-represented?
3. Does the edge work in both directions (model favors home vs away)?
4. Is it specific to favorites, underdogs, or totals?
5. Does it validate on earlier seasons (2020-2023)?
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))
from feature_engineering_v2 import ComprehensiveFeatureBuilder

print("="*80)
print("Path A: High-Confidence Strategy Analysis")
print("="*80)

# ============================================================================
# 1. REBUILD 2024 VALIDATION SET
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: Rebuild 2024 Validation Data")
print("="*80)

builder = ComprehensiveFeatureBuilder(data_dir='.')
builder.load_all_data()

print("\nBuilding 2024 validation dataset...")
val_df = builder.build_training_dataset(
    start_season=2024,
    end_season=2024,
    min_week=4
)

print(f"✓ Loaded {len(val_df)} games from 2024")

# Load trained model
print("\nLoading trained model...")
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# We'll need to retrain quickly or load from file
# For now, let's retrain on 2013-2023
print("Training model on 2013-2023...")
train_df = builder.build_training_dataset(
    start_season=2013,
    end_season=2023,
    min_week=4
)

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

X_train = train_df[feature_cols].fillna(0).values
y_train = train_df['actual_margin'].values

X_val = val_df[feature_cols].fillna(0).values
y_val = val_df['actual_margin'].values

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

# Generate predictions
val_df['bk_v2_prediction'] = model.predict(X_val)
val_df['model_edge'] = val_df['bk_v2_prediction'] - val_df['spread_line']
val_df['abs_edge'] = val_df['model_edge'].abs()

# Calculate bet outcomes
val_df['bet_home'] = (val_df['model_edge'] > 0).astype(int)
val_df['bet_away'] = (val_df['model_edge'] < 0).astype(int)

val_df['bet_won'] = 0
val_df.loc[
    (val_df['bet_home'] == 1) & (val_df['actual_margin'] > val_df['spread_line']),
    'bet_won'
] = 1
val_df.loc[
    (val_df['bet_away'] == 1) & (val_df['actual_margin'] < val_df['spread_line']),
    'bet_won'
] = 1

# ============================================================================
# 2. ANALYZE HIGH-EDGE BETS (≥5 POINTS)
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: High-Edge Bet Analysis (≥5 Points)")
print("="*80)

high_edge = val_df[val_df['abs_edge'] >= 5.0].copy()

print(f"\nHigh-edge bets: {len(high_edge)}")
print(f"Win rate: {high_edge['bet_won'].mean():.1%}")
print(f"Wins: {high_edge['bet_won'].sum()}")
print(f"Losses: {len(high_edge) - high_edge['bet_won'].sum()}")

# ============================================================================
# 2A. DIRECTION ANALYSIS
# ============================================================================

print("\n" + "-"*80)
print("2A. Bet Direction Analysis")
print("-"*80)

home_bets = high_edge[high_edge['bet_home'] == 1]
away_bets = high_edge[high_edge['bet_away'] == 1]

print(f"\nBetting ON home team: {len(home_bets)} bets")
if len(home_bets) > 0:
    print(f"  Win rate: {home_bets['bet_won'].mean():.1%}")
    print(f"  Avg edge: {home_bets['model_edge'].mean():+.2f} pts")

print(f"\nBetting ON away team: {len(away_bets)} bets")
if len(away_bets) > 0:
    print(f"  Win rate: {away_bets['bet_won'].mean():.1%}")
    print(f"  Avg edge: {away_bets['model_edge'].mean():+.2f} pts")

# ============================================================================
# 2B. FAVORITES VS UNDERDOGS
# ============================================================================

print("\n" + "-"*80)
print("2B. Favorites vs Underdogs")
print("-"*80)

# Classify as favorite/underdog based on spread
high_edge['home_is_favorite'] = (high_edge['spread_line'] < 0).astype(int)
high_edge['betting_favorite'] = 0
high_edge['betting_underdog'] = 0

# When betting home team
high_edge.loc[
    (high_edge['bet_home'] == 1) & (high_edge['home_is_favorite'] == 1),
    'betting_favorite'
] = 1
high_edge.loc[
    (high_edge['bet_home'] == 1) & (high_edge['home_is_favorite'] == 0),
    'betting_underdog'
] = 1

# When betting away team
high_edge.loc[
    (high_edge['bet_away'] == 1) & (high_edge['home_is_favorite'] == 0),
    'betting_favorite'
] = 1
high_edge.loc[
    (high_edge['bet_away'] == 1) & (high_edge['home_is_favorite'] == 1),
    'betting_underdog'
] = 1

favorite_bets = high_edge[high_edge['betting_favorite'] == 1]
underdog_bets = high_edge[high_edge['betting_underdog'] == 1]

print(f"\nBetting ON favorites: {len(favorite_bets)} bets")
if len(favorite_bets) > 0:
    print(f"  Win rate: {favorite_bets['bet_won'].mean():.1%}")

print(f"\nBetting ON underdogs: {len(underdog_bets)} bets")
if len(underdog_bets) > 0:
    print(f"  Win rate: {underdog_bets['bet_won'].mean():.1%}")

# ============================================================================
# 2C. TEAMS INVOLVED
# ============================================================================

print("\n" + "-"*80)
print("2C. Most Frequent Teams in High-Edge Bets")
print("-"*80)

# Count team appearances (either home or away)
team_counts = pd.concat([
    high_edge['home_team'].value_counts(),
    high_edge['away_team'].value_counts()
]).groupby(level=0).sum().sort_values(ascending=False)

print("\nTop 10 teams in high-edge bets:")
print(team_counts.head(10))

# Win rate by team (for teams with ≥3 appearances)
print("\nWin rate for teams with ≥3 high-edge bets:")
for team in team_counts[team_counts >= 3].index:
    team_bets = high_edge[
        (high_edge['home_team'] == team) | (high_edge['away_team'] == team)
    ]
    print(f"  {team}: {len(team_bets)} bets, {team_bets['bet_won'].mean():.1%} win rate")

# ============================================================================
# 2D. WEEK DISTRIBUTION
# ============================================================================

print("\n" + "-"*80)
print("2D. Week Distribution")
print("-"*80)

week_dist = high_edge.groupby('week').agg({
    'game_id': 'count',
    'bet_won': 'mean'
}).round(3)
week_dist.columns = ['n_bets', 'win_rate']

print("\nHigh-edge bets by week:")
print(week_dist)

# ============================================================================
# 2E. TOTAL LINE LEVELS
# ============================================================================

print("\n" + "-"*80)
print("2E. Total Line Analysis")
print("-"*80)

# Categorize by total line
high_edge['total_category'] = pd.cut(
    high_edge['total_line'],
    bins=[0, 40, 45, 50, 100],
    labels=['Low (<40)', 'Medium (40-45)', 'High (45-50)', 'Very High (50+)']
)

total_analysis = high_edge.groupby('total_category').agg({
    'game_id': 'count',
    'bet_won': 'mean'
}).round(3)
total_analysis.columns = ['n_bets', 'win_rate']

print("\nHigh-edge bets by total line:")
print(total_analysis)

# ============================================================================
# 3. VALIDATE ON 2020-2023
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Validate Strategy on 2020-2023")
print("="*80)

print("\nBuilding 2020-2023 dataset...")
val_2020_2023 = builder.build_training_dataset(
    start_season=2020,
    end_season=2023,
    min_week=4
)

print(f"✓ Loaded {len(val_2020_2023)} games from 2020-2023")

# Train on 2013-2019, predict on 2020-2023
print("\nTraining on 2013-2019...")
train_early = builder.build_training_dataset(
    start_season=2013,
    end_season=2019,
    min_week=4
)

X_train_early = train_early[feature_cols].fillna(0).values
y_train_early = train_early['actual_margin'].values

model_early = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_samples_split=20,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model_early.fit(X_train_early, y_train_early)

# Predict on 2020-2023
X_val_2020_2023 = val_2020_2023[feature_cols].fillna(0).values
val_2020_2023['bk_v2_prediction'] = model_early.predict(X_val_2020_2023)
val_2020_2023['model_edge'] = val_2020_2023['bk_v2_prediction'] - val_2020_2023['spread_line']
val_2020_2023['abs_edge'] = val_2020_2023['model_edge'].abs()

# Calculate bet outcomes
val_2020_2023['bet_home'] = (val_2020_2023['model_edge'] > 0).astype(int)
val_2020_2023['bet_won'] = 0
val_2020_2023.loc[
    (val_2020_2023['bet_home'] == 1) & (val_2020_2023['actual_margin'] > val_2020_2023['spread_line']),
    'bet_won'
] = 1
val_2020_2023.loc[
    (val_2020_2023['bet_home'] == 0) & (val_2020_2023['actual_margin'] < val_2020_2023['spread_line']),
    'bet_won'
] = 1

# Test ≥5 pt edge strategy
high_edge_2020_2023 = val_2020_2023[val_2020_2023['abs_edge'] >= 5.0]

print(f"\nHigh-edge bets (2020-2023): {len(high_edge_2020_2023)}")
if len(high_edge_2020_2023) > 0:
    win_rate = high_edge_2020_2023['bet_won'].mean()
    wins = high_edge_2020_2023['bet_won'].sum()
    losses = len(high_edge_2020_2023) - wins
    profit = wins * 1.0 - losses * 1.1
    roi = (profit / len(high_edge_2020_2023)) * 100

    print(f"Win rate: {win_rate:.1%}")
    print(f"Record: {wins}W - {losses}L")
    print(f"ROI: {roi:+.1f}%")
    print(f"Profit: {profit:+.1f} units")

    if win_rate >= 0.524:
        print(f"\n✅ VALIDATES: 2020-2023 win rate {win_rate:.1%} ≥ 52.4%")
    else:
        print(f"\n❌ FAILS: 2020-2023 win rate {win_rate:.1%} < 52.4%")
else:
    print("⚠️  No high-edge bets in 2020-2023")

# ============================================================================
# 4. SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: Summary & Recommendations")
print("="*80)

print(f"\n📊 High-Edge Strategy Summary:")
print(f"  2024: {len(high_edge)} bets, {high_edge['bet_won'].mean():.1%} win rate")
print(f"  2020-2023: {len(high_edge_2020_2023)} bets, {high_edge_2020_2023['bet_won'].mean():.1%} win rate")

total_bets = len(high_edge) + len(high_edge_2020_2023)
total_wins = high_edge['bet_won'].sum() + high_edge_2020_2023['bet_won'].sum()
combined_win_rate = total_wins / total_bets if total_bets > 0 else 0

print(f"\n  Combined (2020-2024): {total_bets} bets, {combined_win_rate:.1%} win rate")

if combined_win_rate >= 0.524:
    print(f"\n✅ STRATEGY VALIDATED across {total_bets} bets")
    print(f"   Ready for paper trading on 2025 season")
else:
    print(f"\n⚠️  Combined win rate {combined_win_rate:.1%} below 52.4% threshold")
    print(f"   May be variance - need more data or refinement")

print("\n" + "="*80)
print("Analysis complete.")
print("="*80)
