"""
Test Advanced Features Module
==============================

Validates the advanced feature engineering module by:
1. Testing on a few Week 11 2025 games
2. Showing feature distributions
3. Identifying data gaps

Author: Ball Knower Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.nflverse_data import nflverse
from src import advanced_features

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n" + "="*80)
print("TESTING ADVANCED FEATURES MODULE")
print("="*80)

# ============================================================================
# 1. LOAD TEST DATA
# ============================================================================

print("\n[1/4] Loading Week 11 2025 games...")

games = nflverse.games(season=2025, week=11)
games = games[games['spread_line'].notna()][['home_team', 'away_team', 'spread_line']].copy()

print(f"✓ Loaded {len(games)} games")
print(f"\n{games.to_string(index=False)}")

# ============================================================================
# 2. TEST QB FEATURES
# ============================================================================

print("\n[2/4] Testing QB features on sample games...")

# Load QB data
qbr_data, ngs_data, injuries = advanced_features.load_qb_data()

print(f"\n✓ Loaded data:")
print(f"  QBR weekly: {len(qbr_data)} rows")
print(f"  NGS passing: {len(ngs_data)} rows")
print(f"  Injuries: {len(injuries)} rows")

# Test on a specific game: TB @ BUF
test_game = games[games['home_team'] == 'BUF'].iloc[0]
home_team = test_game['home_team']
away_team = test_game['away_team']

print(f"\n\nTest game: {away_team} @ {home_team}")

# Get starting QBs
home_starter = advanced_features.get_starting_qb(
    home_team, 2024, 10, qbr_data, ngs_data
)
away_starter = advanced_features.get_starting_qb(
    away_team, 2024, 10, qbr_data, ngs_data
)

print(f"  Home starter (Week 10 2024): {home_starter}")
print(f"  Away starter (Week 10 2024): {away_starter}")

# Get rolling QB stats
home_qb_stats = advanced_features.get_qb_rolling_stats(
    home_team, 2024, 11, qbr_data, ngs_data, lookback=3
)
away_qb_stats = advanced_features.get_qb_rolling_stats(
    away_team, 2024, 11, qbr_data, ngs_data, lookback=3
)

print(f"\n  Home QB rolling stats (last 3 games):")
for key, val in home_qb_stats.items():
    print(f"    {key}: {val:.3f}")

print(f"\n  Away QB rolling stats (last 3 games):")
for key, val in away_qb_stats.items():
    print(f"    {key}: {val:.3f}")

# Test QB change detection
home_changed, home_penalty = advanced_features.detect_qb_change(
    home_team, 2024, 11, qbr_data, ngs_data
)
away_changed, away_penalty = advanced_features.detect_qb_change(
    away_team, 2024, 11, qbr_data, ngs_data
)

print(f"\n  QB change detection:")
print(f"    Home QB changed: {home_changed} (impact: {home_penalty:.2f} points)")
print(f"    Away QB changed: {away_changed} (impact: {away_penalty:.2f} points)")

# ============================================================================
# 3. TEST TEAM EPA FEATURES
# ============================================================================

print("\n[3/4] Testing team EPA features...")

epa_data = advanced_features.load_team_epa()
print(f"\n✓ Loaded team EPA data: {len(epa_data)} rows")

# Test on same game
home_epa = advanced_features.get_team_rolling_epa(
    home_team, 2024, 11, epa_data, lookback=5
)
away_epa = advanced_features.get_team_rolling_epa(
    away_team, 2024, 11, epa_data, lookback=5
)

print(f"\n  Home team EPA (last 5 games):")
for key, val in home_epa.items():
    print(f"    {key}: {val:.3f}")

print(f"\n  Away team EPA (last 5 games):")
for key, val in away_epa.items():
    print(f"    {key}: {val:.3f}")

# ============================================================================
# 4. TEST FULL MATCHUP FEATURES
# ============================================================================

print("\n[4/4] Testing full matchup feature calculation...")

matchup_features = advanced_features.calculate_matchup_features(
    home_team=home_team,
    away_team=away_team,
    season=2024,
    week=11,
    qbr_data=qbr_data,
    ngs_data=ngs_data,
    epa_data=epa_data,
    lookback_qb=3,
    lookback_team=5
)

print(f"\n✓ Calculated {len(matchup_features)} features for {away_team} @ {home_team}")

# Show key features
key_features = [
    'qb_rolling_epa_diff',
    'qb_rolling_qbr_diff',
    'rolling_epa_diff',
    'rolling_def_epa_diff',
    'momentum_diff',
    'qb_change_diff',
    'cpoe_diff'
]

print(f"\nKey differential features (home - away):")
for feat in key_features:
    if feat in matchup_features:
        print(f"  {feat}: {matchup_features[feat]:.3f}")

# ============================================================================
# 5. GENERATE FEATURES FOR ALL WEEK 11 GAMES
# ============================================================================

print("\n[5/4] Generating features for all Week 11 games...")

try:
    # Note: Using 2024 data since we don't have complete 2025 EPA/QBR data
    games_with_features = advanced_features.add_advanced_features_to_games(
        games_df=games.copy(),
        season=2024,
        week=11,
        qb_lookback=3,
        team_lookback=5
    )

    print(f"\n✓ Successfully generated features for {len(games_with_features)} games")

    # Show summary statistics
    print(f"\n\nFeature Summary Statistics:")

    feature_cols = [col for col in games_with_features.columns if col not in ['home_team', 'away_team', 'spread_line']]

    summary = games_with_features[feature_cols].describe()
    print(summary.T.round(3).to_string())

    # Save for inspection
    games_with_features.to_csv('output/week_11_advanced_features_test.csv', index=False)
    print(f"\n✓ Saved features to: output/week_11_advanced_features_test.csv")

except Exception as e:
    print(f"\n🔴 ERROR generating features: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("FEATURE TESTING COMPLETE")
print("="*80)

print("""
Next steps:
1. Review feature distributions in output/week_11_advanced_features_test.csv
2. Build Ball Knower v2.0 model incorporating these features
3. Backtest on 2024 season
4. Compare to v1.2 performance
""")

print("\n" + "="*80 + "\n")
