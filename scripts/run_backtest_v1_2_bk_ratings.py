#!/usr/bin/env python
"""
Ball Knower v1.2+BK Backtest Sandbox

Demonstrates how BK ratings would be integrated into v1.2 dataset for modeling.

NOTE: This is a sandbox/demo script. Full historical backtesting requires
BK ratings for historical weeks, which are not yet available.

For now, this script:
1. Shows how v1.2 and v1.2+BK datasets are constructed
2. Demonstrates the feature structure for both models
3. Documents the intended modeling workflow

Future work:
- Generate historical BK ratings (retroactive)
- Run full train/test split with both feature sets
- Compare model performance metrics
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np

print("="*70)
print("Ball Knower v1.2+BK Ratings Backtest Sandbox")
print("="*70)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# For demo purposes, we'll show the structure using Week 12 2025
# (BK ratings are available for this week)
DEMO_SEASON = 2025
DEMO_WEEK = 12

# For actual backtesting, we'd use historical ranges:
# BACKTEST_START_YEAR = 2009
# BACKTEST_END_YEAR = 2024

print("Configuration:")
print(f"  Demo Season: {DEMO_SEASON}")
print(f"  Demo Week: {DEMO_WEEK}")
print()

# ============================================================================
# LOAD DATASETS
# ============================================================================

print("Loading datasets...")
print()

# Standard v1.2 dataset (baseline)
print("1. Standard v1.2 Dataset (Baseline)")
print("   Features: nfelo_diff, rest_advantage, div_game, surface_mod, time_advantage, qb_diff")

from ball_knower.datasets import v1_2

try:
    # For demo, load Week 12 2025 if available
    df_base = v1_2.build_training_frame(
        start_year=DEMO_SEASON,
        end_year=DEMO_SEASON
    )
    # Filter to demo week
    df_base_week = df_base[df_base['week'] == DEMO_WEEK].copy()

    if len(df_base_week) > 0:
        print(f"   ✓ Loaded {len(df_base_week)} games for Week {DEMO_WEEK} {DEMO_SEASON}")
    else:
        print(f"   ⚠ No games found for Week {DEMO_WEEK} {DEMO_SEASON} in nfelo data")
        df_base_week = None
except Exception as e:
    print(f"   ✗ Error loading v1.2 data: {e}")
    df_base_week = None

print()

# v1.2+BK dataset (enhanced)
print("2. v1.2+BK Dataset (Enhanced with BK Ratings)")
print("   Additional Features: bk_rating_home, bk_rating_away, bk_rating_diff")

from ball_knower.datasets import v1_2_bk

try:
    df_bk = v1_2_bk.build_training_frame(
        season=DEMO_SEASON,
        week=DEMO_WEEK
    )
    print(f"   ✓ Loaded {len(df_bk)} games with BK ratings for Week {DEMO_WEEK} {DEMO_SEASON}")
except Exception as e:
    print(f"   ✗ Error loading v1.2+BK data: {e}")
    df_bk = None

print()
print("="*70)
print()

# ============================================================================
# FEATURE STRUCTURE COMPARISON
# ============================================================================

if df_base_week is not None and df_bk is not None:
    print("Feature Structure Comparison:")
    print()

    # v1.2 baseline features
    v1_2_features = [
        'nfelo_diff', 'rest_advantage', 'div_game',
        'surface_mod', 'time_advantage', 'qb_diff'
    ]

    # v1.2+BK enhanced features
    bk_features = v1_2_features + ['bk_rating_diff']

    print(f"Model A (v1.2 baseline): {len(v1_2_features)} features")
    for i, feat in enumerate(v1_2_features, 1):
        print(f"  {i}. {feat}")
    print()

    print(f"Model B (v1.2+BK): {len(bk_features)} features")
    for i, feat in enumerate(bk_features, 1):
        marker = " (NEW)" if feat not in v1_2_features else ""
        print(f"  {i}. {feat}{marker}")
    print()

    # Sample data inspection
    print("="*70)
    print()
    print("Sample Game with BK Ratings:")
    print()

    if len(df_bk) > 0:
        sample = df_bk.iloc[0]
        print(f"  Game: {sample['away_team']} @ {sample['home_team']}")
        print(f"  Season: {sample['season']}, Week: {sample['week']}")
        print()
        print("  Features:")
        print(f"    nfelo_diff: {sample['nfelo_diff']:.2f}")
        print(f"    bk_rating_home: {sample['bk_rating_home']:.2f}")
        print(f"    bk_rating_away: {sample['bk_rating_away']:.2f}")
        print(f"    bk_rating_diff: {sample['bk_rating_diff']:.2f}")
        print()
        if 'vegas_closing_spread' in sample and not pd.isna(sample['vegas_closing_spread']):
            print(f"  Target (Vegas Line): {sample['vegas_closing_spread']:.1f}")
        print()

# ============================================================================
# MODELING WORKFLOW (DOCUMENTATION)
# ============================================================================

print("="*70)
print()
print("Modeling Workflow (Future Implementation):")
print()
print("1. Data Preparation")
print("   - Load historical v1.2 data (2009-2024)")
print("   - Generate historical BK ratings (retroactive)")
print("   - Merge BK ratings into historical dataset")
print()
print("2. Train/Test Split")
print("   - Train: 2009-2022")
print("   - Validation: 2023")
print("   - Test: 2024")
print()
print("3. Model Training")
print("   - Model A: v1.2 baseline (6 features)")
print("   - Model B: v1.2+BK (7 features)")
print("   - Algorithm: GradientBoosting or similar")
print()
print("4. Evaluation Metrics")
print("   - MAE (Mean Absolute Error vs Vegas line)")
print("   - RMSE (Root Mean Squared Error)")
print("   - ATS Win Rate (Against The Spread)")
print("   - Mean Edge (Expected value per bet)")
print()
print("5. Comparison")
print("   - Does BK rating improve prediction accuracy?")
print("   - Does it identify +EV betting opportunities?")
print("   - What's the incremental value over nfelo alone?")
print()

print("="*70)
print()
print("Status: Sandbox/Demo Complete")
print()
print("Next Steps:")
print("  1. Generate historical BK ratings (2009-2024)")
print("  2. Implement full train/test pipeline")
print("  3. Run comparative backtest")
print("  4. Analyze feature importance and edge detection")
print()
