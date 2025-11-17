"""
Ball Knower - Phase 1b: Injury Impact Analysis
===============================================

Focus: QB injuries (most impactful position)

Questions:
1. When a starting QB is out, does Vegas accurately adjust?
2. Do backup QBs beat/miss the spread at different rates?
3. Is there systematic over/under-reaction to QB news?

Method:
- Identify games where starting QB was injured/out
- Compare Vegas performance in QB-out games vs normal games
- Look for systematic patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from team_mapping import normalize_team_name

print("="*80)
print("Ball Knower - Phase 1b: Injury Impact Analysis")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\nLoading data...")

# Load schedules
schedules = pd.read_parquet('schedules.parquet')
df = schedules[
    (schedules['game_type'] == 'REG') &
    (schedules['season'] >= 2010) &
    (schedules['away_score'].notna()) &
    (schedules['home_score'].notna()) &
    (schedules['spread_line'].notna())
].copy()

# Compute metrics
df['actual_margin'] = df['home_score'] - df['away_score']
df['vegas_error_spread'] = df['actual_margin'] - df['spread_line']
df['home_covered'] = (df['actual_margin'] > df['spread_line']).astype(int)
df['away_covered'] = (df['actual_margin'] < df['spread_line']).astype(int)

print(f"✓ Loaded {len(df):,} games ({df['season'].min()}-{df['season'].max()})")

# Load injuries
injuries = pd.read_parquet('injuries.parquet')
injuries['team'] = injuries['team'].apply(normalize_team_name)
print(f"✓ Loaded {len(injuries):,} injury records")

# ============================================================================
# 2. IDENTIFY QB INJURIES
# ============================================================================

print("\n" + "="*80)
print("Identifying QB Injuries")
print("="*80)

# Filter to QBs only
qb_injuries = injuries[injuries['position'] == 'QB'].copy()
print(f"\nQB injury records: {len(qb_injuries):,}")

# Status categories
print("\nInjury status distribution:")
status_counts = qb_injuries['report_status'].value_counts()
print(status_counts)

# Focus on "Out" status (definite absence)
qbs_out = qb_injuries[
    (qb_injuries['report_status'] == 'Out') |
    (qb_injuries['report_status'] == 'Injured Reserve')
].copy()

print(f"\nQBs marked 'Out' or 'IR': {len(qbs_out):,} records")
print(f"Unique QBs affected: {qbs_out['full_name'].nunique()}")
print(f"Seasons covered: {qbs_out['season'].min():.0f}-{qbs_out['season'].max():.0f}")

# ============================================================================
# 3. MATCH INJURIES TO GAMES
# ============================================================================

print("\n" + "="*80)
print("Matching QB Injuries to Games")
print("="*80)

# Create game keys for matching
# Note: season and week are floats in injuries data, must convert properly
qbs_out['game_key'] = (
    qbs_out['season'].astype(int).astype(str) + '_' +
    qbs_out['week'].astype(int).astype(str) + '_' +
    qbs_out['team']
)

df['away_game_key'] = (
    df['season'].astype(str) + '_' +
    df['week'].astype(str) + '_' +
    df['away_team']
)

df['home_game_key'] = (
    df['season'].astype(str) + '_' +
    df['week'].astype(str) + '_' +
    df['home_team']
)

# Flag games where QB was out
qb_out_games = set(qbs_out['game_key'].unique())

df['away_qb_out'] = df['away_game_key'].isin(qb_out_games).astype(int)
df['home_qb_out'] = df['home_game_key'].isin(qb_out_games).astype(int)
df['any_qb_out'] = ((df['away_qb_out'] == 1) | (df['home_qb_out'] == 1)).astype(int)

print(f"\nGames with QB out:")
print(f"  Away QB out: {df['away_qb_out'].sum()} games")
print(f"  Home QB out: {df['home_qb_out'].sum()} games")
print(f"  Total games affected: {df['any_qb_out'].sum()} games")

# ============================================================================
# 4. ANALYZE QB-OUT GAME PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("QB-Out Game Analysis")
print("="*80)

# Compare QB-out games vs normal games
qb_out_games_df = df[df['any_qb_out'] == 1].copy()
normal_games_df = df[df['any_qb_out'] == 0].copy()

print(f"\nSample sizes:")
print(f"  QB-out games: {len(qb_out_games_df)}")
print(f"  Normal games: {len(normal_games_df)}")

if len(qb_out_games_df) >= 30:
    print("\n" + "-"*80)
    print("Spread Performance: QB-Out vs Normal")
    print("-"*80)

    # Overall stats
    qb_out_stats = {
        'n_games': len(qb_out_games_df),
        'avg_total': qb_out_games_df['home_score'].mean() + qb_out_games_df['away_score'].mean(),
        'avg_vegas_error_spread': qb_out_games_df['vegas_error_spread'].mean(),
        'home_cover_rate': qb_out_games_df['home_covered'].mean(),
        'away_cover_rate': qb_out_games_df['away_covered'].mean(),
    }

    normal_stats = {
        'n_games': len(normal_games_df),
        'avg_total': normal_games_df['home_score'].mean() + normal_games_df['away_score'].mean(),
        'avg_vegas_error_spread': normal_games_df['vegas_error_spread'].mean(),
        'home_cover_rate': normal_games_df['home_covered'].mean(),
        'away_cover_rate': normal_games_df['away_covered'].mean(),
    }

    comparison = pd.DataFrame({
        'QB-Out Games': qb_out_stats,
        'Normal Games': normal_stats
    }).round(3)

    print(comparison)

    # ========================================================================
    # 5. DIRECTIONAL ANALYSIS - Do teams with QB out underperform spread?
    # ========================================================================

    print("\n" + "-"*80)
    print("Directional Analysis: Team with QB Out")
    print("-"*80)

    # When home team has QB out
    home_qb_out = df[df['home_qb_out'] == 1].copy()
    if len(home_qb_out) >= 30:
        home_qb_out_cover = home_qb_out['home_covered'].mean()
        away_cover_vs_injured_qb = home_qb_out['away_covered'].mean()

        print(f"\nHome QB Out (n={len(home_qb_out)}):")
        print(f"  Home team (injured QB) cover rate: {home_qb_out_cover:.1%}")
        print(f"  Away team (vs injured QB) cover rate: {away_cover_vs_injured_qb:.1%}")

        if away_cover_vs_injured_qb >= 0.524:
            print(f"  ✓ EDGE: Bet AGAINST team with injured QB (away covers {away_cover_vs_injured_qb:.1%})")
        elif home_qb_out_cover >= 0.524:
            print(f"  ✓ EDGE: Bet ON team with injured QB (covered {home_qb_out_cover:.1%})")
        else:
            print(f"  ✗ No edge found")

    # When away team has QB out
    away_qb_out = df[df['away_qb_out'] == 1].copy()
    if len(away_qb_out) >= 30:
        away_qb_out_cover = away_qb_out['away_covered'].mean()
        home_cover_vs_injured_qb = away_qb_out['home_covered'].mean()

        print(f"\nAway QB Out (n={len(away_qb_out)}):")
        print(f"  Away team (injured QB) cover rate: {away_qb_out_cover:.1%}")
        print(f"  Home team (vs injured QB) cover rate: {home_cover_vs_injured_qb:.1%}")

        if home_cover_vs_injured_qb >= 0.524:
            print(f"  ✓ EDGE: Bet AGAINST team with injured QB (home covers {home_cover_vs_injured_qb:.1%})")
        elif away_qb_out_cover >= 0.524:
            print(f"  ✓ EDGE: Bet ON team with injured QB (covered {away_qb_out_cover:.1%})")
        else:
            print(f"  ✗ No edge found")

    # ========================================================================
    # 6. COMBINED RULE: Always bet against team with injured QB?
    # ========================================================================

    print("\n" + "-"*80)
    print("Combined Rule Test")
    print("-"*80)

    # Combine both scenarios: bet against the team with QB out
    # When home QB out → bet away
    # When away QB out → bet home

    bet_against_injured_qb = []

    # Home QB out → bet away
    for _, row in home_qb_out.iterrows():
        bet_against_injured_qb.append(row['away_covered'])

    # Away QB out → bet home
    for _, row in away_qb_out.iterrows():
        bet_against_injured_qb.append(row['home_covered'])

    if len(bet_against_injured_qb) >= 30:
        win_rate_against_injured = np.mean(bet_against_injured_qb)
        print(f"\nRule: 'Always bet AGAINST team with QB out'")
        print(f"  Sample size: {len(bet_against_injured_qb)} games")
        print(f"  Win rate: {win_rate_against_injured:.1%}")

        if win_rate_against_injured >= 0.524:
            print(f"  ✓ EDGE FOUND: Beats 52.4% threshold")
            print(f"  → PROCEED to Phase 2: Build QB injury model")
        else:
            print(f"  ✗ No edge: {win_rate_against_injured:.1%} < 52.4%")
    else:
        print(f"  ⚠️  Insufficient data (n={len(bet_against_injured_qb)})")

else:
    print(f"\n⚠️  Insufficient QB-out games (n={len(qb_out_games_df)}) for analysis")

# ============================================================================
# 7. ALTERNATIVE: Check "Questionable" status
# ============================================================================

print("\n" + "="*80)
print("Alternative: Questionable QB Status")
print("="*80)

qbs_questionable = injuries[
    (injuries['position'] == 'QB') &
    (injuries['report_status'] == 'Questionable')
].copy()

print(f"\nQBs marked 'Questionable': {len(qbs_questionable):,} records")

# This could be a future expansion - "fade questionable QBs"
# For now, focus on definite "Out" status as it's cleaner

print("\n⚠️  Questionable QB analysis deferred - focus on 'Out' status for Phase 1")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("Phase 1b Summary: Injury Edge")
print("="*80)

if len(qb_out_games_df) >= 30 and len(bet_against_injured_qb) >= 30:
    print(f"\n✓ Analysis complete:")
    print(f"  Sample: {len(bet_against_injured_qb)} games with QB out")
    print(f"  Strategy: Bet against team with injured QB")
    print(f"  Win rate: {win_rate_against_injured:.1%}")

    if win_rate_against_injured >= 0.524:
        print(f"\n🚀 RECOMMENDATION: PROCEED TO PHASE 2")
        print(f"   Build QB injury impact model")
    else:
        print(f"\n⏸️  No systematic edge found in QB injuries")
else:
    print(f"\n⚠️  Insufficient data for conclusive analysis")

print("\n" + "="*80)
print("Analysis complete.")
print("="*80)
