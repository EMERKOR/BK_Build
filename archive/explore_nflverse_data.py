"""
Explore NFLverse data sources for EPA and advanced metrics

Investigates:
- Play-by-play data (EPA, success rate, explosive plays)
- Team stats (offensive/defensive efficiency)
- Weather data
- Player participation
- Injury reports
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("NFLVERSE DATA EXPLORATION")
print("="*80)

# ============================================================================
# PLAY-BY-PLAY DATA
# ============================================================================

print("\n[1/5] Loading play-by-play data (2024 season sample)...")

try:
    import nfl_data_py as nfl

    # Load 2024 play-by-play data
    pbp = nfl.import_pbp_data([2024])

    print(f"  Loaded {len(pbp):,} plays from 2024 season")
    print(f"  Columns: {len(pbp.columns)}")

    # Find EPA-related columns
    epa_cols = [col for col in pbp.columns if 'epa' in col.lower()]
    success_cols = [col for col in pbp.columns if 'success' in col.lower()]
    win_prob_cols = [col for col in pbp.columns if 'wp' in col.lower()]

    print(f"\nEPA-related columns ({len(epa_cols)}):")
    for col in epa_cols[:10]:
        print(f"    {col}")

    print(f"\nSuccess rate columns ({len(success_cols)}):")
    for col in success_cols[:10]:
        print(f"    {col}")

    print(f"\nWin probability columns ({len(win_prob_cols)}):")
    for col in win_prob_cols[:10]:
        print(f"    {col}")

    # Sample a single game
    print("\n" + "="*80)
    print("SAMPLE GAME ANALYSIS")
    print("="*80)

    sample_game = pbp[pbp['game_id'] == pbp['game_id'].iloc[0]]
    print(f"\nGame: {sample_game.iloc[0]['game_id']}")
    print(f"Plays: {len(sample_game)}")

    # Aggregate EPA by team for this game
    game_id = sample_game.iloc[0]['game_id']
    home_team = sample_game.iloc[0]['home_team']
    away_team = sample_game.iloc[0]['away_team']

    print(f"{away_team} @ {home_team}")

    # Offensive EPA
    home_off_epa = sample_game[sample_game['posteam'] == home_team]['epa'].sum()
    away_off_epa = sample_game[sample_game['posteam'] == away_team]['epa'].sum()

    print(f"\nOffensive EPA:")
    print(f"  {home_team}: {home_off_epa:.2f}")
    print(f"  {away_team}: {away_off_epa:.2f}")

    # Success rate
    home_success = sample_game[sample_game['posteam'] == home_team]['success'].mean()
    away_success = sample_game[sample_game['posteam'] == away_team]['success'].mean()

    print(f"\nSuccess Rate:")
    print(f"  {home_team}: {home_success:.2%}")
    print(f"  {away_team}: {away_success:.2%}")

    # Explosive plays (EPA > 0.5)
    home_explosive = (sample_game[sample_game['posteam'] == home_team]['epa'] > 0.5).sum()
    away_explosive = (sample_game[sample_game['posteam'] == away_team]['epa'] > 0.5).sum()

    print(f"\nExplosive Plays (EPA > 0.5):")
    print(f"  {home_team}: {home_explosive}")
    print(f"  {away_team}: {away_explosive}")

except Exception as e:
    print(f"  ERROR: {e}")
    print("  Note: nfl_data_py is deprecated, but still works for exploration")

# ============================================================================
# TEAM STATS
# ============================================================================

print("\n" + "="*80)
print("[2/5] Loading team stats...")
print("="*80)

try:
    # Weekly team stats
    team_stats = nfl.import_weekly_data([2024])

    print(f"  Loaded {len(team_stats):,} team-week records from 2024")
    print(f"  Columns: {len(team_stats.columns)}")

    # Find efficiency-related columns
    efficiency_cols = [col for col in team_stats.columns if any(x in col.lower()
                      for x in ['epa', 'success', 'rate', 'avg', 'efficiency'])]

    print(f"\nEfficiency-related columns ({len(efficiency_cols)}):")
    for col in sorted(efficiency_cols)[:20]:
        print(f"    {col}")

except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# SCHEDULES (for rest, travel, etc.)
# ============================================================================

print("\n" + "="*80)
print("[3/5] Loading schedules...")
print("="*80)

try:
    schedules = nfl.import_schedules([2024])

    print(f"  Loaded {len(schedules):,} games from 2024")
    print(f"  Columns: {len(schedules.columns)}")

    # Find relevant columns
    travel_cols = [col for col in schedules.columns if any(x in col.lower()
                  for x in ['roof', 'surface', 'temp', 'wind', 'weather', 'stadium'])]

    print(f"\nTravel/Weather columns ({len(travel_cols)}):")
    for col in sorted(travel_cols):
        print(f"    {col}")

    # Sample game with weather
    sample_sched = schedules[schedules['temp'].notna()].iloc[0]
    print(f"\nSample game with weather:")
    print(f"  {sample_sched['away_team']} @ {sample_sched['home_team']}")
    print(f"  Temp: {sample_sched['temp']}°F")
    if 'wind' in schedules.columns and pd.notna(sample_sched.get('wind')):
        print(f"  Wind: {sample_sched['wind']} mph")
    print(f"  Roof: {sample_sched['roof']}")
    print(f"  Surface: {sample_sched['surface']}")

except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# ROSTERS (for player availability)
# ============================================================================

print("\n" + "="*80)
print("[4/5] Loading rosters...")
print("="*80)

try:
    rosters = nfl.import_rosters([2024])

    print(f"  Loaded {len(rosters):,} player records from 2024")
    print(f"  Positions: {sorted(rosters['position'].unique())}")

    # QB sample
    qbs = rosters[rosters['position'] == 'QB']
    print(f"\n  QBs: {len(qbs)}")
    print(f"  Sample: {qbs['full_name'].head(10).tolist()}")

except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# INJURIES
# ============================================================================

print("\n" + "="*80)
print("[5/5] Loading injuries...")
print("="*80)

try:
    injuries = nfl.import_injuries([2024])

    print(f"  Loaded {len(injuries):,} injury records from 2024")
    print(f"  Columns: {injuries.columns.tolist()}")

    # Sample
    sample_injuries = injuries.head(5)
    print(f"\nSample injury records:")
    print(sample_injuries[['report_status', 'position', 'full_name']].to_string(index=False))

except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "="*80)
print("KEY FINDINGS FOR v1.3 FEATURE ENGINEERING")
print("="*80)

print("""
Available Data Sources:

1. PLAY-BY-PLAY (EPA Gold Mine):
   - epa: Expected Points Added per play
   - success: Binary success indicator
   - explosive plays: High-value plays
   - Can aggregate by team, situation, down/distance
   - Can build rolling averages for recent form

2. WEEKLY TEAM STATS:
   - Pre-aggregated EPA/play metrics
   - Offensive/defensive efficiency
   - Success rates by situation
   - Faster than aggregating play-by-play

3. SCHEDULES (Contextual Factors):
   - Temperature, wind (weather)
   - Roof type (dome vs outdoor)
   - Surface (grass vs turf)
   - Can calculate travel distance from stadium locations

4. ROSTERS + INJURIES:
   - QB availability (already using 538 QB adj)
   - Could enhance with injury-adjusted depth

Recommended v1.3 Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 1 (Highest Priority):
  ✓ EPA/play differential (offense + defense)
  ✓ Success rate differential
  ✓ Recent form (last 3-5 games EPA)
  ✓ Explosive play rate differential

TIER 2 (High Value):
  ✓ Schedule-adjusted EPA (DVOA-like)
  ✓ Situational EPA (3rd down, red zone, 2-min drill)
  ✓ Home/away splits
  ✓ Weather impact (temp, wind for outdoor games)

TIER 3 (Nice to Have):
  ✓ Pass vs run EPA splits
  ✓ Early down vs late down efficiency
  ✓ Scoring drive rate
  ✓ Turnover-adjusted EPA

Implementation Strategy:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Use WEEKLY_DATA (faster, pre-aggregated)
2. Calculate rolling averages (3-5 game windows)
3. Build opponent adjustments (schedule strength)
4. Merge with existing nfelo + situational features
5. Train v1.3 and compare to v1.2 on same test set
6. Measure improvement via CLV and prediction accuracy

IMPORTANT: Keep v1.2 features as baseline - add EPA features incrementally
to measure marginal value of each feature tier.
""")

print("="*80 + "\n")
