"""
Explore Available Data Sources for Ball Knower v2.0
====================================================

This script inventories all available data sources and shows what we can use
for QB adjustments, Next Gen Stats, and recent performance trends.

Data sources to explore:
1. ESPN QBR (weekly and season)
2. Next Gen Stats (passing, receiving, rushing)
3. Team EPA data (offense/defense)
4. Current week QB rankings
5. Injury data
6. PFF/advanced stats

Author: Ball Knower Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 50)

print("\n" + "="*80)
print("BALL KNOWER v2.0 - DATA INVENTORY")
print("="*80)

# ============================================================================
# 1. ESPN QBR DATA
# ============================================================================

print("\n" + "="*80)
print("1. ESPN QBR DATA")
print("="*80)

try:
    qbr_week = pd.read_parquet('espn_qbr_week.parquet')
    print(f"\n✓ ESPN QBR Weekly: {len(qbr_week)} rows")
    print(f"  Seasons: {qbr_week['season'].min()}-{qbr_week['season'].max()}")
    print(f"  Columns: {list(qbr_week.columns)}")
    print(f"\n  Sample (2024 Week 10):")
    sample = qbr_week[(qbr_week['season'] == 2024) & (qbr_week['week'] == 10)].head(5)
    print(sample[['season', 'week', 'team', 'player_display_name', 'qbr_total', 'qb_plays', 'epa_total']].to_string(index=False))
except Exception as e:
    print(f"\n🔴 ESPN QBR Weekly: {e}")

try:
    qbr_season = pd.read_parquet('espn_qbr_season.parquet')
    print(f"\n✓ ESPN QBR Season: {len(qbr_season)} rows")
    print(f"  Seasons: {qbr_season['season'].min()}-{qbr_season['season'].max()}")
    print(f"  Columns: {list(qbr_season.columns)}")
except Exception as e:
    print(f"\n🔴 ESPN QBR Season: {e}")

# ============================================================================
# 2. NEXT GEN STATS
# ============================================================================

print("\n" + "="*80)
print("2. NEXT GEN STATS")
print("="*80)

try:
    ngs_passing = pd.read_parquet('ngs_passing.parquet')
    print(f"\n✓ NGS Passing: {len(ngs_passing)} rows")
    print(f"  Seasons: {ngs_passing['season'].min()}-{ngs_passing['season'].max()}")
    print(f"  Columns: {list(ngs_passing.columns)}")
    print(f"\n  Sample (2024 Week 10):")
    sample = ngs_passing[(ngs_passing['season'] == 2024) & (ngs_passing['week'] == 10)].head(5)
    if len(sample) > 0:
        cols = ['season', 'week', 'team_abbr', 'player_display_name', 'completions', 'attempts', 'pass_yards', 'pass_touchdowns']
        print(sample[cols].to_string(index=False))
except Exception as e:
    print(f"\n🔴 NGS Passing: {e}")

try:
    ngs_receiving = pd.read_parquet('ngs_receiving.parquet')
    print(f"\n✓ NGS Receiving: {len(ngs_receiving)} rows")
    print(f"  Seasons: {ngs_receiving['season'].min()}-{ngs_receiving['season'].max()}")
except Exception as e:
    print(f"\n🔴 NGS Receiving: {e}")

try:
    ngs_rushing = pd.read_parquet('ngs_rushing.parquet')
    print(f"\n✓ NGS Rushing: {len(ngs_rushing)} rows")
    print(f"  Seasons: {ngs_rushing['season'].min()}-{ngs_rushing['season'].max()}")
except Exception as e:
    print(f"\n🔴 NGS Rushing: {e}")

# ============================================================================
# 3. TEAM EPA DATA
# ============================================================================

print("\n" + "="*80)
print("3. TEAM EPA DATA")
print("="*80)

try:
    team_epa = pd.read_csv('team_week_epa_2013_2024.csv')
    print(f"\n✓ Team Week EPA: {len(team_epa)} rows")
    print(f"  Seasons: {team_epa['season'].min()}-{team_epa['season'].max()}")
    print(f"  Columns: {list(team_epa.columns)}")
    print(f"\n  Sample (2024 Week 10):")
    sample = team_epa[(team_epa['season'] == 2024) & (team_epa['week'] == 10)].head(5)
    if len(sample) > 0:
        print(sample.to_string(index=False))
except Exception as e:
    print(f"\n🔴 Team Week EPA: {e}")

# ============================================================================
# 4. INJURIES DATA
# ============================================================================

print("\n" + "="*80)
print("4. INJURIES DATA")
print("="*80)

try:
    injuries = pd.read_parquet('injuries.parquet')
    print(f"\n✓ Injuries: {len(injuries)} rows")
    print(f"  Seasons: {injuries['season'].min()}-{injuries['season'].max()}")
    print(f"  Columns: {list(injuries.columns)}")
    print(f"\n  Sample (2024 Week 10, QB only):")
    sample = injuries[(injuries['season'] == 2024) & (injuries['week'] == 10) & (injuries['position'] == 'QB')].head(10)
    if len(sample) > 0:
        cols = ['season', 'week', 'team', 'full_name', 'position', 'report_primary_injury', 'report_status']
        print(sample[cols].to_string(index=False))
except Exception as e:
    print(f"\n🔴 Injuries: {e}")

# ============================================================================
# 5. CURRENT WEEK DATA (Week 11 2025)
# ============================================================================

print("\n" + "="*80)
print("5. CURRENT WEEK DATA (Week 11 2025)")
print("="*80)

current_season_dir = Path('data/current_season')

try:
    qb_rankings = pd.read_csv(current_season_dir / 'nfelo_qb_rankings_2025_week_11.csv')
    print(f"\n✓ nfelo QB Rankings Week 11: {len(qb_rankings)} rows")
    print(f"  Columns: {list(qb_rankings.columns)}")
    print(f"\n  Top 10 QBs:")
    print(qb_rankings.head(10).to_string(index=False))
except Exception as e:
    print(f"\n🔴 nfelo QB Rankings: {e}")

try:
    qb_epa = pd.read_csv(current_season_dir / 'substack_qb_epa_2025_week_11.csv')
    print(f"\n✓ Substack QB EPA Week 11: {len(qb_epa)} rows")
    print(f"  Columns: {list(qb_epa.columns)}")
    print(f"\n  Sample:")
    print(qb_epa.head(10).to_string(index=False))
except Exception as e:
    print(f"\n🔴 Substack QB EPA: {e}")

try:
    team_epa_tiers = pd.read_csv(current_season_dir / 'nfelo_epa_tiers_off_def_2025_week_11.csv')
    print(f"\n✓ nfelo EPA Tiers Week 11: {len(team_epa_tiers)} rows")
    print(f"  Columns: {list(team_epa_tiers.columns)}")
    print(f"\n  All teams:")
    print(team_epa_tiers.to_string(index=False))
except Exception as e:
    print(f"\n🔴 nfelo EPA Tiers: {e}")

# ============================================================================
# 6. ADVANCED STATS (PFR)
# ============================================================================

print("\n" + "="*80)
print("6. PRO FOOTBALL REFERENCE ADVANCED STATS")
print("="*80)

try:
    pfr_pass = pd.read_parquet('pfr_adv_pass_week.parquet')
    print(f"\n✓ PFR Advanced Passing: {len(pfr_pass)} rows")
    print(f"  Seasons: {pfr_pass['season'].min()}-{pfr_pass['season'].max()}")
    print(f"  Columns: {list(pfr_pass.columns)}")
except Exception as e:
    print(f"\n🔴 PFR Advanced Passing: {e}")

try:
    pfr_def = pd.read_parquet('pfr_adv_def_week.parquet')
    print(f"\n✓ PFR Advanced Defense: {len(pfr_def)} rows")
    print(f"  Seasons: {pfr_def['season'].min()}-{pfr_def['season'].max()}")
    print(f"  Columns: {list(pfr_def.columns)}")
except Exception as e:
    print(f"\n🔴 PFR Advanced Defense: {e}")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("DATA AVAILABILITY SUMMARY")
print("="*80)

print("""
✅ AVAILABLE DATA FOR v2.0 ENHANCEMENTS:

1. QB PERFORMANCE:
   - ESPN QBR (weekly 2006-2024)
   - Next Gen Stats passing (weekly 2016-2024)
   - PFR advanced passing stats
   - Current week QB rankings (nfelo, Substack)

2. RECENT TRENDS:
   - Team EPA by week (2013-2024)
   - Current week EPA tiers (offense/defense)
   - Rolling stats can be calculated from weekly data

3. INJURIES:
   - Injury reports with status (2009-2024)
   - Position-specific injury tracking
   - Need current week injury data

4. NEXT GEN STATS:
   - Passing: completion %, air yards, time to throw, etc.
   - Receiving: routes, targets, separation
   - Rushing: rush yards over expected, efficiency

RECOMMENDED FEATURE ENGINEERING:

Priority 1 - QB Adjustments:
  - Identify starting QB for each team/week
  - Calculate rolling QB EPA (last 3-5 games)
  - Detect QB injuries/changes
  - Adjust team ratings when backup QB starts

Priority 2 - Recent Team Performance:
  - Rolling team EPA (offense/defense, last 3-5 games)
  - Momentum indicators (trend direction)
  - Form-weighted ratings (recent > distant)

Priority 3 - Next Gen Stats Integration:
  - QB efficiency metrics (completion % over expected)
  - Pressure stats (time to throw, sack rate)
  - Explosive play rates

Priority 4 - Structural Features:
  - Already have: rest_advantage, div_game
  - Add: weather, dome vs outdoor, travel distance
  - Fix: surface_mod, time_advantage (currently 0.0)
""")

print("\n" + "="*80 + "\n")
