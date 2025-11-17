"""
Aggregate Next Gen Stats to Team-Week Level

Takes player-level NGS data and aggregates to team-week for modeling.
Key metrics:
- CPOE (completion percentage over expectation)
- Time to throw
- Aggressiveness
- Rushing efficiency
- Receiver separation
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("AGGREGATING NEXT GEN STATS TO TEAM-WEEK LEVEL")
print("="*80)

# ============================================================================
# LOAD NGS DATA
# ============================================================================

print("\n[1/4] Loading NGS data...")

ngs_passing = pd.read_parquet('/home/user/BK_Build/ngs_passing.parquet')
ngs_rushing = pd.read_parquet('/home/user/BK_Build/ngs_rushing.parquet')
ngs_receiving = pd.read_parquet('/home/user/BK_Build/ngs_receiving.parquet')

# Filter to regular season only
ngs_passing = ngs_passing[ngs_passing['season_type'] == 'REG'].copy()
ngs_rushing = ngs_rushing[ngs_rushing['season_type'] == 'REG'].copy()
ngs_receiving = ngs_receiving[ngs_receiving['season_type'] == 'REG'].copy()

print(f"✓ Passing: {len(ngs_passing):,} player-weeks ({ngs_passing['season'].min()}-{ngs_passing['season'].max()})")
print(f"✓ Rushing: {len(ngs_rushing):,} player-weeks ({ngs_rushing['season'].min()}-{ngs_rushing['season'].max()})")
print(f"✓ Receiving: {len(ngs_receiving):,} player-weeks ({ngs_receiving['season'].min()}-{ngs_receiving['season'].max()})")

# ============================================================================
# AGGREGATE PASSING NGS
# ============================================================================

print("\n[2/4] Aggregating passing NGS to team-week...")

# Group by team-week and aggregate (weighted by attempts)
passing_team = ngs_passing.groupby(['season', 'week', 'team_abbr']).apply(
    lambda x: pd.Series({
        'cpoe': np.average(x['completion_percentage_above_expectation'], weights=x['attempts']) if x['attempts'].sum() > 0 else 0,
        'avg_time_to_throw': np.average(x['avg_time_to_throw'], weights=x['attempts']) if x['attempts'].sum() > 0 else 0,
        'aggressiveness': np.average(x['aggressiveness'], weights=x['attempts']) if x['attempts'].sum() > 0 else 0,
        'avg_completed_air_yards': np.average(x['avg_completed_air_yards'], weights=x['attempts']) if x['attempts'].sum() > 0 else 0,
        'total_attempts': x['attempts'].sum()
    })
).reset_index()

passing_team = passing_team.rename(columns={'team_abbr': 'team'})

print(f"✓ Created {len(passing_team):,} team-week passing records")
print(f"  Metrics: CPOE, time to throw, aggressiveness, completed air yards")

# ============================================================================
# AGGREGATE RUSHING NGS
# ============================================================================

print("\n[3/4] Aggregating rushing NGS to team-week...")

# Group by team-week
rushing_team = ngs_rushing.groupby(['season', 'week', 'team_abbr']).apply(
    lambda x: pd.Series({
        'rush_efficiency': np.average(x['efficiency'], weights=x['rush_attempts']) if x['rush_attempts'].sum() > 0 else 0,
        'pct_8plus_defenders': np.average(x['percent_attempts_gte_eight_defenders'], weights=x['rush_attempts']) if x['rush_attempts'].sum() > 0 else 0,
        'avg_time_to_los': np.average(x['avg_time_to_los'], weights=x['rush_attempts']) if x['rush_attempts'].sum() > 0 else 0,
        'total_rush_attempts': x['rush_attempts'].sum()
    })
).reset_index()

rushing_team = rushing_team.rename(columns={'team_abbr': 'team'})

print(f"✓ Created {len(rushing_team):,} team-week rushing records")
print(f"  Metrics: efficiency, 8+ defenders %, time to LOS")

# ============================================================================
# AGGREGATE RECEIVING NGS
# ============================================================================

print("\n[4/4] Aggregating receiving NGS to team-week...")

# Group by team-week
receiving_team = ngs_receiving.groupby(['season', 'week', 'team_abbr']).apply(
    lambda x: pd.Series({
        'avg_separation': np.average(x['avg_separation'], weights=x['targets']) if x['targets'].sum() > 0 else 0,
        'avg_cushion': np.average(x['avg_cushion'], weights=x['targets']) if x['targets'].sum() > 0 else 0,
        'avg_yac': np.average(x['avg_yac'], weights=x['receptions']) if x['receptions'].sum() > 0 else 0,
        'total_targets': x['targets'].sum()
    })
).reset_index()

receiving_team = receiving_team.rename(columns={'team_abbr': 'team'})

print(f"✓ Created {len(receiving_team):,} team-week receiving records")
print(f"  Metrics: separation, cushion, YAC")

# ============================================================================
# MERGE ALL NGS DATA
# ============================================================================

print("\n" + "="*80)
print("MERGING NGS DATA")
print("="*80)

# Merge all three
ngs_team = passing_team.merge(
    rushing_team,
    on=['season', 'week', 'team'],
    how='outer'
).merge(
    receiving_team,
    on=['season', 'week', 'team'],
    how='outer'
)

# Fill missing values with 0
ngs_team = ngs_team.fillna(0)

print(f"\n✓ Combined NGS data: {len(ngs_team):,} team-week records")
print(f"  Seasons: {ngs_team['season'].min()}-{ngs_team['season'].max()}")
print(f"  Total features: {len(ngs_team.columns) - 3}")  # -3 for season, week, team

# Show sample
print("\nSample data:")
print(ngs_team.head())

# ============================================================================
# SAVE
# ============================================================================

output_file = '/home/user/BK_Build/team_week_ngs_2016_2025.csv'
ngs_team.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")

print("\n" + "="*80)
print("NGS AGGREGATION COMPLETE")
print("="*80)
print(f"""
Summary:
- {len(ngs_team):,} team-week records (2016-2025)
- 10 NGS features:
  Passing: CPOE, time to throw, aggressiveness, completed air yards
  Rushing: efficiency, 8+ defenders %, time to LOS
  Receiving: separation, cushion, YAC

Ready for v1.4 model integration!
""")
