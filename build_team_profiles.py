"""
Advanced Team Profiling System

Aggregates play-by-play and advanced stats to create comprehensive team profiles:
- Offensive scheme tendencies (play-action, RPO, motion, no-huddle rates)
- Defensive coverage schemes (blitz rate, box counts, pressure rates)
- Matchup-specific advantages (scheme vs scheme)
- Personnel & execution metrics (drops, bad throws, missed tackles)

This creates the foundation for matchup analysis and our own line generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("BUILDING COMPREHENSIVE TEAM PROFILES")
print("="*80)

# ============================================================================
# LOAD FTN CHARTING DATA (Scheme & Coverage)
# ============================================================================

print("\n[1/5] Loading FTN charting data (play-by-play schemes)...")

ftn = pd.read_parquet('/home/user/BK_Build/ftn_charting.parquet')

# Filter to regular season
ftn = ftn[ftn['nflverse_game_id'].notna()].copy()

# Extract season, week, teams from game_id
ftn[['season', 'week', 'away', 'home']] = ftn['nflverse_game_id'].str.extract(r'(\d{4})_(\d+)_(\w+)_(\w+)')
ftn['season'] = ftn['season'].astype(int)
ftn['week'] = ftn['week'].astype(int)

print(f"✓ Loaded {len(ftn):,} plays ({ftn['season'].min()}-{ftn['season'].max()})")

# ============================================================================
# AGGREGATE OFFENSIVE SCHEME TENDENCIES (Team-Week Level)
# ============================================================================

print("\n[2/5] Aggregating offensive scheme tendencies...")

# Need to determine which team is on offense for each play
# FTN data doesn't have this directly, so we'll aggregate at game level for now
# and split by team later using nflverse play-by-play if needed

# Group by game and aggregate
offense_schemes = ftn.groupby(['season', 'week', 'nflverse_game_id']).agg({
    # Play-calling tendencies
    'is_play_action': 'mean',
    'is_rpo': 'mean',
    'is_screen_pass': 'mean',
    'is_motion': 'mean',
    'is_no_huddle': 'mean',
    'is_qb_sneak': 'mean',
    'is_trick_play': 'mean',

    # QB behavior
    'is_qb_out_of_pocket': 'mean',

    # Pass quality
    'is_throw_away': 'mean',
    'is_interception_worthy': 'mean',
    'is_drop': 'mean',
    'is_catchable_ball': 'mean',
    'is_contested_ball': 'mean',
    'is_created_reception': 'mean',

    # Formation
    'n_offense_backfield': 'mean'
}).reset_index()

print(f"✓ Created offensive scheme profiles for {len(offense_schemes):,} team-games")

# ============================================================================
# AGGREGATE DEFENSIVE SCHEME TENDENCIES
# ============================================================================

print("\n[3/5] Aggregating defensive scheme tendencies...")

defense_schemes = ftn.groupby(['season', 'week', 'nflverse_game_id']).agg({
    # Defensive pressure
    'n_blitzers': 'mean',
    'n_defense_box': 'mean',
    'n_pass_rushers': 'mean',

    # QB pressure results
    'is_qb_fault_sack': 'mean'
}).reset_index()

print(f"✓ Created defensive scheme profiles for {len(defense_schemes):,} team-games")

# ============================================================================
# LOAD PFR ADVANCED STATS
# ============================================================================

print("\n[4/5] Loading PFR advanced defensive stats...")

pfr_def = pd.read_parquet('/home/user/BK_Build/pfr_adv_def_week.parquet')

# Aggregate to team-week level
team_defense = pfr_def.groupby(['season', 'week', 'team']).agg({
    # Coverage quality
    'def_targets': 'sum',
    'def_completions_allowed': 'sum',
    'def_completion_pct': 'mean',
    'def_yards_allowed': 'sum',
    'def_yards_allowed_per_tgt': 'mean',
    'def_yards_after_catch': 'sum',

    # Pass rush
    'def_pressures': 'sum',
    'def_sacks': 'sum',
    'def_times_blitzed': 'sum',
    'def_times_hitqb': 'sum',
    'def_times_hurried': 'sum',

    # Tackling
    'def_missed_tackles': 'sum',
    'def_missed_tackle_pct': 'mean',

    # Allowed
    'def_ints': 'sum',
    'def_receiving_td_allowed': 'sum',
    'def_passer_rating_allowed': 'mean'
}).reset_index()

print(f"✓ Created defensive quality profiles for {len(team_defense):,} team-weeks")

# Load PFR advanced passing
pfr_pass = pd.read_parquet('/home/user/BK_Build/pfr_adv_pass_week.parquet')

team_offense = pfr_pass.groupby(['season', 'week', 'team']).agg({
    # Pressure faced
    'times_pressured': 'sum',
    'times_pressured_pct': 'mean',
    'times_blitzed': 'sum',
    'times_hurried': 'sum',
    'times_hit': 'sum',
    'times_sacked': 'sum',

    # Execution
    'passing_bad_throws': 'sum',
    'passing_bad_throw_pct': 'mean',
    'passing_drops': 'sum',
    'passing_drop_pct': 'mean'
}).reset_index()

print(f"✓ Created offensive quality profiles for {len(team_offense):,} team-weeks")

# ============================================================================
# COMBINE INTO COMPREHENSIVE TEAM PROFILES
# ============================================================================

print("\n[5/5] Combining all metrics into team profiles...")

# Merge defensive and offensive stats
team_profiles = team_defense.merge(
    team_offense,
    on=['season', 'week', 'team'],
    how='outer',
    suffixes=('_def', '_off')
)

# Add derived metrics
# Pressure rate (as defense)
team_profiles['def_pressure_rate'] = team_profiles['def_pressures'] / team_profiles['def_targets'].replace(0, 1)

# Coverage quality (completion % allowed - lower is better)
team_profiles['def_coverage_quality'] = 1 - team_profiles['def_completion_pct']

# Tackling efficiency
team_profiles['def_tackle_efficiency'] = 1 - team_profiles['def_missed_tackle_pct']

# Offensive line quality (inverse of pressure rate)
team_profiles['off_oline_quality'] = 1 - team_profiles['times_pressured_pct']

# Passing accuracy
team_profiles['off_passing_accuracy'] = 1 - team_profiles['passing_bad_throw_pct']

# WR quality (inverse of drop rate)
team_profiles['off_wr_quality'] = 1 - team_profiles['passing_drop_pct']

print(f"✓ Created comprehensive profiles for {len(team_profiles):,} team-weeks")
print(f"  Total features: {len(team_profiles.columns)}")

# ============================================================================
# SAVE TEAM PROFILES
# ============================================================================

output_file = Path('/home/user/BK_Build/team_profiles_advanced_2018_2025.csv')
team_profiles.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("TEAM PROFILE SUMMARY")
print("="*80)

print(f"\nData Coverage:")
print(f"  Seasons: {team_profiles['season'].min()}-{team_profiles['season'].max()}")
print(f"  Total team-weeks: {len(team_profiles):,}")
print(f"  Teams: {team_profiles['team'].nunique()}")

print(f"\nFeature Categories:")
print(f"  • Coverage Metrics: completion%, yards/target, YAC allowed")
print(f"  • Pass Rush: pressures, sacks, blitzes, hits, hurries")
print(f"  • Tackling: missed tackles, tackle efficiency")
print(f"  • Offensive Line: pressure%, times blitzed, protection quality")
print(f"  • Passing Execution: bad throw%, accuracy")
print(f"  • Receiving: drop%, WR quality")

print(f"\nDerived Quality Metrics:")
print(f"  • def_coverage_quality: {team_profiles['def_coverage_quality'].mean():.3f} avg")
print(f"  • def_pressure_rate: {team_profiles['def_pressure_rate'].mean():.3f} avg")
print(f"  • def_tackle_efficiency: {team_profiles['def_tackle_efficiency'].mean():.3f} avg")
print(f"  • off_oline_quality: {team_profiles['off_oline_quality'].mean():.3f} avg")
print(f"  • off_passing_accuracy: {team_profiles['off_passing_accuracy'].mean():.3f} avg")
print(f"  • off_wr_quality: {team_profiles['off_wr_quality'].mean():.3f} avg")

print(f"\n2025 Coverage:")
if 2025 in team_profiles['season'].values:
    print(f"  2025 team-weeks: {len(team_profiles[team_profiles['season']==2025]):,}")
else:
    print(f"  2025 data: Not yet available (will update weekly)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
Team profiles created! Now we can:

1. Build matchup-specific features:
   - Offensive scheme vs defensive coverage
   - Pass rush quality vs OL quality
   - WR quality vs coverage quality

2. Create rolling team profiles:
   - 3/5/10 game averages of these metrics
   - Trend analysis (improving vs declining)

3. Generate our own point differential predictions:
   - Model expected points based on matchups
   - Compare to Vegas lines to find value

4. Build matchup matrices:
   - How does this offense perform vs this defense?
   - Historical matchup analysis

Ready to build the matchup engine!
""")

print("="*80 + "\n")
