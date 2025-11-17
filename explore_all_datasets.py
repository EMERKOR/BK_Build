"""
COMPREHENSIVE DATA EXPLORATION

Systematically explores ALL available datasets to identify:
1. Data coverage (seasons, weeks, completeness)
2. Key features that could improve predictions
3. Data quality and availability for modeling

Datasets to explore:
✓ Already used: schedules, EPA, NGS, team_profiles, FTN charting, PFR adv def/pass
✗ Not yet used: QBR, injuries, officials, PFR adv rec/rush, player stats,
                snap counts, rosters, team stats, draft picks, trades
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("COMPREHENSIVE DATA EXPLORATION - ALL DATASETS")
print("="*80)

datasets = {}

# ============================================================================
# 1. ESPN QBR (WEEK LEVEL)
# ============================================================================

print("\n" + "="*80)
print("1. ESPN QBR (WEEK LEVEL)")
print("="*80)

qbr_week = pd.read_parquet('/home/user/BK_Build/espn_qbr_week.parquet')
datasets['qbr_week'] = qbr_week

print(f"\nShape: {qbr_week.shape}")
print(f"Seasons: {qbr_week['season'].min()}-{qbr_week['season'].max()}")
print(f"Total QB-weeks: {len(qbr_week):,}")

print("\nKey Columns:")
for col in qbr_week.columns:
    non_null = qbr_week[col].notna().sum()
    pct = non_null / len(qbr_week) * 100
    print(f"  {col:<30} {non_null:>6,} / {len(qbr_week):,} ({pct:>5.1f}%)")

print("\nSample QBR Metrics:")
print(qbr_week[['season', 'week_num', 'team_abb', 'qbr_total', 'pts_added']].head(10))

print("\nKey Insights:")
print(f"  • QBR available for {qbr_week['season'].nunique()} seasons")
print(f"  • Coverage: {len(qbr_week):,} QB performances")
print(f"  • Metrics: QBR, points added, play action stats, EPA")

# ============================================================================
# 2. INJURIES
# ============================================================================

print("\n" + "="*80)
print("2. INJURIES")
print("="*80)

injuries = pd.read_parquet('/home/user/BK_Build/injuries.parquet')
datasets['injuries'] = injuries

print(f"\nShape: {injuries.shape}")
print(f"Seasons: {injuries['season'].min()}-{injuries['season'].max()}")
print(f"Total injury reports: {len(injuries):,}")

print("\nKey Columns:")
for col in injuries.columns[:15]:  # First 15 columns
    print(f"  {col}")

print("\nSample:")
print(injuries[['season', 'week', 'team', 'full_name', 'report_status',
                'position']].head(10))

# Analyze by status
if 'report_status' in injuries.columns:
    print("\nInjury Status Distribution:")
    print(injuries['report_status'].value_counts())

# Analyze by position
if 'position' in injuries.columns:
    print("\nTop 10 Positions with Injuries:")
    print(injuries['position'].value_counts().head(10))

print("\nKey Insights:")
print(f"  • {len(injuries):,} injury reports")
print(f"  • Could aggregate to team-week injury impact")
print(f"  • Focus on QB, WR, OL, CB, EDGE injuries")

# ============================================================================
# 3. OFFICIALS (REFEREE STATS)
# ============================================================================

print("\n" + "="*80)
print("3. OFFICIALS (REFEREE STATS)")
print("="*80)

officials = pd.read_parquet('/home/user/BK_Build/officials.parquet')
datasets['officials'] = officials

print(f"\nShape: {officials.shape}")
print(f"Total games with official data: {len(officials):,}")

print("\nKey Columns:")
for col in officials.columns:
    print(f"  {col}")

print("\nSample:")
print(officials.head(10))

# Check seasons
if 'season' in officials.columns:
    print(f"\nSeasons: {officials['season'].min()}-{officials['season'].max()}")

print("\nKey Insights:")
print(f"  • Referee assignments by game")
print(f"  • Could analyze home/away bias by ref")
print(f"  • Penalty tendencies (high/low flag refs)")

# ============================================================================
# 4. PFR ADVANCED RECEIVING
# ============================================================================

print("\n" + "="*80)
print("4. PFR ADVANCED RECEIVING")
print("="*80)

pfr_rec = pd.read_parquet('/home/user/BK_Build/pfr_adv_rec_week.parquet')
datasets['pfr_rec'] = pfr_rec

print(f"\nShape: {pfr_rec.shape}")
print(f"Seasons: {pfr_rec['season'].min()}-{pfr_rec['season'].max()}")
print(f"Total player-weeks: {len(pfr_rec):,}")

print("\nKey Columns:")
for col in pfr_rec.columns:
    print(f"  {col}")

print("\nSample:")
print(pfr_rec.head(10))

# Key metrics
print("\nKey Receiving Metrics Available:")
rec_metrics = [col for col in pfr_rec.columns if 'rec_' in col or 'target' in col]
for metric in rec_metrics[:10]:
    print(f"  • {metric}")

print("\nKey Insights:")
print(f"  • Player-level receiving advanced stats")
print(f"  • Can aggregate to team level")
print(f"  • Already partially used in team profiles")

# ============================================================================
# 5. PFR ADVANCED RUSHING
# ============================================================================

print("\n" + "="*80)
print("5. PFR ADVANCED RUSHING")
print("="*80)

pfr_rush = pd.read_parquet('/home/user/BK_Build/pfr_adv_rush_week.parquet')
datasets['pfr_rush'] = pfr_rush

print(f"\nShape: {pfr_rush.shape}")
print(f"Seasons: {pfr_rush['season'].min()}-{pfr_rush['season'].max()}")
print(f"Total player-weeks: {len(pfr_rush):,}")

print("\nKey Columns:")
for col in pfr_rush.columns:
    print(f"  {col}")

print("\nSample:")
print(pfr_rush.head(10))

# Key metrics
print("\nKey Rushing Metrics Available:")
rush_metrics = [col for col in pfr_rush.columns if 'rush_' in col or 'carry' in col]
for metric in rush_metrics[:10]:
    print(f"  • {metric}")

print("\nKey Insights:")
print(f"  • Player-level rushing advanced stats")
print(f"  • Yards before contact, yards after contact")
print(f"  • Can aggregate to team rushing quality")

# ============================================================================
# 6. PLAYER STATS (COMPREHENSIVE)
# ============================================================================

print("\n" + "="*80)
print("6. PLAYER STATS (COMPREHENSIVE)")
print("="*80)

player_stats = pd.read_parquet('/home/user/BK_Build/player_stats_week.parquet')
datasets['player_stats'] = player_stats

print(f"\nShape: {player_stats.shape}")
print(f"Seasons: {player_stats['season'].min()}-{player_stats['season'].max()}")
print(f"Total player-week stats: {len(player_stats):,}")

print("\nColumns (first 30):")
for col in player_stats.columns[:30]:
    print(f"  {col}")

print(f"\n... and {len(player_stats.columns) - 30} more columns")

print("\nSample:")
sample_cols = ['season', 'week', 'player_display_name', 'team',
               'position', 'completions', 'attempts', 'passing_yards',
               'rushing_yards', 'receiving_yards']
available_cols = [col for col in sample_cols if col in player_stats.columns]
print(player_stats[available_cols].head(10))

print("\nKey Insights:")
print(f"  • {len(player_stats.columns)} total stat columns")
print(f"  • Comprehensive offensive stats")
print(f"  • Can identify star players and aggregate to team")

# ============================================================================
# 7. SNAP COUNTS
# ============================================================================

print("\n" + "="*80)
print("7. SNAP COUNTS")
print("="*80)

snaps = pd.read_parquet('/home/user/BK_Build/snap_counts.parquet')
datasets['snaps'] = snaps

print(f"\nShape: {snaps.shape}")
print(f"Seasons: {snaps['season'].min()}-{snaps['season'].max()}")
print(f"Total player-game snap records: {len(snaps):,}")

print("\nKey Columns:")
for col in snaps.columns:
    print(f"  {col}")

print("\nSample:")
print(snaps[['season', 'week', 'player', 'team', 'position',
             'offense_snaps', 'offense_pct', 'defense_snaps',
             'defense_pct']].head(10))

# Position breakdown
if 'position' in snaps.columns:
    print("\nSnap Counts by Position:")
    print(snaps.groupby('position')[['offense_snaps', 'defense_snaps']].sum().head(10))

print("\nKey Insights:")
print(f"  • WHO is actually playing (starters vs backups)")
print(f"  • Can weight player stats by snap count")
print(f"  • Identify feature backs, rotational players")

# ============================================================================
# 8. TEAM STATS (COMPREHENSIVE)
# ============================================================================

print("\n" + "="*80)
print("8. TEAM STATS (COMPREHENSIVE)")
print("="*80)

team_stats = pd.read_parquet('/home/user/BK_Build/team_stats_week.parquet')
datasets['team_stats'] = team_stats

print(f"\nShape: {team_stats.shape}")
print(f"Seasons: {team_stats['season'].min()}-{team_stats['season'].max()}")
print(f"Total team-weeks: {len(team_stats):,}")

print("\nColumns (first 40):")
for col in team_stats.columns[:40]:
    print(f"  {col}")

print(f"\n... and {len(team_stats.columns) - 40} more columns")

print("\nSample Key Stats:")
sample_cols = ['season', 'week', 'team', 'attempts', 'passing_yards',
               'passing_tds', 'rushing_yards', 'rushing_tds']
available_cols = [col for col in sample_cols if col in team_stats.columns]
print(team_stats[available_cols].head(10))

print("\nKey Insights:")
print(f"  • {len(team_stats.columns)} comprehensive team stats")
print(f"  • Offense: yards, TDs, 3rd downs, red zone")
print(f"  • Defense: yards allowed, sacks, turnovers")
print(f"  • Special teams: FG%, punt/kick return")

# ============================================================================
# 9. ROSTERS (WEEKLY)
# ============================================================================

print("\n" + "="*80)
print("9. ROSTERS (WEEKLY)")
print("="*80)

rosters = pd.read_parquet('/home/user/BK_Build/rosters_weekly.parquet')
datasets['rosters'] = rosters

print(f"\nShape: {rosters.shape}")
print(f"Seasons: {rosters['season'].min()}-{rosters['season'].max()}")
print(f"Total player-week rosters: {len(rosters):,}")

print("\nKey Columns:")
for col in rosters.columns[:20]:
    print(f"  {col}")

print("\nSample:")
print(rosters[['season', 'week', 'team', 'position', 'full_name',
               'depth_chart_position', 'status']].head(10))

# Status breakdown
if 'status' in rosters.columns:
    print("\nPlayer Status:")
    print(rosters['status'].value_counts())

print("\nKey Insights:")
print(f"  • Weekly roster changes")
print(f"  • Depth chart positions")
print(f"  • Player status (active, IR, practice squad)")

# ============================================================================
# 10. DRAFT PICKS
# ============================================================================

print("\n" + "="*80)
print("10. DRAFT PICKS")
print("="*80)

draft = pd.read_parquet('/home/user/BK_Build/draft_picks.parquet')
datasets['draft'] = draft

print(f"\nShape: {draft.shape}")
print(f"Draft years: {draft['season'].min()}-{draft['season'].max()}")
print(f"Total picks: {len(draft):,}")

print("\nKey Columns:")
for col in draft.columns:
    print(f"  {col}")

print("\nSample:")
print(draft[['season', 'round', 'pick', 'team', 'pfr_player_name',
             'position']].head(10))

# Picks by position
if 'position' in draft.columns:
    print("\nDraft Picks by Position (Top 10):")
    print(draft['position'].value_counts().head(10))

print("\nKey Insights:")
print(f"  • Historical draft capital by team")
print(f"  • Can aggregate recent draft quality")
print(f"  • Young talent indicator")

# ============================================================================
# 11. TRADES
# ============================================================================

print("\n" + "="*80)
print("11. TRADES")
print("="*80)

trades = pd.read_parquet('/home/user/BK_Build/trades.parquet')
datasets['trades'] = trades

print(f"\nShape: {trades.shape}")
print(f"Seasons: {trades['season'].min()}-{trades['season'].max()}")
print(f"Total trades: {len(trades):,}")

print("\nKey Columns:")
for col in trades.columns:
    print(f"  {col}")

print("\nSample:")
print(trades.head(10))

# Trades by season
if 'season' in trades.columns:
    print("\nTrades by Season:")
    print(trades['season'].value_counts().sort_index().tail(10))

print("\nKey Insights:")
print(f"  • Mid-season roster changes")
print(f"  • Could flag weeks after major trades")
print(f"  • Team improvement/decline signal")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("DATASET SUMMARY & MODELING RECOMMENDATIONS")
print("="*80)

print("\n┌─────────────────────────┬──────────┬──────────┬─────────────┐")
print("│ Dataset                 │ Records  │ Seasons  │ Priority    │")
print("├─────────────────────────┼──────────┼──────────┼─────────────┤")

priorities = [
    ("ESPN QBR Week", len(qbr_week), f"{qbr_week['season'].min()}-{qbr_week['season'].max()}", "HIGH"),
    ("Injuries", len(injuries), f"{injuries['season'].min()}-{injuries['season'].max()}", "HIGH"),
    ("Team Stats Week", len(team_stats), f"{team_stats['season'].min()}-{team_stats['season'].max()}", "HIGH"),
    ("Snap Counts", len(snaps), f"{snaps['season'].min()}-{snaps['season'].max()}", "MEDIUM"),
    ("Officials", len(officials), "2006-2025", "LOW"),
    ("PFR Adv Receiving", len(pfr_rec), f"{pfr_rec['season'].min()}-{pfr_rec['season'].max()}", "MEDIUM"),
    ("PFR Adv Rushing", len(pfr_rush), f"{pfr_rush['season'].min()}-{pfr_rush['season'].max()}", "MEDIUM"),
    ("Player Stats", len(player_stats), f"{player_stats['season'].min()}-{player_stats['season'].max()}", "MEDIUM"),
    ("Rosters Weekly", len(rosters), f"{rosters['season'].min()}-{rosters['season'].max()}", "LOW"),
    ("Draft Picks", len(draft), f"{draft['season'].min()}-{draft['season'].max()}", "LOW"),
    ("Trades", len(trades), f"{trades['season'].min()}-{trades['season'].max()}", "LOW"),
]

for dataset, records, seasons, priority in priorities:
    print(f"│ {dataset:<23} │ {records:>8,} │ {seasons:>8} │ {priority:>11} │")

print("└─────────────────────────┴──────────┴──────────┴─────────────┘")

print("\n" + "="*80)
print("RECOMMENDED MODEL ITERATIONS")
print("="*80)

print("""
Current: v1.4 (1.42 MAE) with EPA + NGS
         v2.0 (poor) with matchup features
         v2.1 (not tested) hybrid

Next Iterations to Build:

v2.2: + ESPN QBR (weekly QB performance)
      Features: QBR total, points added, play action stats
      Expected impact: HIGH (QB is most important position)

v2.3: + Injury Impact (key position injuries)
      Features: QB out, top WR out, OL injuries, CB injuries
      Expected impact: HIGH (major injuries matter)

v2.4: + Team Stats (comprehensive offensive/defensive stats)
      Features: 3rd down %, red zone %, turnover differential
      Expected impact: MEDIUM (overlaps with EPA)

v2.5: + Snap Count Weighted Stats
      Features: Weight player stats by snap %
      Expected impact: MEDIUM (better than simple aggregation)

v2.6: + Situational Stats (from team_stats)
      Features: Goal-to-go %, short yardage %, 2-minute offense
      Expected impact: MEDIUM (game script insights)

v2.7: + Referee Tendencies
      Features: Home bias, penalty rate, over/under bias
      Expected impact: LOW (small edge if any)

v2.8: ENSEMBLE MODEL
      Combine best features from v2.2-v2.7
      Expected impact: Highest overall

Testing Protocol:
1. Build each model sequentially
2. Track MAE improvement over v1.4 (1.42 baseline)
3. Feature importance analysis for each
4. Keep features that improve by >1%
5. Final ensemble with best features only
""")

print("="*80 + "\n")

# Save summary
summary = pd.DataFrame(priorities, columns=['Dataset', 'Records', 'Seasons', 'Priority'])
summary.to_csv('/home/user/BK_Build/dataset_exploration_summary.csv', index=False)

print("✓ Summary saved to: dataset_exploration_summary.csv\n")
