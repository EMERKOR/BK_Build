"""
Download NFLverse Play-by-Play Data (Run on Local Machine)

This script should be run on your local machine where network restrictions
don't apply. It will download and aggregate EPA data for you to commit.

Requirements:
    pip install nfl_data_py pandas pyarrow

Usage:
    python download_pbp_locally.py

What it does:
    1. Downloads play-by-play data from NFLverse (2009-2024)
    2. Saves raw data to local parquet file (backup)
    3. Aggregates to team-week stats
    4. Saves small aggregated CSV for Git commit
"""

import sys

print("\n" + "="*80)
print("NFLVERSE DATA DOWNLOAD (Local Machine)")
print("="*80)

# Check if running with required libraries
try:
    import nfl_data_py as nfl
    import pandas as pd
    print("\n✓ Required libraries installed")
except ImportError as e:
    print(f"\n✗ Missing required library: {e}")
    print("\nPlease install:")
    print("  pip install nfl_data_py pandas pyarrow")
    sys.exit(1)

from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

START_YEAR = 2009
END_YEAR = 2024
OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_FILE = OUTPUT_DIR / f'pbp_{START_YEAR}_{END_YEAR}_raw.parquet'
AGGREGATED_FILE = OUTPUT_DIR / f'team_week_epa_{START_YEAR}_{END_YEAR}.csv'

# ============================================================================
# DOWNLOAD
# ============================================================================

print(f"\n[1/3] Downloading play-by-play data ({START_YEAR}-{END_YEAR})...")
print("This may take a few minutes depending on your internet speed...")

try:
    seasons = list(range(START_YEAR, END_YEAR + 1))
    pbp = nfl.import_pbp_data(seasons)

    print(f"  ✓ Downloaded {len(pbp):,} plays")
    print(f"  Columns: {len(pbp.columns)}")

    # Check for EPA columns
    epa_cols = [col for col in pbp.columns if 'epa' in col.lower()]
    print(f"  EPA-related columns: {len(epa_cols)}")

    # Save raw backup
    print(f"\n[2/3] Saving raw data to {RAW_FILE}...")
    pbp.to_parquet(RAW_FILE)
    file_size_mb = RAW_FILE.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved {file_size_mb:.1f} MB")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nTroubleshooting:")
    print("  - Check internet connection")
    print("  - Verify nfl_data_py is up to date: pip install -U nfl_data_py")
    print("  - Try reducing year range if download times out")
    sys.exit(1)

# ============================================================================
# AGGREGATE
# ============================================================================

print(f"\n[3/3] Aggregating to team-week stats...")

def aggregate_team_offensive(plays, team, season, week):
    """Calculate offensive stats for team in a week"""
    team_plays = plays[
        (plays['posteam'] == team) &
        (plays['season'] == season) &
        (plays['week'] == week) &
        (plays['play_type'].isin(['pass', 'run']))
    ]

    if len(team_plays) == 0:
        return None

    return {
        'season': season,
        'week': week,
        'team': team,
        'off_plays': len(team_plays),
        'off_epa_total': team_plays['epa'].sum(),
        'off_epa_per_play': team_plays['epa'].mean(),
        'off_success_rate': team_plays['success'].mean(),
        'off_explosive_rate': (team_plays['epa'] > 0.5).mean(),
        'off_pass_epa': team_plays[team_plays['play_type'] == 'pass']['epa'].mean(),
        'off_run_epa': team_plays[team_plays['play_type'] == 'run']['epa'].mean(),
    }

def aggregate_team_defensive(plays, team, season, week):
    """Calculate defensive stats (EPA allowed) for team in a week"""
    team_plays = plays[
        (plays['defteam'] == team) &
        (plays['season'] == season) &
        (plays['week'] == week) &
        (plays['play_type'].isin(['pass', 'run']))
    ]

    if len(team_plays) == 0:
        return None

    return {
        'def_plays': len(team_plays),
        'def_epa_allowed_total': team_plays['epa'].sum(),
        'def_epa_allowed_per_play': team_plays['epa'].mean(),
        'def_success_allowed_rate': team_plays['success'].mean(),
        'def_explosive_allowed_rate': (team_plays['epa'] > 0.5).mean(),
        'def_pass_epa_allowed': team_plays[team_plays['play_type'] == 'pass']['epa'].mean(),
        'def_run_epa_allowed': team_plays[team_plays['play_type'] == 'run']['epa'].mean(),
    }

# Get all teams
all_teams = set()
all_teams.update(pbp['posteam'].dropna().unique())
all_teams.update(pbp['defteam'].dropna().unique())
all_teams = sorted(all_teams)

print(f"  Processing {len(all_teams)} teams across {END_YEAR - START_YEAR + 1} seasons...")

all_stats = []
total = len(all_teams) * (END_YEAR - START_YEAR + 1) * 18  # Approx 18 weeks per season
processed = 0

for season in range(START_YEAR, END_YEAR + 1):
    for week in range(1, 23):  # Regular season + playoffs
        for team in all_teams:
            off_stats = aggregate_team_offensive(pbp, team, season, week)
            def_stats = aggregate_team_defensive(pbp, team, season, week)

            if off_stats and def_stats:
                combined = {**off_stats, **def_stats}
                all_stats.append(combined)

            processed += 1
            if processed % 1000 == 0:
                pct = processed / total * 100
                print(f"    Progress: {pct:.1f}%", end='\r')

print(f"\n  ✓ Generated {len(all_stats):,} team-week records")

team_week_df = pd.DataFrame(all_stats)
team_week_df = team_week_df.sort_values(['season', 'week', 'team'])

# Save aggregated
team_week_df.to_csv(AGGREGATED_FILE, index=False)
agg_size_kb = AGGREGATED_FILE.stat().st_size / 1024

print(f"  ✓ Saved to {AGGREGATED_FILE} ({agg_size_kb:.1f} KB)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DOWNLOAD COMPLETE!")
print("="*80)

print(f"\nFiles created:")
print(f"  1. {RAW_FILE} ({file_size_mb:.1f} MB) - Raw play-by-play backup")
print(f"  2. {AGGREGATED_FILE} ({agg_size_kb:.1f} KB) - Team-week aggregates for Git")

print(f"\nData summary:")
print(f"  Seasons: {team_week_df['season'].min()}-{team_week_df['season'].max()}")
print(f"  Teams: {team_week_df['team'].nunique()}")
print(f"  Records: {len(team_week_df):,}")

print(f"\nSample (2023, Week 1):")
sample = team_week_df[(team_week_df['season'] == 2023) & (team_week_df['week'] == 1)]
print(sample[['team', 'off_epa_per_play', 'def_epa_allowed_per_play']].head(5).to_string(index=False))

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print(f"""
1. Transfer {AGGREGATED_FILE} to your development environment

2. Commit to Git:
   git add {AGGREGATED_FILE}
   git commit -m "Add historical team-week EPA data ({START_YEAR}-{END_YEAR})"
   git push

3. The model can now use this data for v1.3 training!

4. For weekly updates:
   - Re-run this script (it will fetch latest data)
   - Or just download current week and append

Optional:
- Keep {RAW_FILE} as backup
- Add to .gitignore (too large for Git)
- Re-aggregate anytime with: python aggregate_pbp_to_team_stats.py {RAW_FILE}
""")

print("="*80 + "\n")
