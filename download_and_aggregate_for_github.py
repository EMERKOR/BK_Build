"""
Download and aggregate play-by-play data - Upload small aggregated file

GitHub has 25 MB file size limit, so we aggregate BEFORE uploading.
This creates a tiny ~100 KB CSV file that's perfect for GitHub.

Run this on your local machine to:
1. Download NFLverse play-by-play (2009-2024)
2. Aggregate to team-week EPA stats
3. Create small CSV file for GitHub upload (~100 KB)

Requirements:
    pip install nfl_data_py pandas

Usage:
    python download_and_aggregate_for_github.py

Output:
    data/team_week_epa_2009_2024.csv (~100 KB - ready to commit)
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("DOWNLOAD & AGGREGATE FOR GITHUB UPLOAD")
print("="*80)

# Check dependencies
try:
    import nfl_data_py as nfl
    import pandas as pd
    print("\n✓ Required libraries installed")
except ImportError as e:
    print(f"\n✗ Missing library: {e}")
    print("\nInstall with: pip install nfl_data_py pandas")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

START_YEAR = 2009
END_YEAR = 2024
OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f'team_week_epa_{START_YEAR}_{END_YEAR}.csv'

# ============================================================================
# DOWNLOAD
# ============================================================================

print(f"\n[1/2] Downloading play-by-play data ({START_YEAR}-{END_YEAR})...")
print("This will take a few minutes...")

try:
    seasons = list(range(START_YEAR, END_YEAR + 1))
    pbp = nfl.import_pbp_data(seasons)

    print(f"  ✓ Downloaded {len(pbp):,} plays")
    print(f"  Seasons: {pbp['season'].min()}-{pbp['season'].max()}")

    # Check for EPA columns
    if 'epa' not in pbp.columns or 'success' not in pbp.columns:
        print(f"  ✗ ERROR: Missing EPA or success columns")
        print(f"  Available columns: {pbp.columns.tolist()[:20]}")
        sys.exit(1)

    print(f"  ✓ EPA and success columns present")

except Exception as e:
    print(f"\n✗ ERROR downloading data: {e}")
    print("\nTroubleshooting:")
    print("  - Check internet connection")
    print("  - Update nfl_data_py: pip install -U nfl_data_py")
    sys.exit(1)

# ============================================================================
# AGGREGATE
# ============================================================================

print(f"\n[2/2] Aggregating to team-week stats...")

def aggregate_team_week_stats(plays_df):
    """Aggregate play-by-play to team-week level"""

    # Get all teams
    all_teams = set()
    all_teams.update(plays_df['posteam'].dropna().unique())
    all_teams.update(plays_df['defteam'].dropna().unique())
    all_teams = sorted(all_teams)

    print(f"  Processing {len(all_teams)} teams...")

    all_stats = []
    total_seasons = plays_df['season'].nunique()
    processed_seasons = 0

    for season in sorted(plays_df['season'].unique()):
        season_data = plays_df[plays_df['season'] == season]
        weeks = sorted(season_data['week'].dropna().unique())

        for week in weeks:
            week_data = season_data[season_data['week'] == week]

            for team in all_teams:
                # Offensive plays
                off_plays = week_data[
                    (week_data['posteam'] == team) &
                    (week_data['play_type'].isin(['pass', 'run']))
                ]

                # Defensive plays
                def_plays = week_data[
                    (week_data['defteam'] == team) &
                    (week_data['play_type'].isin(['pass', 'run']))
                ]

                if len(off_plays) > 0 and len(def_plays) > 0:
                    stats = {
                        'season': int(season),
                        'week': int(week),
                        'team': team,

                        # Offensive EPA
                        'off_plays': len(off_plays),
                        'off_epa_total': float(off_plays['epa'].sum()),
                        'off_epa_per_play': float(off_plays['epa'].mean()),
                        'off_success_rate': float(off_plays['success'].mean()),
                        'off_explosive_rate': float((off_plays['epa'] > 0.5).mean()),

                        # Pass vs Run
                        'off_pass_epa': float(off_plays[off_plays['play_type'] == 'pass']['epa'].mean()),
                        'off_run_epa': float(off_plays[off_plays['play_type'] == 'run']['epa'].mean()),

                        # Defensive EPA (allowed to opponent)
                        'def_plays': len(def_plays),
                        'def_epa_allowed_total': float(def_plays['epa'].sum()),
                        'def_epa_allowed_per_play': float(def_plays['epa'].mean()),
                        'def_success_allowed_rate': float(def_plays['success'].mean()),
                        'def_explosive_allowed_rate': float((def_plays['epa'] > 0.5).mean()),

                        'def_pass_epa_allowed': float(def_plays[def_plays['play_type'] == 'pass']['epa'].mean()),
                        'def_run_epa_allowed': float(def_plays[def_plays['play_type'] == 'run']['epa'].mean()),
                    }

                    all_stats.append(stats)

        processed_seasons += 1
        print(f"    Completed season {season} ({processed_seasons}/{total_seasons})")

    return pd.DataFrame(all_stats)

try:
    team_week_df = aggregate_team_week_stats(pbp)

    print(f"\n  ✓ Generated {len(team_week_df):,} team-week records")

    # Sort
    team_week_df = team_week_df.sort_values(['season', 'week', 'team'])

    # Save
    team_week_df.to_csv(OUTPUT_FILE, index=False)

    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"  ✓ Saved to: {OUTPUT_FILE}")
    print(f"  File size: {file_size_kb:.1f} KB")

except Exception as e:
    print(f"\n✗ ERROR during aggregation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUCCESS!")
print("="*80)

print(f"\nData created:")
print(f"  File: {OUTPUT_FILE}")
print(f"  Size: {file_size_kb:.1f} KB (well under GitHub's 25 MB limit!)")
print(f"  Records: {len(team_week_df):,}")
print(f"  Seasons: {team_week_df['season'].min()}-{team_week_df['season'].max()}")
print(f"  Teams: {team_week_df['team'].nunique()}")

print(f"\nSample data (2023, Week 1):")
sample = team_week_df[(team_week_df['season'] == 2023) & (team_week_df['week'] == 1)]
if len(sample) > 0:
    display_cols = ['team', 'off_epa_per_play', 'def_epa_allowed_per_play', 'off_success_rate']
    print(sample[display_cols].head(5).round(3).to_string(index=False))

print("\n" + "="*80)
print("UPLOAD TO GITHUB")
print("="*80)

print(f"""
The aggregated file is ready! Upload to GitHub:

Method 1: Via Git Command Line
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd /path/to/BK_Build
  cp {OUTPUT_FILE} data/
  git add data/{OUTPUT_FILE.name}
  git commit -m "Add team-week EPA data ({START_YEAR}-{END_YEAR})"
  git push

Method 2: Via GitHub Web Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Go to: https://github.com/EMERKOR/BK_Build
  2. Navigate to data/ folder
  3. Click "Add file" → "Upload files"
  4. Drag {OUTPUT_FILE.name}
  5. Commit with message: "Add team-week EPA data"

Then let Claude know:
  "I've uploaded the EPA data file"
""")

print("\n" + "="*80)
print("WHAT CLAUDE WILL DO NEXT")
print("="*80)

print("""
Once you upload the file, Claude will:

1. Build v1.3 model with EPA features:
   - EPA per play differential (offense - defense)
   - Success rate differential
   - Explosive play rate differential
   - Pass vs run efficiency splits
   - Rolling averages for recent form
   - Merge with existing nfelo features

2. Train and evaluate:
   - Same train/test split as v1.2 (2025 holdout)
   - Compare MAE, R², feature importance
   - Measure improvement vs v1.2 baseline

3. Run comprehensive backtest:
   - CLV analysis with EPA features
   - EV and Kelly sizing
   - Historical performance tracking

Expected Results:
  - Better prediction accuracy (15-25% MAE reduction)
  - Stronger feature importance for true game efficiency
  - Professional-grade model matching research standards
  - All metrics from your Gemini research implemented
""")

print("="*80 + "\n")
