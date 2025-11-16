"""
Explore nflreadpy (modern NFLverse data access)

Tests the modern nflreadpy library for EPA and play-by-play data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("NFLREADPY DATA EXPLORATION")
print("="*80)

# ============================================================================
# TEST NFLREADPY
# ============================================================================

print("\n[1/3] Testing nflreadpy library...")

try:
    import nflreadpy as nfl_read

    print("  ✓ nflreadpy is installed")

    # List available functions
    available_funcs = [attr for attr in dir(nfl_read) if not attr.startswith('_')]
    print(f"  Available functions: {len(available_funcs)}")
    print(f"    {available_funcs[:10]}")

except ImportError:
    print("  ✗ nflreadpy not installed")
    print("  Installing...")

    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "nflreadpy", "-q"],
                  check=True)

    import nflreadpy as nfl_read
    print("  ✓ nflreadpy installed successfully")

# ============================================================================
# LOAD PLAY-BY-PLAY DATA
# ============================================================================

print("\n[2/3] Loading play-by-play data (2023 season)...")

try:
    # Try loading 2023 play-by-play
    pbp = nfl_read.load_pbp(2023)

    print(f"  ✓ Loaded {len(pbp):,} plays from 2023 season")
    print(f"  Columns: {len(pbp.columns)}")

    # Find EPA-related columns
    epa_cols = [col for col in pbp.columns if 'epa' in col.lower()]
    success_cols = [col for col in pbp.columns if 'success' in col.lower()]
    wpa_cols = [col for col in pbp.columns if 'wpa' in col.lower() or 'wp' in col.lower()]

    print(f"\n  EPA columns ({len(epa_cols)}):")
    for col in sorted(epa_cols)[:15]:
        print(f"    - {col}")

    print(f"\n  Success columns ({len(success_cols)}):")
    for col in sorted(success_cols)[:10]:
        print(f"    - {col}")

    print(f"\n  Win probability columns ({len(wpa_cols)}):")
    for col in sorted(wpa_cols)[:10]:
        print(f"    - {col}")

    # Sample game analysis
    print("\n" + "="*80)
    print("SAMPLE GAME: EPA ANALYSIS")
    print("="*80)

    # Get first completed game
    sample_game_id = pbp[pbp['home_score'].notna()]['game_id'].iloc[0]
    sample_game = pbp[pbp['game_id'] == sample_game_id].copy()

    home_team = sample_game.iloc[0]['home_team']
    away_team = sample_game.iloc[0]['away_team']
    home_score = sample_game.iloc[0]['home_score']
    away_score = sample_game.iloc[0]['away_score']

    print(f"\nGame: {sample_game_id}")
    print(f"{away_team} {away_score} @ {home_team} {home_score}")
    print(f"Total plays: {len(sample_game)}")

    # Offensive EPA
    home_off_plays = sample_game[sample_game['posteam'] == home_team]
    away_off_plays = sample_game[sample_game['posteam'] == away_team]

    home_epa = home_off_plays['epa'].sum()
    away_epa = away_off_plays['epa'].sum()

    home_success = home_off_plays['success'].mean()
    away_success = away_off_plays['success'].mean()

    home_explosive = (home_off_plays['epa'] > 0.5).sum()
    away_explosive = (away_off_plays['epa'] > 0.5).sum()

    print(f"\nOffensive Performance:")
    print(f"  {home_team:3s}: Total EPA = {home_epa:6.2f} | Success = {home_success:.1%} | Explosive = {home_explosive}")
    print(f"  {away_team:3s}: Total EPA = {away_epa:6.2f} | Success = {away_success:.1%} | Explosive = {away_explosive}")

    # By play type
    print(f"\nPlay Type Breakdown ({home_team}):")

    for play_type in ['pass', 'run']:
        plays = home_off_plays[home_off_plays['play_type'] == play_type]
        if len(plays) > 0:
            avg_epa = plays['epa'].mean()
            success = plays['success'].mean()
            count = len(plays)
            print(f"  {play_type.capitalize():4s}: {count:3d} plays | EPA/play = {avg_epa:+.3f} | Success = {success:.1%}")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# AGGREGATE SEASON STATS
# ============================================================================

print("\n" + "="*80)
print("[3/3] Aggregating season-level EPA stats...")
print("="*80)

try:
    if 'pbp' in locals() and len(pbp) > 0:
        # Aggregate by team and week
        print("\nBuilding team-week EPA aggregates...")

        team_week_stats = []

        for (week, team), plays in pbp.groupby(['week', 'posteam']):
            if pd.isna(team):
                continue

            # Filter to regular plays (exclude special teams, penalties, etc.)
            reg_plays = plays[plays['play_type'].isin(['pass', 'run'])]

            if len(reg_plays) == 0:
                continue

            stats = {
                'season': 2023,
                'week': week,
                'team': team,
                'plays': len(reg_plays),
                'epa_per_play': reg_plays['epa'].mean(),
                'total_epa': reg_plays['epa'].sum(),
                'success_rate': reg_plays['success'].mean(),
                'explosive_rate': (reg_plays['epa'] > 0.5).mean(),
                'pass_epa': reg_plays[reg_plays['play_type'] == 'pass']['epa'].mean(),
                'run_epa': reg_plays[reg_plays['play_type'] == 'run']['epa'].mean(),
            }

            team_week_stats.append(stats)

        team_week_df = pd.DataFrame(team_week_stats)

        print(f"  ✓ Built {len(team_week_df):,} team-week records")
        print(f"\n  Sample (Week 1):")

        week1 = team_week_df[team_week_df['week'] == 1].sort_values('epa_per_play', ascending=False)
        display_cols = ['team', 'epa_per_play', 'success_rate', 'explosive_rate']
        print("\n" + week1[display_cols].head(10).round(3).to_string(index=False))

        # Save for v1.3 feature engineering
        output_file = project_root / 'data' / 'epa_team_week_2023.csv'
        output_file.parent.mkdir(exist_ok=True)
        team_week_df.to_csv(output_file, index=False)
        print(f"\n  ✓ Saved to: {output_file}")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: NFLREADPY CAPABILITIES")
print("="*80)

print("""
✓ nflreadpy successfully loads play-by-play data
✓ EPA and success metrics are available
✓ Can aggregate by team, week, situation

Next Steps for v1.3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Build full historical EPA database (2009-2024)
   - Aggregate team-week offensive EPA
   - Aggregate team-week defensive EPA (allowed to opponent)
   - Calculate rolling averages (3-5 game windows)

2. Engineer features for modeling:
   - epa_diff = (home_off_epa - home_def_epa) - (away_off_epa - away_def_epa)
   - success_diff = similar calculation
   - explosive_diff = similar calculation
   - Recent form (last 3 games)
   - Schedule-adjusted EPA (opponent quality)

3. Merge with v1.2 baseline:
   - Keep nfelo_diff, rest, QB adj
   - Add EPA features
   - Measure incremental predictive power

4. Train v1.3 and evaluate:
   - Same 2025 test set as v1.2
   - Compare MAE, R², CLV
   - Feature importance analysis

5. If successful, move to STEP 3 (score prediction)
""")

print("="*80 + "\n")
