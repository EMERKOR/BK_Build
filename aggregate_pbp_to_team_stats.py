"""
Aggregate play-by-play data into team-week EPA statistics

Processes full NFLverse play-by-play data into compact team-week aggregates
suitable for Git storage and model training.

Usage:
    python aggregate_pbp_to_team_stats.py <input_file.parquet> [--append]

Examples:
    # Process full historical data
    python aggregate_pbp_to_team_stats.py pbp_2009_2024.parquet

    # Append new week to existing file
    python aggregate_pbp_to_team_stats.py pbp_week12.parquet --append

Output:
    data/team_week_epa_2009_2024.csv (~100KB for full history)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_FILE = Path('data/team_week_epa_2009_2024.csv')

# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_team_week_offensive(plays_df, team, season, week):
    """Aggregate offensive stats for a team in a specific week"""

    # Filter to this team's offensive plays
    team_plays = plays_df[
        (plays_df['posteam'] == team) &
        (plays_df['season'] == season) &
        (plays_df['week'] == week) &
        (plays_df['play_type'].isin(['pass', 'run']))  # Exclude special teams
    ].copy()

    if len(team_plays) == 0:
        return None

    # Calculate metrics
    stats = {
        'season': season,
        'week': week,
        'team': team,

        # Overall offensive efficiency
        'off_plays': len(team_plays),
        'off_epa_total': team_plays['epa'].sum(),
        'off_epa_per_play': team_plays['epa'].mean(),
        'off_success_rate': team_plays['success'].mean(),
        'off_explosive_rate': (team_plays['epa'] > 0.5).mean(),

        # Pass vs run
        'off_pass_plays': (team_plays['play_type'] == 'pass').sum(),
        'off_pass_epa': team_plays[team_plays['play_type'] == 'pass']['epa'].mean(),
        'off_run_plays': (team_plays['play_type'] == 'run').sum(),
        'off_run_epa': team_plays[team_plays['play_type'] == 'run']['epa'].mean(),

        # Situational
        'off_third_down_plays': (team_plays['down'] == 3).sum(),
        'off_third_down_conv_rate': team_plays[team_plays['down'] == 3]['third_down_converted'].mean()
                                     if (team_plays['down'] == 3).any() else np.nan,

        'off_redzone_plays': (team_plays['yardline_100'] <= 20).sum(),
        'off_redzone_td_rate': (
            team_plays[(team_plays['yardline_100'] <= 20)]['touchdown'].mean()
            if (team_plays['yardline_100'] <= 20).any() else np.nan
        ),

        # Big plays
        'off_plays_20plus_yards': (team_plays['yards_gained'] >= 20).sum(),
        'off_plays_20plus_rate': (team_plays['yards_gained'] >= 20).mean(),

        # Turnovers
        'off_interceptions': (team_plays['interception'] == 1).sum(),
        'off_fumbles_lost': (team_plays['fumble_lost'] == 1).sum(),
        'off_turnover_rate': ((team_plays['interception'] == 1) | (team_plays['fumble_lost'] == 1)).mean(),
    }

    return stats


def aggregate_team_week_defensive(plays_df, team, season, week):
    """Aggregate defensive stats (EPA allowed) for a team in a specific week"""

    # Filter to plays where this team is on defense
    team_plays = plays_df[
        (plays_df['defteam'] == team) &
        (plays_df['season'] == season) &
        (plays_df['week'] == week) &
        (plays_df['play_type'].isin(['pass', 'run']))
    ].copy()

    if len(team_plays) == 0:
        return None

    stats = {
        # Defensive efficiency (EPA allowed to opponent)
        'def_plays': len(team_plays),
        'def_epa_allowed_total': team_plays['epa'].sum(),
        'def_epa_allowed_per_play': team_plays['epa'].mean(),
        'def_success_allowed_rate': team_plays['success'].mean(),
        'def_explosive_allowed_rate': (team_plays['epa'] > 0.5).mean(),

        # Pass vs run defense
        'def_pass_plays': (team_plays['play_type'] == 'pass').sum(),
        'def_pass_epa_allowed': team_plays[team_plays['play_type'] == 'pass']['epa'].mean(),
        'def_run_plays': (team_plays['play_type'] == 'run').sum(),
        'def_run_epa_allowed': team_plays[team_plays['play_type'] == 'run']['epa'].mean(),

        # Situational
        'def_third_down_plays': (team_plays['down'] == 3).sum(),
        'def_third_down_conv_allowed_rate': team_plays[team_plays['down'] == 3]['third_down_converted'].mean()
                                              if (team_plays['down'] == 3).any() else np.nan,

        'def_redzone_plays': (team_plays['yardline_100'] <= 20).sum(),
        'def_redzone_td_allowed_rate': (
            team_plays[(team_plays['yardline_100'] <= 20)]['touchdown'].mean()
            if (team_plays['yardline_100'] <= 20).any() else np.nan
        ),

        # Big plays allowed
        'def_plays_20plus_allowed': (team_plays['yards_gained'] >= 20).sum(),
        'def_plays_20plus_allowed_rate': (team_plays['yards_gained'] >= 20).mean(),

        # Turnovers forced
        'def_interceptions_forced': (team_plays['interception'] == 1).sum(),
        'def_fumbles_forced': (team_plays['fumble_lost'] == 1).sum(),
        'def_turnover_forced_rate': ((team_plays['interception'] == 1) | (team_plays['fumble_lost'] == 1)).mean(),
    }

    return stats


def aggregate_pbp_to_team_week(pbp_df):
    """Main aggregation function - processes full PBP into team-week stats"""

    print(f"Processing {len(pbp_df):,} plays...")

    # Get unique combinations of season, week, team
    teams = []
    for col in ['posteam', 'defteam']:
        teams.extend(pbp_df[col].dropna().unique())
    teams = sorted(set(teams))

    seasons = sorted(pbp_df['season'].dropna().unique())
    weeks = sorted(pbp_df['week'].dropna().unique())

    print(f"  Seasons: {len(seasons)} ({seasons[0]}-{seasons[-1]})")
    print(f"  Weeks: {len(weeks)}")
    print(f"  Teams: {len(teams)}")

    all_stats = []

    # Process each team-week combination
    total_combinations = len(teams) * len(seasons) * len(weeks)
    processed = 0

    for season in seasons:
        for week in weeks:
            for team in teams:
                # Get offensive stats
                off_stats = aggregate_team_week_offensive(pbp_df, team, season, week)

                # Get defensive stats
                def_stats = aggregate_team_week_defensive(pbp_df, team, season, week)

                # Merge if both exist
                if off_stats and def_stats:
                    combined = {**off_stats, **def_stats}
                    all_stats.append(combined)

                processed += 1
                if processed % 1000 == 0:
                    print(f"  Processed {processed:,}/{total_combinations:,} combinations...")

    print(f"  ✓ Generated {len(all_stats):,} team-week records")

    return pd.DataFrame(all_stats)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Aggregate PBP data to team-week stats')
    parser.add_argument('input_file', help='Path to play-by-play parquet file')
    parser.add_argument('--append', action='store_true', help='Append to existing data')
    parser.add_argument('--output', default=str(OUTPUT_FILE), help='Output file path')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PLAY-BY-PLAY TO TEAM-WEEK AGGREGATION")
    print("="*80)

    # Load input
    print(f"\nLoading {args.input_file}...")
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    if input_path.suffix == '.parquet':
        pbp = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        pbp = pd.read_csv(input_path)
    else:
        print(f"ERROR: Unsupported file type: {input_path.suffix}")
        print("Supported: .parquet, .csv")
        return 1

    print(f"  ✓ Loaded {len(pbp):,} plays")

    # Check required columns
    required_cols = ['season', 'week', 'posteam', 'defteam', 'play_type', 'epa', 'success']
    missing = [col for col in required_cols if col not in pbp.columns]

    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {pbp.columns.tolist()[:20]}")
        return 1

    # Aggregate
    team_week_stats = aggregate_pbp_to_team_week(pbp)

    # Handle append mode
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if args.append and output_path.exists():
        print(f"\nAppending to existing data: {output_path}")
        existing = pd.read_csv(output_path)
        print(f"  Existing records: {len(existing):,}")

        # Combine and deduplicate
        combined = pd.concat([existing, team_week_stats], ignore_index=True)
        combined = combined.drop_duplicates(subset=['season', 'week', 'team'], keep='last')

        print(f"  Combined records: {len(combined):,}")
        team_week_stats = combined

    # Sort
    team_week_stats = team_week_stats.sort_values(['season', 'week', 'team'])

    # Save
    print(f"\nSaving to {output_path}...")
    team_week_stats.to_csv(output_path, index=False)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  ✓ Saved {len(team_week_stats):,} records ({file_size_kb:.1f} KB)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nData coverage:")
    print(f"  Seasons: {team_week_stats['season'].min()}-{team_week_stats['season'].max()}")
    print(f"  Weeks: {team_week_stats['week'].min()}-{team_week_stats['week'].max()}")
    print(f"  Teams: {team_week_stats['team'].nunique()}")
    print(f"  Total records: {len(team_week_stats):,}")

    print(f"\nSample data (Week 1, 2023):")
    sample = team_week_stats[
        (team_week_stats['season'] == 2023) & (team_week_stats['week'] == 1)
    ].head(3)

    display_cols = ['team', 'off_epa_per_play', 'def_epa_allowed_per_play', 'off_success_rate']
    print(sample[display_cols].to_string(index=False))

    print(f"\nFile: {output_path}")
    print(f"Size: {file_size_kb:.1f} KB")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    print(f"""
1. Review the output file: {output_path}
2. Commit to Git:
   git add {output_path}
   git commit -m "Add team-week EPA data"
   git push

3. Use in model training:
   - Load with: pd.read_csv('{output_path}')
   - Merge with nfelo data on (season, week, team)
   - Build rolling averages for recent form
   - Train v1.3 model with EPA features

4. Weekly updates:
   python aggregate_pbp_to_team_stats.py new_week.parquet --append
""")

    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
