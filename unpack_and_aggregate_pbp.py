"""
Unpack uploaded play-by-play zip and aggregate to team-week stats

This runs in the Claude Code environment after you upload the zip file.

Usage:
    python unpack_and_aggregate_pbp.py data/pbp_2009_2024.zip

If split into chunks:
    python unpack_and_aggregate_pbp.py data/pbp_2009_2024.zip.001

Output:
    data/team_week_epa_2009_2024.csv
"""

import sys
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_FILE = Path('data/team_week_epa_2009_2024.csv')
TEMP_DIR = Path('data/temp_pbp')

# ============================================================================
# REASSEMBLE CHUNKS (if split)
# ============================================================================

def reassemble_chunks(first_chunk_path):
    """Reassemble split zip file chunks"""

    print("Detected split zip file. Reassembling chunks...")

    base_path = str(first_chunk_path).rsplit('.', 1)[0]  # Remove .001
    chunk_num = 1
    chunks = []

    # Find all chunks
    while True:
        chunk_path = Path(f"{base_path}.{chunk_num:03d}")
        if not chunk_path.exists():
            break
        chunks.append(chunk_path)
        print(f"  Found: {chunk_path.name}")
        chunk_num += 1

    if len(chunks) < 2:
        return first_chunk_path  # Not actually split

    # Reassemble
    reassembled = Path(f"{base_path}.zip")
    print(f"\n  Reassembling into: {reassembled.name}...")

    with open(reassembled, 'wb') as outfile:
        for chunk in chunks:
            with open(chunk, 'rb') as infile:
                outfile.write(infile.read())

    size_mb = reassembled.stat().st_size / (1024 * 1024)
    print(f"  ✓ Reassembled: {size_mb:.1f} MB")

    return reassembled

# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_team_week(pbp_df):
    """Aggregate play-by-play to team-week statistics"""

    print(f"\nAggregating {len(pbp_df):,} plays to team-week level...")

    # Get unique teams
    all_teams = set()
    all_teams.update(pbp_df['posteam'].dropna().unique())
    all_teams.update(pbp_df['defteam'].dropna().unique())
    all_teams = sorted(all_teams)

    seasons = sorted(pbp_df['season'].dropna().unique())
    print(f"  Seasons: {seasons[0]}-{seasons[-1]}")
    print(f"  Teams: {len(all_teams)}")

    all_stats = []
    total = len(all_teams) * len(seasons) * 22  # Approx weeks
    processed = 0

    for season in seasons:
        season_data = pbp_df[pbp_df['season'] == season]
        weeks = sorted(season_data['week'].dropna().unique())

        for week in weeks:
            week_data = season_data[season_data['week'] == week]

            for team in all_teams:
                # Offensive stats
                off_plays = week_data[
                    (week_data['posteam'] == team) &
                    (week_data['play_type'].isin(['pass', 'run']))
                ]

                # Defensive stats
                def_plays = week_data[
                    (week_data['defteam'] == team) &
                    (week_data['play_type'].isin(['pass', 'run']))
                ]

                if len(off_plays) > 0 and len(def_plays) > 0:
                    stats = {
                        'season': season,
                        'week': week,
                        'team': team,

                        # Offensive
                        'off_plays': len(off_plays),
                        'off_epa_total': off_plays['epa'].sum(),
                        'off_epa_per_play': off_plays['epa'].mean(),
                        'off_success_rate': off_plays['success'].mean(),
                        'off_explosive_rate': (off_plays['epa'] > 0.5).mean(),

                        # Pass/Run splits
                        'off_pass_epa': off_plays[off_plays['play_type'] == 'pass']['epa'].mean(),
                        'off_run_epa': off_plays[off_plays['play_type'] == 'run']['epa'].mean(),

                        # Defensive (EPA allowed)
                        'def_plays': len(def_plays),
                        'def_epa_allowed_total': def_plays['epa'].sum(),
                        'def_epa_allowed_per_play': def_plays['epa'].mean(),
                        'def_success_allowed_rate': def_plays['success'].mean(),
                        'def_explosive_allowed_rate': (def_plays['epa'] > 0.5).mean(),

                        'def_pass_epa_allowed': def_plays[def_plays['play_type'] == 'pass']['epa'].mean(),
                        'def_run_epa_allowed': def_plays[def_plays['play_type'] == 'run']['epa'].mean(),
                    }

                    all_stats.append(stats)

                processed += 1
                if processed % 500 == 0:
                    pct = (processed / total * 100) if total > 0 else 0
                    print(f"    Progress: {processed:,} combinations processed", end='\r')

    print(f"\n  ✓ Generated {len(all_stats):,} team-week records")

    return pd.DataFrame(all_stats)

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python unpack_and_aggregate_pbp.py <zip_file>")
        print("Example: python unpack_and_aggregate_pbp.py data/pbp_2009_2024.zip")
        return 1

    zip_path = Path(sys.argv[1])

    print("\n" + "="*80)
    print("UNPACK AND AGGREGATE PLAY-BY-PLAY DATA")
    print("="*80)

    # Check if file exists
    if not zip_path.exists():
        print(f"\n✗ ERROR: File not found: {zip_path}")
        return 1

    # Handle split files
    if zip_path.suffix.startswith('.') and zip_path.suffix[1:].isdigit():
        zip_path = reassemble_chunks(zip_path)

    # Unzip
    print(f"\n[1/3] Extracting {zip_path.name}...")
    TEMP_DIR.mkdir(exist_ok=True, parents=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            print(f"  Contents: {file_list}")

            parquet_files = [f for f in file_list if f.endswith('.parquet')]
            if not parquet_files:
                print(f"  ✗ No parquet files found in zip!")
                return 1

            # Extract first parquet file
            parquet_file = parquet_files[0]
            print(f"  Extracting: {parquet_file}...")
            zf.extract(parquet_file, TEMP_DIR)

            extracted_path = TEMP_DIR / parquet_file

    except Exception as e:
        print(f"  ✗ ERROR extracting zip: {e}")
        return 1

    # Load play-by-play
    print(f"\n[2/3] Loading play-by-play data...")

    try:
        pbp = pd.read_parquet(extracted_path)
        print(f"  ✓ Loaded {len(pbp):,} plays")
        print(f"  Columns: {len(pbp.columns)}")

        # Verify required columns
        required = ['season', 'week', 'posteam', 'defteam', 'play_type', 'epa', 'success']
        missing = [col for col in required if col not in pbp.columns]

        if missing:
            print(f"  ✗ Missing required columns: {missing}")
            return 1

        print(f"  ✓ All required columns present")

    except Exception as e:
        print(f"  ✗ ERROR loading parquet: {e}")
        return 1

    # Aggregate
    print(f"\n[3/3] Aggregating to team-week stats...")

    try:
        team_week_df = aggregate_team_week(pbp)

        # Sort
        team_week_df = team_week_df.sort_values(['season', 'week', 'team'])

        # Save
        OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
        team_week_df.to_csv(OUTPUT_FILE, index=False)

        size_kb = OUTPUT_FILE.stat().st_size / 1024
        print(f"\n  ✓ Saved to: {OUTPUT_FILE}")
        print(f"  File size: {size_kb:.1f} KB")

    except Exception as e:
        print(f"  ✗ ERROR during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Cleanup
    print(f"\nCleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    print(f"  ✓ Removed temp directory")

    # Summary
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)

    print(f"\nData summary:")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Records: {len(team_week_df):,}")
    print(f"  Seasons: {team_week_df['season'].min()}-{team_week_df['season'].max()}")
    print(f"  Teams: {team_week_df['team'].nunique()}")

    print(f"\nSample (2023, Week 1):")
    sample = team_week_df[(team_week_df['season'] == 2023) & (team_week_df['week'] == 1)]
    if len(sample) > 0:
        print(sample[['team', 'off_epa_per_play', 'def_epa_allowed_per_play']].head(5).to_string(index=False))

    print("\n" + "="*80)
    print("NEXT STEP: Build v1.3 Model with EPA Features")
    print("="*80)

    print("""
Now we can build v1.3 with professional EPA metrics!

The model will include:
  ✓ EPA per play differential (offense - defense)
  ✓ Success rate differential
  ✓ Explosive play rate differential
  ✓ Recent form (rolling 3-5 game averages)
  ✓ Pass vs run efficiency splits
  ✓ All existing v1.2 features (nfelo, rest, QB, etc.)

Expected improvements:
  - Better prediction accuracy (lower MAE)
  - More reliable probability estimates
  - Stronger feature importance for true game efficiency
  - Professional-grade model matching research standards
""")

    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
