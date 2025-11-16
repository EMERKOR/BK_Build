"""
Test multiple methods to access NFLverse play-by-play and EPA data

Tries:
1. Direct parquet URLs
2. CSV alternatives
3. Pre-aggregated weekly stats
4. Different hosts/CDNs
5. Raw requests library
"""

import pandas as pd
import requests
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("TESTING ALTERNATIVE DATA ACCESS METHODS")
print("="*80)

# ============================================================================
# METHOD 1: Direct Parquet URLs (Different Hosts)
# ============================================================================

print("\n[METHOD 1] Direct parquet URLs...")

# Try different base URLs
base_urls = [
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet",
    "https://github.com/nflverse/nflverse-data/raw/master/data/play_by_play_{year}.parquet",
    "https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/play_by_play_{year}.parquet",
]

test_year = 2023  # Use 2023 instead of 2024

for i, url_template in enumerate(base_urls, 1):
    try:
        url = url_template.format(year=test_year)
        print(f"\n  [{i}] Trying: {url[:80]}...")

        # Try with requests first to check headers
        response = requests.head(url, timeout=10)
        print(f"      Status: {response.status_code}")

        if response.status_code == 200:
            print(f"      ✓ URL is accessible! Attempting to load...")
            df = pd.read_parquet(url)
            print(f"      ✓ SUCCESS! Loaded {len(df):,} plays")
            print(f"      Columns: {len(df.columns)}")

            # Check for EPA columns
            epa_cols = [c for c in df.columns if 'epa' in c.lower()]
            print(f"      EPA columns: {epa_cols[:5]}")

            # Save sample
            sample_file = project_root / 'data' / f'pbp_{test_year}_sample.parquet'
            sample_file.parent.mkdir(exist_ok=True)
            df.head(1000).to_parquet(sample_file)
            print(f"      ✓ Saved sample to {sample_file}")
            break

    except Exception as e:
        print(f"      ✗ Failed: {str(e)[:100]}")

# ============================================================================
# METHOD 2: CSV Format
# ============================================================================

print("\n" + "="*80)
print("[METHOD 2] CSV format...")
print("="*80)

csv_urls = [
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz",
    "https://raw.githubusercontent.com/nflverse/nflverse-data/master/data/play_by_play_{year}.csv",
]

for i, url_template in enumerate(csv_urls, 1):
    try:
        url = url_template.format(year=test_year)
        print(f"\n  [{i}] Trying: {url[:80]}...")

        response = requests.head(url, timeout=10)
        print(f"      Status: {response.status_code}")

        if response.status_code == 200:
            print(f"      ✓ URL is accessible! Attempting to load...")
            df = pd.read_csv(url, nrows=1000)  # Load just 1000 rows to test
            print(f"      ✓ SUCCESS! Loaded {len(df):,} plays (sample)")
            print(f"      Columns: {len(df.columns)}")
            break

    except Exception as e:
        print(f"      ✗ Failed: {str(e)[:100]}")

# ============================================================================
# METHOD 3: Pre-Aggregated Weekly Stats
# ============================================================================

print("\n" + "="*80)
print("[METHOD 3] Pre-aggregated weekly team stats...")
print("="*80)

weekly_urls = [
    "https://github.com/nflverse/nflverse-data/releases/download/weekly/weekly_{year}.parquet",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.parquet",
]

for i, url_template in enumerate(weekly_urls, 1):
    try:
        url = url_template.format(year=test_year)
        print(f"\n  [{i}] Trying: {url[:80]}...")

        response = requests.head(url, timeout=10)
        print(f"      Status: {response.status_code}")

        if response.status_code == 200:
            print(f"      ✓ URL is accessible! Attempting to load...")
            df = pd.read_parquet(url)
            print(f"      ✓ SUCCESS! Loaded {len(df):,} records")
            print(f"      Columns: {df.columns.tolist()[:20]}")

            # Check for EPA-related columns
            epa_cols = [c for c in df.columns if 'epa' in c.lower() or 'efficiency' in c.lower()]
            if epa_cols:
                print(f"      EPA/efficiency columns: {epa_cols[:10]}")

            break

    except Exception as e:
        print(f"      ✗ Failed: {str(e)[:100]}")

# ============================================================================
# METHOD 4: Direct HTTP Download with Custom Headers
# ============================================================================

print("\n" + "="*80)
print("[METHOD 4] Direct HTTP with custom headers...")
print("="*80)

try:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{test_year}.parquet"

    print(f"\n  Trying with custom headers: {url[:80]}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/octet-stream,*/*',
    }

    response = requests.get(url, headers=headers, timeout=30, stream=True)
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        # Save to temp file then read
        temp_file = project_root / 'data' / 'temp_pbp.parquet'
        temp_file.parent.mkdir(exist_ok=True)

        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  ✓ Downloaded to temp file")

        df = pd.read_parquet(temp_file)
        print(f"  ✓ SUCCESS! Loaded {len(df):,} plays")

        # Clean up
        temp_file.unlink()

except Exception as e:
    print(f"  ✗ Failed: {str(e)[:200]}")

# ============================================================================
# METHOD 5: Alternative Data Sources
# ============================================================================

print("\n" + "="*80)
print("[METHOD 5] Alternative data sources...")
print("="*80)

alt_sources = [
    {
        'name': 'nflfastR raw GitHub',
        'url': 'https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/pbp_{year}.rds'
    },
    {
        'name': 'Sports Reference (if available)',
        'url': 'https://www.sports-reference.com/cfb/years/{year}-play-index.html'
    },
]

for source in alt_sources:
    try:
        url = source['url'].format(year=test_year)
        print(f"\n  {source['name']}: {url[:80]}...")

        response = requests.head(url, timeout=10)
        print(f"    Status: {response.status_code}")

        if response.status_code == 200:
            print(f"    ✓ Available!")

    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:100]}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print("""
If any method above succeeded:
  → We can use that approach going forward
  → Set up automated weekly downloads
  → Store aggregated data (not full play-by-play)

If all methods failed:
  → OPTION A: Manual download and commit
    - Download files manually outside this environment
    - Commit aggregated team-week stats (much smaller than full PBP)
    - Update weekly via manual process

  → OPTION B: Use pre-aggregated data only
    - Focus on weekly team stats (smaller files)
    - Skip play-by-play entirely
    - Still get EPA, success rate, etc.

  → OPTION C: Build without EPA for now
    - Use v1.2 features (strong baseline)
    - Add EPA later when access improves
    - Focus on score prediction architecture

Recommended File Storage Strategy:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DON'T store full play-by-play (too large)
2. DO store team-week aggregates:
   - data/team_week_epa_2009_2024.csv (~100KB instead of 500MB)
   - Columns: season, week, team, off_epa, def_epa, success_rate, etc.
   - Covers all historical data, small enough to commit

3. Weekly update process:
   - Download current week PBP (small)
   - Aggregate to team stats
   - Append to historical file
   - Commit updated aggregates only

4. For current week predictions:
   - Fetch fresh schedule data (works now)
   - Use pre-aggregated historical stats for team ratings
   - No need to download full PBP every time
""")

print("="*80 + "\n")
