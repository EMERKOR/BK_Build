"""
Create compressed play-by-play data for GitHub upload

Run this on your local machine to:
1. Download NFLverse play-by-play data
2. Compress to zip file
3. Check if it meets GitHub's file size limits
4. Provide upload instructions

GitHub Limits:
- Individual file: 100 MB max (recommended 50 MB)
- Use Git LFS for files 50-100 MB
- Split into multiple files if > 100 MB

Usage:
    python create_pbp_zip_for_upload.py
"""

import sys
import zipfile
from pathlib import Path

print("\n" + "="*80)
print("CREATE PBP ZIP FOR GITHUB UPLOAD")
print("="*80)

# Check dependencies
try:
    import nfl_data_py as nfl
    import pandas as pd
    print("\n✓ Required libraries installed")
except ImportError as e:
    print(f"\n✗ Missing library: {e}")
    print("\nInstall with: pip install nfl_data_py pandas pyarrow")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

START_YEAR = 2009
END_YEAR = 2024
OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

# File paths
RAW_PARQUET = OUTPUT_DIR / f'pbp_{START_YEAR}_{END_YEAR}.parquet'
ZIP_FILE = OUTPUT_DIR / f'pbp_{START_YEAR}_{END_YEAR}.zip'

# GitHub limits (in MB)
GITHUB_LIMIT_MB = 100
RECOMMENDED_MB = 50

# ============================================================================
# DOWNLOAD
# ============================================================================

print(f"\n[1/3] Downloading play-by-play data ({START_YEAR}-{END_YEAR})...")
print("This may take several minutes...")

try:
    seasons = list(range(START_YEAR, END_YEAR + 1))
    pbp = nfl.import_pbp_data(seasons)

    print(f"  ✓ Downloaded {len(pbp):,} plays")
    print(f"  Columns: {len(pbp.columns)}")

except Exception as e:
    print(f"\n✗ ERROR downloading data: {e}")
    sys.exit(1)

# ============================================================================
# SAVE AND COMPRESS
# ============================================================================

print(f"\n[2/3] Saving and compressing...")

# Save to parquet (already compressed format)
print(f"  Saving to parquet: {RAW_PARQUET}...")
pbp.to_parquet(RAW_PARQUET, compression='gzip')

parquet_size_mb = RAW_PARQUET.stat().st_size / (1024 * 1024)
print(f"  ✓ Parquet file: {parquet_size_mb:.1f} MB")

# Compress to zip
print(f"  Compressing to zip: {ZIP_FILE}...")
with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    zf.write(RAW_PARQUET, RAW_PARQUET.name)

zip_size_mb = ZIP_FILE.stat().st_size / (1024 * 1024)
compression_ratio = (1 - zip_size_mb / parquet_size_mb) * 100

print(f"  ✓ Zip file: {zip_size_mb:.1f} MB")
print(f"  Compression: {compression_ratio:.1f}% reduction")

# ============================================================================
# CHECK GITHUB LIMITS
# ============================================================================

print(f"\n[3/3] Checking GitHub compatibility...")

if zip_size_mb <= RECOMMENDED_MB:
    print(f"  ✓ File size is perfect! ({zip_size_mb:.1f} MB < {RECOMMENDED_MB} MB)")
    print(f"  Can upload directly to GitHub")
    split_needed = False

elif zip_size_mb <= GITHUB_LIMIT_MB:
    print(f"  ⚠ File is large ({zip_size_mb:.1f} MB)")
    print(f"  Recommend using Git LFS for files > {RECOMMENDED_MB} MB")
    print(f"  Or can upload directly (under {GITHUB_LIMIT_MB} MB hard limit)")
    split_needed = False

else:
    print(f"  ✗ File too large for GitHub! ({zip_size_mb:.1f} MB > {GITHUB_LIMIT_MB} MB)")
    print(f"  Will split into smaller chunks...")
    split_needed = True

# ============================================================================
# SPLIT IF NEEDED
# ============================================================================

if split_needed:
    print(f"\nSplitting into chunks...")

    chunk_size_mb = 90  # Leave margin below 100 MB limit
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    chunk_num = 1
    with open(ZIP_FILE, 'rb') as f:
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break

            chunk_file = OUTPUT_DIR / f'pbp_{START_YEAR}_{END_YEAR}.zip.{chunk_num:03d}'
            with open(chunk_file, 'wb') as cf:
                cf.write(chunk)

            chunk_mb = chunk_file.stat().st_size / (1024 * 1024)
            print(f"  Created: {chunk_file.name} ({chunk_mb:.1f} MB)")
            chunk_num += 1

    print(f"\n  ✓ Split into {chunk_num - 1} chunks")

    # Create reassembly script
    reassemble_script = OUTPUT_DIR / 'reassemble_pbp_chunks.sh'
    with open(reassemble_script, 'w') as f:
        f.write(f"""#!/bin/bash
# Reassemble split PBP zip file

echo "Reassembling pbp_{START_YEAR}_{END_YEAR}.zip..."
cat pbp_{START_YEAR}_{END_YEAR}.zip.* > pbp_{START_YEAR}_{END_YEAR}.zip
echo "✓ Done! File: pbp_{START_YEAR}_{END_YEAR}.zip"
""")
    reassemble_script.chmod(0o755)
    print(f"  ✓ Created reassembly script: {reassemble_script.name}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("UPLOAD INSTRUCTIONS")
print("="*80)

if not split_needed:
    print(f"""
✓ Single file ready for upload!

File: {ZIP_FILE}
Size: {zip_size_mb:.1f} MB

Upload to GitHub:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option 1: Via Git (if file < {RECOMMENDED_MB} MB)
  cd /path/to/BK_Build
  cp {ZIP_FILE} data/
  git add data/{ZIP_FILE.name}
  git commit -m "Add historical play-by-play data ({START_YEAR}-{END_YEAR})"
  git push

Option 2: Via GitHub Web Interface (if file < {GITHUB_LIMIT_MB} MB)
  1. Go to: https://github.com/EMERKOR/BK_Build
  2. Navigate to data/ folder
  3. Click "Add file" → "Upload files"
  4. Drag {ZIP_FILE.name}
  5. Commit with message: "Add historical play-by-play data"

Option 3: Git LFS (if file {RECOMMENDED_MB}-{GITHUB_LIMIT_MB} MB)
  git lfs track "data/*.zip"
  git add .gitattributes data/{ZIP_FILE.name}
  git commit -m "Add PBP data via LFS"
  git push
""")

else:
    print(f"""
File was split into chunks to meet GitHub limits:

Chunks created:
""")
    for i in range(1, chunk_num):
        chunk_file = OUTPUT_DIR / f'pbp_{START_YEAR}_{END_YEAR}.zip.{i:03d}'
        chunk_mb = chunk_file.stat().st_size / (1024 * 1024)
        print(f"  - {chunk_file.name} ({chunk_mb:.1f} MB)")

    print(f"""
Upload to GitHub:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd /path/to/BK_Build
  cp data/pbp_{START_YEAR}_{END_YEAR}.zip.* data/
  cp data/reassemble_pbp_chunks.sh data/
  git add data/pbp_{START_YEAR}_{END_YEAR}.zip.*
  git add data/reassemble_pbp_chunks.sh
  git commit -m "Add PBP data (split into {chunk_num - 1} chunks)"
  git push

After upload, Claude will reassemble and process the chunks automatically.
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print(f"""
1. Upload file(s) to GitHub using instructions above

2. Let Claude know it's ready:
   "I've uploaded the play-by-play zip file"

3. Claude will:
   - Unzip the data
   - Aggregate to team-week stats
   - Build v1.3 model with EPA features
   - Compare to v1.2 baseline

4. You get:
   - Professional EPA-based model
   - Full historical analysis
   - No more manual updates (can re-run weekly)
""")

print("="*80 + "\n")
