"""
Data Ingestion Module

Provides automated data fetching and staging for weekly predictions and model training.
This module handles downloading and caching of:
- nfelo ratings
- play-by-play aggregates
- weekly vegas lines
- team metadata
- injury reports
- roster/QB depth charts

All data is staged in a consistent local directory structure under data/current_season/
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Try to import requests, but make it optional for environments without it
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests library not available - ingestion functionality limited")

# Try to import pandas for CSV processing
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas library not available - ingestion functionality limited")


# ============================================================================
# DATA SOURCE ENDPOINTS
# ============================================================================

# NFLverse data sources (public GitHub repos)
# Note: greerreNFL/nfelo has all-seasons file, we filter by game_id pattern
NFELO_RATINGS_URL = "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv"
NFLVERSE_SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/main/schedules/sched_{season}.csv"
VEGAS_LINES_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/main/vegas/lines/lines_{season}.csv"

# TODO: Future data sources
INJURY_REPORTS_URL = "https://example.com/api/injuries/{season}/{week}"  # TODO: Replace with actual endpoint
ROSTER_DEPTH_URL = "https://example.com/api/rosters/{season}/{week}"  # TODO: Replace with actual endpoint


# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

def get_data_dir() -> Path:
    """Get the root data directory."""
    return Path(__file__).resolve().parents[2] / "data"


def get_season_data_dir(season: int) -> Path:
    """Get the data directory for a specific season."""
    return get_data_dir() / "current_season"


def get_ratings_dir(season: int) -> Path:
    """Get the ratings directory for a specific season."""
    return get_season_data_dir(season) / "ratings"


def get_vegas_dir(season: int) -> Path:
    """Get the vegas lines directory for a specific season."""
    return get_season_data_dir(season) / "vegas"


def get_pbp_dir(season: int) -> Path:
    """Get the play-by-play directory for a specific season."""
    return get_season_data_dir(season) / "pbp"


# ============================================================================
# CACHING AND DOWNLOAD UTILITIES
# ============================================================================

def _download_file(url: str, destination: Path, force: bool = False) -> bool:
    """
    Download a file from URL to destination path.

    Args:
        url: Source URL
        destination: Local file path
        force: If True, overwrite existing file

    Returns:
        True if file was downloaded, False if skipped (already exists and not forced)

    Raises:
        requests.HTTPError: If download fails
        IOError: If file cannot be written

    TODO:
        - Add authentication headers support
        - Add retry logic with exponential backoff
        - Add progress bar for large files
        - Add checksum validation
    """
    if not HAS_REQUESTS:
        print(f"      WARNING: Cannot download {url} - requests library not available")
        print(f"      Please install with: pip install requests")
        return False

    if destination.exists() and not force:
        print(f"      Cached: {destination.name}")
        return False

    print(f"      Downloading {destination.name}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Ensure directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        destination.write_bytes(response.content)

        print(f"      Downloaded: {destination.name} ({len(response.content)} bytes)")
        return True

    except requests.HTTPError as e:
        print(f"      ERROR: HTTP {e.response.status_code} - {url}")
        raise
    except requests.RequestException as e:
        print(f"      ERROR: Download failed - {e}")
        raise
    except IOError as e:
        print(f"      ERROR: Cannot write file - {e}")
        raise


def _check_cache(file_path: Path, max_age_hours: int = 24) -> bool:
    """
    Check if cached file is fresh enough.

    Args:
        file_path: Path to cached file
        max_age_hours: Maximum age in hours before refresh needed

    Returns:
        True if cache is valid, False if refresh needed

    TODO:
        - Implement actual cache age checking
        - Add cache validation metadata (checksums, version tags)
    """
    if not file_path.exists():
        return False

    # TODO: Check file modification time
    # TODO: Compare against max_age_hours

    return True


# ============================================================================
# SEASON DATA INGESTION
# ============================================================================

def fetch_season_data(season: int, *, force: bool = False) -> Dict[str, Path]:
    """
    Fetches or updates all required data for a given NFL season.

    Downloads or refreshes:
        - NFLverse schedule
        - Vegas lines

    Args:
        season: NFL season year (e.g., 2025)
        force: If True, re-download even if cached data exists

    Returns:
        Dictionary mapping data types to their local file paths

    Example:
        >>> paths = fetch_season_data(2024)
        >>> print(paths['schedule'])
        Path('data/current_season/ratings/schedule_2024.csv')
    """
    print(f"\n{'='*60}")
    print(f"INGESTION: Fetching season data for {season}")
    print(f"{'='*60}")

    # Ensure directory structure exists
    ratings_dir = get_ratings_dir(season)
    vegas_dir = get_vegas_dir(season)
    pbp_dir = get_pbp_dir(season)

    ratings_dir.mkdir(parents=True, exist_ok=True)
    vegas_dir.mkdir(parents=True, exist_ok=True)
    pbp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Data directories ready:")
    print(f"  - Ratings: {ratings_dir}")
    print(f"  - Vegas: {vegas_dir}")
    print(f"  - PBP: {pbp_dir}")

    # Define target files
    files = {}

    print(f"\nDownloading data (force={force}):")

    # Download NFLverse schedule
    print(f"\n[1/2] NFLverse schedule for {season}")
    schedule_url = NFLVERSE_SCHEDULE_URL.format(season=season)
    schedule_file = ratings_dir / f"schedule_{season}.csv"
    try:
        _download_file(schedule_url, schedule_file, force=force)
        files['schedule'] = schedule_file
    except Exception as e:
        print(f"      WARNING: Could not download schedule - {e}")

    # Download Vegas lines for entire season
    print(f"\n[2/2] Vegas lines for {season}")
    vegas_url = VEGAS_LINES_URL.format(season=season)
    vegas_file = vegas_dir / f"vegas_lines_{season}.csv"
    try:
        _download_file(vegas_url, vegas_file, force=force)
        files['vegas'] = vegas_file
    except Exception as e:
        print(f"      WARNING: Could not download vegas lines - {e}")

    print(f"\n{'='*60}")
    print(f"  Season data ingestion complete")
    downloaded_count = len([f for f in files.values() if f.exists()])
    print(f"  Downloaded {downloaded_count} / {len(files)} files")
    print(f"{'='*60}\n")

    return files


# ============================================================================
# WEEKLY DATA INGESTION
# ============================================================================

def fetch_week_data(season: int, week: int, *, force: bool = False) -> Dict[str, Path]:
    """
    Fetches weekly supplemental data.

    Downloads or refreshes:
        - weekly nfelo ratings (filtered)
        - weekly vegas lines (filtered)

    Args:
        season: NFL season year (e.g., 2025)
        week: Week number (1-18 for regular season)
        force: If True, re-download even if cached data exists

    Returns:
        Dictionary mapping data types to their local file paths

    Example:
        >>> paths = fetch_week_data(2024, 11)
        >>> print(paths['vegas'])
        Path('data/current_season/vegas/vegas_week_2024_11.csv')
    """
    print(f"\n{'='*60}")
    print(f"INGESTION: Fetching week data for {season} Week {week}")
    print(f"{'='*60}")

    # Ensure directory structure exists
    vegas_dir = get_vegas_dir(season)
    ratings_dir = get_ratings_dir(season)
    vegas_dir.mkdir(parents=True, exist_ok=True)
    ratings_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Data directories ready:")
    print(f"  - Vegas: {vegas_dir}")
    print(f"  - Ratings: {ratings_dir}")

    # Define target files
    files = {}

    print(f"\nDownloading weekly data (force={force}):")

    # Download and filter nfelo ratings for this week
    print(f"\n[1/2] nfelo ratings for Week {week}")
    nfelo_url = NFELO_RATINGS_URL  # All-seasons file, no formatting needed
    nfelo_week_file = ratings_dir / f"nfelo_week_{season}_{week}.csv"

    if not nfelo_week_file.exists() or force:
        try:
            if not HAS_PANDAS:
                print(f"      ERROR: pandas required for filtering nfelo data")
            elif not HAS_REQUESTS:
                print(f"      ERROR: requests required for downloading data")
            else:
                print(f"      Downloading nfelo data (all seasons)...")
                response = requests.get(nfelo_url, timeout=30)
                response.raise_for_status()

                # Parse and filter by game_id pattern (YYYY_WW_AWAY_HOME)
                df = pd.read_csv(io.StringIO(response.text))

                if 'game_id' in df.columns:
                    # Extract season and week from game_id
                    df['parsed_season'] = df['game_id'].str.split('_').str[0].astype(int)
                    df['parsed_week'] = df['game_id'].str.split('_').str[1].astype(int)

                    # Filter for this season and week
                    df_week = df[(df['parsed_season'] == season) & (df['parsed_week'] == week)]

                    # Drop parsed columns before saving
                    df_week = df_week.drop(columns=['parsed_season', 'parsed_week'])
                    df_week.to_csv(nfelo_week_file, index=False)
                    print(f"      Downloaded: {nfelo_week_file.name} ({len(df_week)} games)")
                    files['nfelo'] = nfelo_week_file
                else:
                    print(f"      WARNING: nfelo data missing game_id column")
        except Exception as e:
            print(f"      WARNING: Could not download nfelo ratings - {e}")
    else:
        print(f"      Cached: {nfelo_week_file.name}")
        files['nfelo'] = nfelo_week_file

    # Download and filter Vegas lines for this week  
    print(f"\n[2/2] Vegas lines for Week {week}")
    vegas_url = VEGAS_LINES_URL.format(season=season)
    vegas_week_file = vegas_dir / f"vegas_week_{season}_{week}.csv"
    
    if not vegas_week_file.exists() or force:
        try:
            if not HAS_PANDAS:
                print(f"      ERROR: pandas required for filtering vegas data")
            elif not HAS_REQUESTS:
                print(f"      ERROR: requests required for downloading data")
            else:
                print(f"      Downloading full season vegas data...")
                response = requests.get(vegas_url, timeout=30)
                response.raise_for_status()
                
                # Parse and filter by week
                df = pd.read_csv(io.StringIO(response.text))
                
                # Filter for this week
                if 'week' in df.columns:
                    df_week = df[df['week'] == week]
                    df_week.to_csv(vegas_week_file, index=False)
                    print(f"      Downloaded: {vegas_week_file.name} ({len(df_week)} games)")
                    files['vegas'] = vegas_week_file
                else:
                    print(f"      WARNING: vegas data missing week column")
        except Exception as e:
            print(f"      WARNING: Could not download vegas lines - {e}")
    else:
        print(f"      Cached: {vegas_week_file.name}")
        files['vegas'] = vegas_week_file

    print(f"\n{'='*60}")
    print(f"  Weekly data ingestion complete")
    downloaded_count = len([f for f in files.values() if f.exists()])
    print(f"  Downloaded {downloaded_count} files")
    print(f"{'='*60}\n")

    return files


# ============================================================================
# DATA STAGING FOR PIPELINE
# ============================================================================

def stage_data_for_pipeline(season: int, week: int) -> Dict[str, Path]:
    """
    Ensures the local directory structure is populated with all required data.

    Validates that required files exist and returns their paths for use in
    prediction and training pipelines.

    Args:
        season: NFL season year (e.g., 2025)
        week: Week number (1-18 for regular season)

    Returns:
        Dictionary of fully-qualified paths to staged files:
        {
            'schedule': Path to NFLverse schedule CSV,
            'nfelo': Path to nfelo week ratings CSV,
            'vegas': Path to vegas week lines CSV
        }

    Example:
        >>> paths = stage_data_for_pipeline(2024, 11)
        >>> df_schedule = pd.read_csv(paths['schedule'])

    TODO:
        - Add schema validation for each staged file
        - Add data quality checks (null counts, date ranges)
        - Add automatic fetch if missing (call fetch_season_data/fetch_week_data)
        - Add version/timestamp metadata to staged manifest
    """
    print(f"\n{'='*60}")
    print(f"STAGING: Validating data for {season} Week {week}")
    print(f"{'='*60}")

    # Define expected file paths
    schedule_file = get_ratings_dir(season) / f"schedule_{season}.csv"
    nfelo_file = get_ratings_dir(season) / f"nfelo_week_{season}_{week}.csv"
    vegas_file = get_vegas_dir(season) / f"vegas_week_{season}_{week}.csv"

    staged_paths = {
        'schedule': schedule_file,
        'nfelo': nfelo_file,
        'vegas': vegas_file
    }

    # Check which files exist
    print(f"\nValidating staged files:")

    required_files = ['schedule', 'nfelo', 'vegas']
    missing_required = []

    for file_type in required_files:
        path = staged_paths[file_type]
        if path.exists():
            print(f"  ✓ {file_type}: {path}")
        else:
            print(f"  ✗ {file_type}: MISSING - {path}")
            missing_required.append(file_type)

    if missing_required:
        print(f"\n✗ Missing required files: {', '.join(missing_required)}")
        print(f"   Run: bk_build.py ingest --season {season} --week {week}")
        # TODO: Optionally auto-fetch missing data
        # fetch_season_data(season)
        # fetch_week_data(season, week)
    else:
        print(f"\n✓ All required data staged and ready")

    print(f"{'='*60}\n")

    return staged_paths


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _save_csv_string(content: str, path: Path) -> None:
    """
    Save CSV content string to a file.

    Args:
        content: CSV content as string
        path: Destination file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
