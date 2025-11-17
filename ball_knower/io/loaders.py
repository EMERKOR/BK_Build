"""
Ball Knower Data Loaders - Unified Naming System

This module implements the unified naming convention from DATA_SOURCES.md:
    <category>_<source>_<year>_week_<week>.csv

Canonical Categories:
    - power_ratings: Team strength ratings (offense, defense, overall)
    - team_epa: Team EPA metrics (offense/defense, pass/rush)
    - qb_metrics: QB-level metrics (tiers, EPA, CPOE)
    - schedule_context: Game context (SOS, rest, travel, stadium)

Data Sources (providers):
    - nfelo
    - substack
    - fivethirtyeight
    - pff
    - custom

Directory Structure:
    data/raw/current_season/    - Weekly current season files
    data/raw/historical/         - Multi-year historical files (.parquet)
    data/processed/              - Processed/blended outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Determine project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
CURRENT_SEASON_DIR = RAW_DIR / 'current_season'
HISTORICAL_DIR = RAW_DIR / 'historical'
REFERENCE_DIR = RAW_DIR / 'reference'
PROCESSED_DIR = DATA_DIR / 'processed'

# Canonical categories
VALID_CATEGORIES = {'power_ratings', 'team_epa', 'qb_metrics', 'schedule_context'}

# Known providers (extensible)
KNOWN_PROVIDERS = {'nfelo', 'substack', 'fivethirtyeight', 'pff', 'espn', 'custom', 'blended'}


# ============================================================================
# CORE LOADER FUNCTIONS
# ============================================================================

def load_weekly_file(
    category: str,
    source: str,
    year: int,
    week: int,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load a weekly data file using the unified naming convention.

    Args:
        category: One of 'power_ratings', 'team_epa', 'qb_metrics', 'schedule_context'
        source: Data provider (e.g., 'nfelo', 'substack', 'fivethirtyeight')
        year: Season year (e.g., 2025)
        week: NFL week number (1-18)
        validate: Whether to validate category name (default: True)

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        ValueError: If category is invalid
        FileNotFoundError: If file does not exist

    Example:
        >>> df = load_weekly_file('power_ratings', 'nfelo', 2025, 11)
        >>> # Loads: data/raw/current_season/power_ratings_nfelo_2025_week_11.csv
    """
    if validate and category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}\n"
            f"See docs/DATA_SOURCES.md for the canonical category list."
        )

    # Construct filename following convention: <category>_<source>_<year>_week_<week>.csv
    filename = f"{category}_{source}_{year}_week_{week}.csv"
    filepath = CURRENT_SEASON_DIR / filename

    if not filepath.exists():
        # Provide helpful error message with suggestions
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n\n"
            f"Expected naming convention: {category}_{source}_{year}_week_{week}.csv\n"
            f"Expected location: {CURRENT_SEASON_DIR}/\n\n"
            f"Possible issues:\n"
            f"  1. File may use old naming convention (e.g., '{source}_{category}_...')\n"
            f"  2. File may be in wrong directory\n"
            f"  3. Week or year may be incorrect\n\n"
            f"See docs/DATA_SOURCES.md for the unified naming specification."
        )

    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        print(f"✓ Loaded {category} from {source}: {len(df)} rows ({filename})")
        return df
    except Exception as e:
        raise Exception(f"Failed to load {filepath}: {e}")


def load_historical_file(
    category: str,
    source: str,
    season: int,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load a historical data file (multi-season parquet).

    Args:
        category: One of 'power_ratings', 'team_epa', 'qb_metrics', 'schedule_context'
        source: Data provider
        season: Season year (or 'all' for multi-season files)
        validate: Whether to validate category name

    Returns:
        pd.DataFrame: Historical data

    Example:
        >>> df = load_historical_file('team_epa', 'nflverse', 2024)
        >>> # Loads: data/raw/historical/team_epa_nflverse_2024.parquet
    """
    if validate and category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {VALID_CATEGORIES}")

    # Historical files use .parquet format
    filename = f"{category}_{source}_{season}.parquet"
    filepath = HISTORICAL_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Historical file not found: {filepath}\n"
            f"Expected: {HISTORICAL_DIR}/{filename}"
        )

    try:
        df = pd.read_parquet(filepath)
        print(f"✓ Loaded historical {category} from {source}: {len(df)} rows")
        return df
    except Exception as e:
        raise Exception(f"Failed to load {filepath}: {e}")


# ============================================================================
# CATEGORY-SPECIFIC LOADERS
# ============================================================================

def load_power_ratings(
    source: str,
    year: int,
    week: int,
    historical: bool = False
) -> pd.DataFrame:
    """
    Load power ratings for a specific week.

    Power ratings typically include:
        - overall_rating, offense_rating, defense_rating
        - qb_adjustment (optional)
        - elo, strength metrics

    Args:
        source: Provider (e.g., 'nfelo', 'substack', 'fivethirtyeight')
        year: Season year
        week: NFL week number
        historical: Load from historical files instead of current_season

    Returns:
        pd.DataFrame: Power ratings by team

    Example:
        >>> nfelo_ratings = load_power_ratings('nfelo', 2025, 11)
        >>> substack_ratings = load_power_ratings('substack', 2025, 11)
    """
    if historical:
        return load_historical_file('power_ratings', source, year)
    else:
        return load_weekly_file('power_ratings', source, year, week)


def load_team_epa(
    source: str,
    year: int,
    week: int,
    historical: bool = False
) -> pd.DataFrame:
    """
    Load team EPA metrics for a specific week.

    Team EPA typically includes:
        - offense_epa_total, offense_epa_pass, offense_epa_rush
        - defense_epa_total, defense_epa_pass, defense_epa_rush
        - epa_margin

    Args:
        source: Provider (e.g., 'nfelo', 'substack')
        year: Season year
        week: NFL week number
        historical: Load from historical files

    Returns:
        pd.DataFrame: Team EPA metrics

    Example:
        >>> epa = load_team_epa('nfelo', 2025, 11)
    """
    if historical:
        return load_historical_file('team_epa', source, year)
    else:
        return load_weekly_file('team_epa', source, year, week)


def load_qb_metrics(
    source: str,
    year: int,
    week: int,
    historical: bool = False
) -> pd.DataFrame:
    """
    Load QB-level metrics for a specific week.

    QB metrics typically include:
        - qb_tier or qb_rating
        - qb_epa_per_play
        - cpoe (completion percentage over expected)
        - pressure_efficiency

    Args:
        source: Provider (e.g., 'nfelo', 'substack')
        year: Season year
        week: NFL week number
        historical: Load from historical files

    Returns:
        pd.DataFrame: QB metrics

    Example:
        >>> qb_data = load_qb_metrics('nfelo', 2025, 11)
    """
    if historical:
        return load_historical_file('qb_metrics', source, year)
    else:
        return load_weekly_file('qb_metrics', source, year, week)


def load_schedule_context(
    source: str,
    year: int,
    week: int,
    historical: bool = False
) -> pd.DataFrame:
    """
    Load schedule context and strength of schedule data.

    Schedule context typically includes:
        - sos (strength of schedule)
        - rest_days
        - travel_distance
        - surface, roof, altitude
        - stadium attributes

    Args:
        source: Provider (e.g., 'nfelo')
        year: Season year
        week: NFL week number
        historical: Load from historical files

    Returns:
        pd.DataFrame: Schedule context metrics

    Example:
        >>> sos = load_schedule_context('nfelo', 2025, 11)
    """
    if historical:
        return load_historical_file('schedule_context', source, year)
    else:
        return load_weekly_file('schedule_context', source, year, week)


# ============================================================================
# MULTI-SOURCE LOADERS
# ============================================================================

def load_all_sources(
    category: str,
    year: int,
    week: int,
    sources: Optional[List[str]] = None
) -> dict:
    """
    Load a category from multiple sources.

    Args:
        category: Data category
        year: Season year
        week: NFL week number
        sources: List of sources to load (default: ['nfelo', 'substack'])

    Returns:
        dict: Dictionary mapping source name to DataFrame

    Example:
        >>> ratings = load_all_sources('power_ratings', 2025, 11)
        >>> nfelo_df = ratings['nfelo']
        >>> substack_df = ratings['substack']
    """
    if sources is None:
        sources = ['nfelo', 'substack']

    data = {}
    for source in sources:
        try:
            df = load_weekly_file(category, source, year, week)
            data[source] = df
        except FileNotFoundError:
            print(f"⚠ Skipping {source} - file not found")
            continue

    if not data:
        raise FileNotFoundError(
            f"No data files found for category '{category}' from sources {sources}"
        )

    return data


def load_blended_file(
    category: str,
    year: int,
    week: int
) -> pd.DataFrame:
    """
    Load a pre-blended file that combines multiple sources.

    Blended files follow the same convention but use 'blended' as the source:
        <category>_blended_<year>_week_<week>.csv

    Args:
        category: Data category
        year: Season year
        week: NFL week number

    Returns:
        pd.DataFrame: Blended data combining multiple sources

    Example:
        >>> blended_ratings = load_blended_file('power_ratings', 2025, 11)
    """
    return load_weekly_file(category, 'blended', year, week)


# ============================================================================
# LEGACY COMPATIBILITY LAYER (for migration)
# ============================================================================

# Old filename patterns that need to be migrated
LEGACY_MAPPINGS = {
    # nfelo patterns (source_category_year_week)
    'nfelo_power_ratings': ('power_ratings', 'nfelo'),
    'nfelo_epa_tiers': ('team_epa', 'nfelo'),
    'nfelo_qb_rankings': ('qb_metrics', 'nfelo'),
    'nfelo_strength_of_schedule': ('schedule_context', 'nfelo'),
    'nfelo_epa_tiers_off_def': ('team_epa', 'nfelo'),

    # substack patterns
    'substack_power_ratings': ('power_ratings', 'substack'),
    'substack_qb_epa': ('qb_metrics', 'substack'),
    'substack_weekly_proj_elo': ('power_ratings', 'substack'),
    'substack_weekly_proj_ppg': ('power_ratings', 'substack'),
}


def load_weekly_file_legacy(
    old_pattern: str,
    year: int,
    week: int
) -> pd.DataFrame:
    """
    DEPRECATED: Load using old naming pattern.

    This function provides backward compatibility during migration.
    Use load_weekly_file() with new naming convention instead.

    Args:
        old_pattern: Old file pattern (e.g., 'nfelo_power_ratings')
        year: Season year
        week: NFL week number

    Returns:
        pd.DataFrame: Loaded data

    Example (deprecated):
        >>> df = load_weekly_file_legacy('nfelo_power_ratings', 2025, 11)

    Example (new):
        >>> df = load_power_ratings('nfelo', 2025, 11)
    """
    warnings.warn(
        f"load_weekly_file_legacy() is deprecated. "
        f"Use category-specific loaders (load_power_ratings, etc.) instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if old_pattern not in LEGACY_MAPPINGS:
        raise ValueError(
            f"Unknown legacy pattern '{old_pattern}'. "
            f"Known patterns: {list(LEGACY_MAPPINGS.keys())}"
        )

    category, source = LEGACY_MAPPINGS[old_pattern]

    # Try new naming convention first
    try:
        return load_weekly_file(category, source, year, week)
    except FileNotFoundError:
        # Fall back to old naming pattern if file exists
        old_filename = f"{old_pattern}_{year}_week_{week}.csv"
        old_filepath = CURRENT_SEASON_DIR / old_filename

        if old_filepath.exists():
            print(f"⚠ WARNING: Using legacy filename pattern: {old_filename}")
            print(f"  → Please rename to: {category}_{source}_{year}_week_{week}.csv")
            return pd.read_csv(old_filepath, encoding='utf-8-sig')
        else:
            raise


# ============================================================================
# VALIDATION & DIAGNOSTICS
# ============================================================================

def list_available_files(
    category: Optional[str] = None,
    year: Optional[int] = None,
    week: Optional[int] = None
) -> pd.DataFrame:
    """
    List all available data files matching the naming convention.

    Args:
        category: Filter by category (optional)
        year: Filter by year (optional)
        week: Filter by week (optional)

    Returns:
        pd.DataFrame: Available files with parsed metadata

    Example:
        >>> files = list_available_files(category='power_ratings', year=2025)
        >>> print(files[['category', 'source', 'week', 'filename']])
    """
    pattern = "*.csv"
    files = list(CURRENT_SEASON_DIR.glob(pattern))

    parsed = []
    for f in files:
        parts = f.stem.split('_')

        # Expected format: <category>_<source>_<year>_week_<week>
        # Minimum parts: category, source, year, 'week', week_number
        if len(parts) >= 5 and parts[-2] == 'week':
            try:
                cat = parts[0]
                src = parts[1]
                yr = int(parts[2])
                wk = int(parts[-1])

                # Apply filters
                if category and cat != category:
                    continue
                if year and yr != year:
                    continue
                if week and wk != week:
                    continue

                parsed.append({
                    'category': cat,
                    'source': src,
                    'year': yr,
                    'week': wk,
                    'filename': f.name,
                    'valid': cat in VALID_CATEGORIES
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(parsed)
    if len(df) > 0:
        df = df.sort_values(['category', 'source', 'year', 'week'])
    return df


def validate_naming_convention() -> dict:
    """
    Validate all files in current_season directory against naming convention.

    Returns:
        dict: Validation report with valid/invalid files

    Example:
        >>> report = validate_naming_convention()
        >>> print(f"Valid: {len(report['valid'])}, Invalid: {len(report['invalid'])}")
    """
    all_files = list(CURRENT_SEASON_DIR.glob("*.csv"))

    valid = []
    invalid = []
    needs_rename = []

    for f in all_files:
        parts = f.stem.split('_')

        # Check if follows convention: <category>_<source>_<year>_week_<week>
        if len(parts) >= 5 and parts[-2] == 'week':
            try:
                category = parts[0]
                source = parts[1]
                year = int(parts[2])
                week = int(parts[-1])

                if category in VALID_CATEGORIES:
                    valid.append(f.name)
                else:
                    invalid.append({
                        'file': f.name,
                        'reason': f"Invalid category '{category}'"
                    })
            except ValueError:
                invalid.append({
                    'file': f.name,
                    'reason': 'Invalid year/week format'
                })
        else:
            # Check if it matches a legacy pattern
            for old_pattern, (new_cat, new_src) in LEGACY_MAPPINGS.items():
                if f.stem.startswith(old_pattern):
                    needs_rename.append({
                        'old': f.name,
                        'suggested': f.stem.replace(old_pattern, f"{new_cat}_{new_src}") + '.csv',
                        'category': new_cat,
                        'source': new_src
                    })
                    break
            else:
                invalid.append({
                    'file': f.name,
                    'reason': 'Does not match naming convention'
                })

    return {
        'valid': valid,
        'invalid': invalid,
        'needs_rename': needs_rename,
        'summary': {
            'total': len(all_files),
            'valid_count': len(valid),
            'invalid_count': len(invalid),
            'rename_count': len(needs_rename)
        }
    }


def print_validation_report():
    """
    Print a formatted validation report of all data files.

    Example:
        >>> print_validation_report()
    """
    report = validate_naming_convention()

    print("\n" + "="*80)
    print("DATA FILES VALIDATION REPORT")
    print("="*80)

    print(f"\nTotal files: {report['summary']['total']}")
    print(f"✓ Valid: {report['summary']['valid_count']}")
    print(f"⚠ Needs rename: {report['summary']['rename_count']}")
    print(f"✗ Invalid: {report['summary']['invalid_count']}")

    if report['needs_rename']:
        print("\n" + "-"*80)
        print("FILES NEEDING RENAME (old → new convention):")
        print("-"*80)
        for item in report['needs_rename']:
            print(f"\n  Old: {item['old']}")
            print(f"  New: {item['suggested']}")
            print(f"  Category: {item['category']} | Source: {item['source']}")

    if report['invalid']:
        print("\n" + "-"*80)
        print("INVALID FILES:")
        print("-"*80)
        for item in report['invalid']:
            print(f"\n  File: {item['file']}")
            print(f"  Reason: {item['reason']}")

    print("\n" + "="*80 + "\n")


# ============================================================================
# INITIALIZATION
# ============================================================================

def ensure_directories():
    """Create required directories if they don't exist."""
    for directory in [RAW_DIR, CURRENT_SEASON_DIR, HISTORICAL_DIR, REFERENCE_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()


if __name__ == "__main__":
    # Run validation when module is executed directly
    print_validation_report()
