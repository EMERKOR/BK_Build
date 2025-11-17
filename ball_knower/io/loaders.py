"""
Unified Data Loaders for Ball Knower

This module provides a unified interface for loading NFL data from multiple sources.
It supports both the new category-first naming convention and legacy provider-first
filenames via automatic fallback.

Target naming convention (category-first):
    {category}_{provider}_{season}_week_{week}.csv

Legacy naming convention (provider-first):
    {provider}_{category_variant}_{season}_week_{week}.csv

The loaders automatically try the new pattern first, then fall back to known legacy
patterns for the current dataset (2025 Week 11).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import warnings

# Default data directory (relative to this file: ball_knower/io/loaders.py)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "current_season"

# Valid categories in the new naming scheme
VALID_CATEGORIES = [
    "power_ratings",
    "epa_tiers",
    "strength_of_schedule",
    "qb_rankings",
    "qb_epa",
    "weekly_projections_ppg",
    "weekly_projections_elo",
    "win_totals",
    "receiving_leaders",
]

# Valid data providers
VALID_PROVIDERS = ["nfelo", "substack", "pff", "espn", "bk"]

# Fallback filename patterns for current dataset (2025 Week 11)
# Maps (category, provider) -> legacy filename pattern
FALLBACK_FILENAMES = {
    # nfelo files (provider-first)
    ("power_ratings", "nfelo"): "nfelo_power_ratings_{season}_week_{week}.csv",
    ("epa_tiers", "nfelo"): "nfelo_epa_tiers_off_def_{season}_week_{week}.csv",
    ("strength_of_schedule", "nfelo"): "nfelo_strength_of_schedule_{season}_week_{week}.csv",
    ("qb_rankings", "nfelo"): "nfelo_qb_rankings_{season}_week_{week}.csv",
    ("win_totals", "nfelo"): "nfelo_nfl_win_totals_{season}_week_{week} (1).csv",  # Note: includes " (1)"
    ("receiving_leaders", "nfelo"): "nfelo_nfl_receiving_leaders_{season}_week_{week}.csv",

    # substack files (provider-first)
    ("power_ratings", "substack"): "substack_power_ratings_{season}_week_{week}.csv",
    ("qb_epa", "substack"): "substack_qb_epa_{season}_week_{week}.csv",
    ("weekly_projections_elo", "substack"): "substack_weekly_proj_elo_{season}_week_{week}.csv",
    ("weekly_projections_ppg", "substack"): "substack_weekly_proj_ppg_{season}_week_{week}.csv",
}


def _resolve_file_path(
    category: str,
    provider: str,
    week: int,
    season: int,
    data_dir: Path
) -> Path:
    """
    Resolve file path for a given category, provider, week, and season.

    Tries the new category-first pattern first, then falls back to known
    legacy provider-first patterns if the file doesn't exist.

    Args:
        category: Data category (e.g., 'power_ratings', 'epa_tiers')
        provider: Data provider (e.g., 'nfelo', 'substack')
        week: NFL week number
        season: NFL season year
        data_dir: Directory containing data files

    Returns:
        Path object pointing to the resolved file

    Raises:
        FileNotFoundError: If file cannot be found using either pattern
        ValueError: If category or provider is invalid
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}"
        )

    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Invalid provider '{provider}'. Must be one of: {', '.join(VALID_PROVIDERS)}"
        )

    # Try new category-first pattern
    new_pattern = f"{category}_{provider}_{season}_week_{week}.csv"
    new_path = data_dir / new_pattern

    if new_path.exists():
        return new_path

    # Try legacy provider-first pattern
    fallback_key = (category, provider)
    if fallback_key in FALLBACK_FILENAMES:
        legacy_pattern = FALLBACK_FILENAMES[fallback_key].format(
            season=season,
            week=week
        )
        legacy_path = data_dir / legacy_pattern

        if legacy_path.exists():
            # Issue a deprecation warning
            warnings.warn(
                f"Using legacy filename pattern '{legacy_pattern}'. "
                f"Consider renaming to '{new_pattern}' for consistency.",
                DeprecationWarning,
                stacklevel=3
            )
            return legacy_path

    # Neither pattern found - raise informative error
    tried_paths = [str(new_path)]
    if fallback_key in FALLBACK_FILENAMES:
        tried_paths.append(str(legacy_path))

    raise FileNotFoundError(
        f"Could not find data file for category='{category}', provider='{provider}', "
        f"week={week}, season={season}.\n"
        f"Tried paths:\n  " + "\n  ".join(tried_paths)
    )


def _normalize_team_column_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize team column names and standardize team abbreviations.

    Looks for columns named 'Team', 'team', or similar and:
    1. Renames to lowercase 'team'
    2. Applies team name normalization from src.team_mapping

    Args:
        df: DataFrame with team data

    Returns:
        DataFrame with normalized team column (modified in place)
    """
    from src.team_mapping import normalize_team_name

    # Find team column (case-insensitive)
    team_col = None
    for col in df.columns:
        if col.lower() in ['team', 'teams']:
            team_col = col
            break

    if team_col is None:
        # No team column found - return as-is
        return df

    # Rename to lowercase 'team' if needed
    if team_col != 'team':
        df = df.rename(columns={team_col: 'team'})

    # Normalize team names using existing mapping
    df['team'] = df['team'].apply(lambda x: normalize_team_name(str(x).strip()))

    return df


def load_power_ratings(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load power ratings data for a given provider, week, and season.

    Args:
        provider: Data provider ('nfelo', 'substack', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with power ratings data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("power_ratings", provider, week, season, data_dir)

    # Special handling for substack power ratings (has multi-row header)
    if provider == "substack":
        df = pd.read_csv(file_path, skiprows=[0])  # Skip first header row
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
    else:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

    df = _normalize_team_column_inplace(df)

    return df


def load_epa_tiers(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load EPA tiers data (offensive and defensive EPA per play).

    Args:
        provider: Data provider ('nfelo', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with EPA tiers data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("epa_tiers", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_strength_of_schedule(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load strength of schedule data.

    Args:
        provider: Data provider ('nfelo', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with strength of schedule data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("strength_of_schedule", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_qb_rankings(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load QB rankings data.

    Args:
        provider: Data provider ('nfelo', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with QB rankings data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("qb_rankings", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_qb_epa(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load QB EPA data.

    Args:
        provider: Data provider ('substack', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with QB EPA data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("qb_epa", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_weekly_projections_ppg(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load weekly points-per-game projections.

    Args:
        provider: Data provider ('substack', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with weekly PPG projections
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("weekly_projections_ppg", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_weekly_projections_elo(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load weekly Elo-based projections.

    Args:
        provider: Data provider ('substack', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with weekly Elo projections
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("weekly_projections_elo", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_win_totals(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load season win totals projections.

    Args:
        provider: Data provider ('nfelo', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with win totals data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("win_totals", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_receiving_leaders(
    provider: str,
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load receiving leaders data.

    Args:
        provider: Data provider ('nfelo', etc.)
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)

    Returns:
        DataFrame with receiving leaders data
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    file_path = _resolve_file_path("receiving_leaders", provider, week, season, data_dir)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = _normalize_team_column_inplace(df)

    return df


def load_all_sources(
    week: int,
    season: int = 2025,
    data_dir: Optional[Path] = None,
    providers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all available data sources for a given week and season.

    Attempts to load all combinations of categories and providers. If a file
    is not found, silently skips that combination. Also generates a merged
    team ratings table combining all available power ratings and EPA data.

    Args:
        week: NFL week number
        season: NFL season year (default: 2025)
        data_dir: Data directory (default: DEFAULT_DATA_DIR)
        providers: List of providers to load (default: VALID_PROVIDERS)

    Returns:
        Dictionary mapping data keys to DataFrames:
            - "{category}_{provider}" for each loaded file
            - "merged_ratings" for the unified team ratings table
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    if providers is None:
        providers = VALID_PROVIDERS

    data = {}

    # Define which categories each provider typically has
    # This avoids unnecessary FileNotFoundError exceptions
    provider_categories = {
        "nfelo": [
            "power_ratings",
            "epa_tiers",
            "strength_of_schedule",
            "qb_rankings",
            "win_totals",
            "receiving_leaders"
        ],
        "substack": [
            "power_ratings",
            "qb_epa",
            "weekly_projections_ppg",
            "weekly_projections_elo"
        ],
    }

    # Load all available data
    for provider in providers:
        categories = provider_categories.get(provider, VALID_CATEGORIES)

        for category in categories:
            key = f"{category}_{provider}"
            try:
                loader_func = globals()[f"load_{category}"]
                data[key] = loader_func(provider, week, season, data_dir)
            except FileNotFoundError:
                # File doesn't exist - skip silently
                continue
            except Exception as e:
                # Other error - warn but continue
                warnings.warn(
                    f"Error loading {category} from {provider}: {str(e)}",
                    UserWarning
                )
                continue

    # Generate merged ratings table
    if len(data) > 0:
        data["merged_ratings"] = merge_team_ratings(data)

    return data


def merge_team_ratings(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge team ratings from multiple sources into a unified table.

    Creates a single DataFrame with one row per team, combining power ratings,
    EPA metrics, and other team-level statistics from all available sources.

    Args:
        data: Dictionary of DataFrames from load_all_sources()

    Returns:
        DataFrame with merged team ratings (one row per team)

    Raises:
        ValueError: If required base data (power_ratings_nfelo) is not available
    """
    # Start with nfelo power ratings as base
    if "power_ratings_nfelo" not in data:
        raise ValueError(
            "Cannot merge team ratings: 'power_ratings_nfelo' is required as base"
        )

    merged = data["power_ratings_nfelo"][["team", "nfelo"]].copy()

    # Add EPA tiers from nfelo (offensive and defensive EPA)
    if "epa_tiers_nfelo" in data:
        epa_df = data["epa_tiers_nfelo"][["team", "EPA/Play", "EPA/Play Against"]].copy()
        epa_df = epa_df.rename(columns={
            "EPA/Play": "epa_off",
            "EPA/Play Against": "epa_def"
        })
        merged = merged.merge(epa_df, on="team", how="left")

    # Add substack power ratings
    if "power_ratings_substack" in data:
        substack_df = data["power_ratings_substack"][["team", "Ovr."]].copy()
        substack_df = substack_df.rename(columns={"Ovr.": "substack_power"})
        merged = merged.merge(substack_df, on="team", how="left")

    # Add strength of schedule from nfelo
    if "strength_of_schedule_nfelo" in data:
        # Get the actual columns from the file
        sos_df = data["strength_of_schedule_nfelo"]
        # Look for columns that might contain SOS data
        sos_cols = ["team"]
        for col in sos_df.columns:
            if "sos" in col.lower() or "strength" in col.lower():
                sos_cols.append(col)

        if len(sos_cols) > 1:
            sos_subset = sos_df[sos_cols].copy()
            merged = merged.merge(sos_subset, on="team", how="left")

    # Add win totals from nfelo
    if "win_totals_nfelo" in data:
        win_df = data["win_totals_nfelo"]
        # Look for projected wins column
        win_cols = ["team"]
        for col in win_df.columns:
            if "win" in col.lower() or "proj" in col.lower():
                win_cols.append(col)

        if len(win_cols) > 1:
            win_subset = win_df[win_cols].copy()
            merged = merged.merge(win_subset, on="team", how="left")

    return merged


# Sanity check / demo
if __name__ == "__main__":
    print("=" * 80)
    print("Ball Knower Unified Loaders - Sanity Check")
    print("=" * 80)
    print()

    # Test loading all sources for Week 11, 2025
    print(f"Loading all data sources for Week 11, 2025...")
    print(f"Data directory: {DEFAULT_DATA_DIR}")
    print()

    try:
        data = load_all_sources(week=11, season=2025)

        print(f"✓ Successfully loaded {len(data)} datasets:")
        for key in sorted(data.keys()):
            if key == "merged_ratings":
                continue
            df = data[key]
            print(f"  - {key:40s} ({len(df):3d} rows, {len(df.columns):2d} columns)")

        print()
        print("=" * 80)
        print("Merged Team Ratings (first 10 teams):")
        print("=" * 80)
        print()

        merged = data["merged_ratings"]
        print(merged.head(10).to_string())
        print()
        print(f"Total teams in merged ratings: {len(merged)}")
        print(f"Columns: {', '.join(merged.columns)}")

    except Exception as e:
        print(f"✗ Error during sanity check: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

    print()
    print("=" * 80)
    print("✓ Sanity check complete - all loaders working correctly!")
    print("=" * 80)
