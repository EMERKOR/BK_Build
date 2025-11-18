"""
Unified Data Loader Module

This is the canonical, future-facing loader API for Ball Knower.
Defaults to category-first naming convention, with provider-first fallback for compatibility.

All new code should import from here:
    from ball_knower.io import loaders
    data = loaders.load_all_sources(season=2025, week=11)

File naming conventions:
    Primary (category-first): {category}_{provider}_{season}_week_{week}.csv
    Legacy (provider-first): {provider}_{category}_{season}_week_{week}.csv

Supported categories:
    - power_ratings
    - epa_tiers
    - strength_of_schedule
    - qb_epa
    - qb_rankings
    - weekly_projections_ppg
    - weekly_projections_elo
    - nfl_receiving_leaders
    - nfl_win_totals

Supported providers:
    - nfelo (nfeloapp.com)
    - 538 (FiveThirtyEight)
    - espn (ESPN Analytics)
    - pff (Pro Football Focus)
    - gsis (NFL Game Statistics & Info System)
    - user (custom user uploads)
    - manual (hand-curated datasets)
    - substack (legacy - being migrated to specific providers above)

Note: 'substack' is being phased out as a provider name in favor of specific
provider identification (nfelo, 538, etc.). Files with 'substack' in the name
are supported for backward compatibility.
"""

from pathlib import Path
import warnings
import pandas as pd
from typing import Dict, Optional, Union
import sys
import importlib.util

# Import team normalization function directly (avoid importing models.py dependencies via src/__init__.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Direct import of team_mapping module without going through src/__init__.py
_team_mapping_path = _PROJECT_ROOT / "src" / "team_mapping.py"
_spec = importlib.util.spec_from_file_location("team_mapping", _team_mapping_path)
_team_mapping = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_team_mapping)
normalize_team = _team_mapping.normalize_team_name

# Default data directory: repo_root/data/current_season
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "current_season"


def _normalize_team_column(df: pd.DataFrame, team_col: str = "team") -> pd.DataFrame:
    """
    Standardize team names to canonical nfl_data_py abbreviations.

    Args:
        df: DataFrame with a team column
        team_col: Name of the team column (default: "team")

    Returns:
        DataFrame with normalized team names

    Raises:
        ValueError: If team_col is not in the DataFrame
    """
    if team_col not in df.columns:
        raise ValueError(f"Expected a '{team_col}' column in dataframe. Found: {list(df.columns)}")

    df = df.copy()
    df[team_col] = df[team_col].map(normalize_team)

    # Check for any unmapped teams
    unmapped = df[df[team_col].isna()]
    if len(unmapped) > 0:
        warnings.warn(
            f"Found {len(unmapped)} rows with unmapped team names. These will be dropped.",
            UserWarning
        )
        df = df.dropna(subset=[team_col])

    return df


def _resolve_file(
    category: str,
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve file path with category-first primary and provider-first fallback.

    Args:
        category: Data category (e.g., "power_ratings", "epa_tiers")
        provider: Data provider (e.g., "nfelo", "substack")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory (defaults to repo/data/current_season)

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If no matching file is found
    """
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    # Primary: category-first
    primary = base_dir / f"{category}_{provider}_{season}_week_{week}.csv"

    # Legacy/provider-first fallback patterns
    # Also handle common abbreviations (proj vs projections)
    category_short = category.replace("_projections_", "_proj_")

    legacy_candidates = [
        base_dir / f"{provider}_{category}_{season}_week_{week}.csv",
        base_dir / f"{provider}_{category}_off_def_{season}_week_{week}.csv",
        base_dir / f"{provider}_{category_short}_{season}_week_{week}.csv",
    ]

    if primary.exists():
        return primary

    for cand in legacy_candidates:
        if cand.exists():
            warnings.warn(
                f"Using legacy filename for {category}/{provider}: {cand.name}. "
                f"Consider renaming to: {primary.name}",
                UserWarning,
            )
            return cand

    raise FileNotFoundError(
        f"Could not find file for category='{category}', provider='{provider}', "
        f"season={season}, week={week} in {base_dir}. Tried:\n"
        f"  - {primary.name}\n"
        f"  - {legacy_candidates[0].name}\n"
        f"  - {legacy_candidates[1].name}"
    )


def load_power_ratings(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load power ratings for a given provider, season, and week.

    Args:
        provider: Data provider (e.g., "nfelo", "538", "espn", "pff", "substack")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with normalized 'team' column and provider-specific ratings
    """
    path = _resolve_file("power_ratings", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers (Substack files have this)
    # Check if first row is all NaN or if column names look like junk (X.1, X.2, etc.)
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts (X.1, X.2, etc.)
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Rename 'Team' to 'team' if it exists, otherwise use first column
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})
    elif len(df.columns) > 0:
        # First non-X column is likely team
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'team'})

    return _normalize_team_column(df, team_col="team")


def load_epa_tiers(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load EPA tiers (offensive/defensive EPA per play).

    Args:
        provider: Data provider (e.g., "nfelo", "gsis")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with normalized 'team' column and EPA metrics
    """
    path = _resolve_file("epa_tiers", provider, season, week, data_dir)
    df = pd.read_csv(path)

    if df.iloc[0].isna().all() or df.iloc[0, 0] == df.columns[0]:
        df = pd.read_csv(path, skiprows=1)

    # Rename 'Team' to 'team' if it exists
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})

    return _normalize_team_column(df, team_col="team")


def load_strength_of_schedule(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load strength of schedule metrics.

    Args:
        provider: Data provider (e.g., "nfelo", "538")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with normalized 'team' column and SOS metrics
    """
    path = _resolve_file("strength_of_schedule", provider, season, week, data_dir)
    df = pd.read_csv(path)

    if df.iloc[0].isna().all() or df.iloc[0, 0] == df.columns[0]:
        df = pd.read_csv(path, skiprows=1)

    # Rename 'Team' to 'team' if it exists
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})

    return _normalize_team_column(df, team_col="team")


def load_qb_epa(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load quarterback EPA metrics.

    Args:
        provider: Data provider (e.g., "nfelo", "pff", "gsis", "substack")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with normalized 'team' column and QB EPA metrics
    """
    path = _resolve_file("qb_epa", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts (X.1, X.2, etc.)
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Handle Substack QB EPA which uses 'Tms' column with lowercase team codes
    if 'Tms' in df.columns:
        # Some QBs have multiple teams (e.g., "cle, cin") - take the first team
        df['Tms'] = df['Tms'].astype(str).str.split(',').str[0].str.strip()
        df = df.rename(columns={'Tms': 'team'})
    elif 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})
    elif len(df.columns) > 0:
        # First non-X column might be team
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'team'})

    return _normalize_team_column(df, team_col="team")


def load_weekly_projections_ppg(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load weekly projections (points per game, spreads, etc.).

    Args:
        provider: Data provider (e.g., "nfelo", "538", "substack")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with team column and projection metrics
        Note: This may contain matchup data rather than team-level data
    """
    path = _resolve_file("weekly_projections_ppg", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts (X.1, X.2, etc.)
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Weekly projections may not have a single team column
    # For now, add a placeholder 'team' column for consistency
    # Individual providers can handle this differently
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})
        return _normalize_team_column(df, team_col="team")
    else:
        # Return as-is for matchup-based projections
        # Add a dummy 'team' column to avoid errors
        df['team'] = None
        return df


def load_weekly_projections_elo(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load weekly ELO-based projections.

    Args:
        provider: Data provider (e.g., "nfelo", "538")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with ELO projections and win probabilities
    """
    path = _resolve_file("weekly_projections_elo", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    return df


def load_qb_rankings(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load QB rankings and statistics.

    Args:
        provider: Data provider (e.g., "nfelo", "pff")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with QB rankings and stats
    """
    path = _resolve_file("qb_rankings", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Rename 'Team' to 'team' if it exists
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})

    # QB rankings may have a team column - normalize if present
    if 'team' in df.columns:
        return _normalize_team_column(df, team_col="team")

    return df


def load_nfl_receiving_leaders(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load NFL receiving leaders statistics.

    Args:
        provider: Data provider (e.g., "nfelo", "gsis")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with receiving statistics
    """
    path = _resolve_file("nfl_receiving_leaders", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Rename 'Team' to 'team' if it exists
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})

    # Normalize team column if present
    if 'team' in df.columns:
        return _normalize_team_column(df, team_col="team")

    return df


def load_nfl_win_totals(
    provider: str,
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load NFL season win total projections.

    Args:
        provider: Data provider (e.g., "nfelo", "538")
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        DataFrame with season win projections
    """
    path = _resolve_file("nfl_win_totals", provider, season, week, data_dir)
    df = pd.read_csv(path)

    # Handle potential multi-row headers
    if (df.iloc[0].isna().all() or
        df.iloc[0, 0] == df.columns[0] or
        any(col.startswith('X.') for col in df.columns[:3])):
        df = pd.read_csv(path, skiprows=1)

    # Remove weird header artifacts
    df = df.loc[:, ~df.columns.str.startswith('X.')]

    # Rename 'Team' to 'team' if it exists
    if 'Team' in df.columns:
        df = df.rename(columns={'Team': 'team'})

    return _normalize_team_column(df, team_col="team")


def discover_available_sources(
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None
) -> Dict[str, list]:
    """
    Auto-detect available data sources in the data directory.

    Scans the data directory for files matching the naming convention and
    returns a dictionary of available categories and their providers.

    Args:
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory (defaults to repo/data/current_season)

    Returns:
        Dictionary mapping category → list of available providers

    Example:
        >>> sources = discover_available_sources(2025, 11)
        >>> print(sources)
        {
            'power_ratings': ['nfelo', 'substack'],
            'epa_tiers': ['nfelo'],
            'qb_epa': ['nfelo', 'substack'],
            ...
        }
    """
    import re

    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    # Categories we're looking for
    categories = [
        "power_ratings",
        "epa_tiers",
        "strength_of_schedule",
        "qb_epa",
        "qb_rankings",
        "weekly_projections_ppg",
        "weekly_projections_elo",
        "nfl_receiving_leaders",
        "nfl_win_totals",
    ]

    # Providers we might find
    providers = ["nfelo", "538", "espn", "pff", "gsis", "user", "manual", "substack"]

    available = {}

    # Scan for category-first files
    for csv_file in base_dir.glob(f"*_{season}_week_{week}.csv"):
        filename = csv_file.stem  # Remove .csv extension

        # Try to parse category-first: {category}_{provider}_{season}_week_{week}
        pattern = rf"^(.+?)_({'|'.join(providers)})_{season}_week_{week}$"
        match = re.match(pattern, filename)

        if match:
            category, provider = match.groups()
            if category in categories:
                if category not in available:
                    available[category] = []
                if provider not in available[category]:
                    available[category].append(provider)
                continue

        # Try to parse provider-first: {provider}_{category}_{season}_week_{week}
        pattern = rf"^({'|'.join(providers)})_(.+?)_{season}_week_{week}$"
        match = re.match(pattern, filename)

        if match:
            provider, category = match.groups()
            # Normalize category (e.g., "weekly_proj_elo" → "weekly_projections_elo")
            for canonical_category in categories:
                if category in canonical_category or canonical_category in category:
                    if canonical_category not in available:
                        available[canonical_category] = []
                    if provider not in available[canonical_category]:
                        available[canonical_category].append(provider)
                    break

    return available


def merge_team_ratings(
    sources: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all team-level rating tables into a single frame keyed by 'team'.

    Expected keys in `sources`:
        - 'power_ratings_nfelo'
        - 'epa_tiers_nfelo'
        - 'strength_of_schedule_nfelo'
        - 'power_ratings_substack'
        - 'qb_epa_substack'
        - 'weekly_projections_ppg_substack'

    Args:
        sources: Dictionary of DataFrames from load_all_sources()

    Returns:
        Merged DataFrame with all ratings, indexed by 'team'
    """
    # Start from nfelo power ratings as the base
    if "power_ratings_nfelo" not in sources:
        raise ValueError("Expected 'power_ratings_nfelo' in sources dict")

    base = sources["power_ratings_nfelo"].copy()

    # Define merge order and suffixes
    merge_specs = [
        ("epa_tiers_nfelo", "_epa"),
        ("strength_of_schedule_nfelo", "_sos"),
        ("power_ratings_substack", "_substack"),
        ("qb_epa_substack", "_qb_epa"),
        ("weekly_projections_ppg_substack", "_proj_ppg"),
    ]

    for key, suffix in merge_specs:
        if key not in sources:
            warnings.warn(f"Missing expected source: {key}. Skipping merge.", UserWarning)
            continue

        df = sources[key]
        if "team" not in df.columns:
            warnings.warn(f"DataFrame '{key}' missing 'team' column. Skipping merge.", UserWarning)
            continue

        # Left merge on team
        base = base.merge(
            df,
            on="team",
            how="left",
            suffixes=("", suffix)
        )

    return base


def load_all_sources(
    season: int,
    week: int,
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all data sources for a given season and week.

    Orchestrator function that loads:
        - nfelo: power ratings, epa tiers, strength of schedule
        - substack: power ratings, qb epa, weekly projections ppg

    Args:
        season: NFL season year
        week: NFL week number
        data_dir: Optional data directory

    Returns:
        Dictionary with keys:
            - "power_ratings_nfelo": nfelo power ratings
            - "epa_tiers_nfelo": nfelo EPA tiers
            - "strength_of_schedule_nfelo": nfelo SOS
            - "power_ratings_substack": Substack power ratings
            - "qb_epa_substack": Substack QB EPA
            - "weekly_projections_ppg_substack": Substack weekly projections
            - "merged_ratings": All ratings merged on 'team'

    Example:
        >>> data = load_all_sources(season=2025, week=11)
        >>> merged = data["merged_ratings"]
        >>> print(merged.head())
    """
    result = {}

    # Load nfelo sources
    try:
        result["power_ratings_nfelo"] = load_power_ratings("nfelo", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load nfelo power ratings: {e}", UserWarning)

    try:
        result["epa_tiers_nfelo"] = load_epa_tiers("nfelo", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load nfelo EPA tiers: {e}", UserWarning)

    try:
        result["strength_of_schedule_nfelo"] = load_strength_of_schedule("nfelo", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load nfelo strength of schedule: {e}", UserWarning)

    # Load Substack sources
    try:
        result["power_ratings_substack"] = load_power_ratings("substack", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load Substack power ratings: {e}", UserWarning)

    try:
        result["qb_epa_substack"] = load_qb_epa("substack", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load Substack QB EPA: {e}", UserWarning)

    try:
        result["weekly_projections_ppg_substack"] = load_weekly_projections_ppg("substack", season, week, data_dir)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load Substack weekly projections: {e}", UserWarning)

    # Merge all ratings
    if "power_ratings_nfelo" in result:
        try:
            result["merged_ratings"] = merge_team_ratings(result)
        except Exception as e:
            warnings.warn(f"Could not merge ratings: {e}", UserWarning)
    else:
        warnings.warn("Cannot create merged_ratings without nfelo power ratings as base", UserWarning)

    return result
