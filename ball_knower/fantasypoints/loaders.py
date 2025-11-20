"""
FantasyPoints Data Loaders

This module provides loading and normalization functions for FantasyPoints CSV exports.

All loaders:
- Clean column names to snake_case
- Normalize team names to BK canonical codes
- Ensure season column exists
- Return tidy DataFrames ready for aggregation
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FPD_DATA_DIR = PROJECT_ROOT / "data" / "fantasypoints"
TEAM_MAP_PATH = PROJECT_ROOT / "config" / "team_name_map_fpd.csv"

# Load team name mapping
_TEAM_MAPPING: Optional[Dict[str, str]] = None


def _load_team_mapping() -> Dict[str, str]:
    """Load FantasyPoints team name to BK canonical team code mapping."""
    global _TEAM_MAPPING
    if _TEAM_MAPPING is None:
        df = pd.read_csv(TEAM_MAP_PATH)
        _TEAM_MAPPING = dict(zip(df['fpd_name'], df['bk_team']))
    return _TEAM_MAPPING


def _normalize_team_name(team_name: str) -> str:
    """
    Normalize a FantasyPoints team name to BK canonical code.

    Args:
        team_name: Team name from FantasyPoints export

    Returns:
        BK canonical team code (e.g., 'KC', 'LAR', 'BUF')

    Raises:
        ValueError: If team name cannot be mapped
    """
    if pd.isna(team_name):
        raise ValueError("Team name is NaN")

    team_name = str(team_name).strip()

    # Handle comma-separated team names (e.g., "CIN, CLV") - take the first
    if ',' in team_name:
        team_name = team_name.split(',')[0].strip()

    mapping = _load_team_mapping()

    if team_name in mapping:
        return mapping[team_name]

    raise ValueError(f"Unknown FantasyPoints team name: '{team_name}'")


def _clean_column_name(col: str) -> str:
    """
    Convert column name to snake_case.

    Args:
        col: Original column name

    Returns:
        Cleaned snake_case column name
    """
    # Remove special characters and convert to lowercase
    col = str(col).strip()

    # Replace spaces and slashes with underscores
    col = re.sub(r'[\s/\-]+', '_', col)

    # Remove parentheses and their contents
    col = re.sub(r'\([^)]*\)', '', col)

    # Remove percentage signs and other special chars
    col = re.sub(r'[%+]', '', col)

    # Convert to lowercase
    col = col.lower()

    # Remove multiple underscores
    col = re.sub(r'_+', '_', col)

    # Remove leading/trailing underscores
    col = col.strip('_')

    return col


def _clean_dataframe(df: pd.DataFrame, team_col: str = 'team_name') -> pd.DataFrame:
    """
    Clean DataFrame: normalize column names and team names.

    Args:
        df: Raw DataFrame from CSV
        team_col: Name of the team column to normalize

    Returns:
        Cleaned DataFrame with snake_case columns and normalized team names
    """
    # Clean column names
    cleaned_cols = [_clean_column_name(col) for col in df.columns]

    # Handle duplicate column names by appending suffix
    seen = {}
    final_cols = []
    for col in cleaned_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)

    df.columns = final_cols

    # Normalize team names if team column exists
    if team_col in df.columns:
        # Filter out rows where team_col is NaN
        df = df[df[team_col].notna()].copy()
        df['team'] = df[team_col].apply(_normalize_team_name)
        df = df.drop(columns=[team_col])
    elif 'team' in df.columns:
        # Already has 'team' column, filter NaN and normalize
        df = df[df['team'].notna()].copy()
        df['team'] = df['team'].apply(_normalize_team_name)

    return df


def load_coverage_matrix(season: int = 2025) -> pd.DataFrame:
    """
    Load defensive coverage matrix data.

    Contains defensive coverage percentages (man, zone, cover 0-6, etc.)

    Args:
        season: NFL season year

    Returns:
        DataFrame with columns:
            - season, team
            - g (games played)
            - man_pct, zone_pct
            - cover_0_pct, cover_1_pct, cover_2_pct, cover_2_man_pct
            - cover_3_pct, cover_4_pct, cover_6_pct
            - mof_closed_pct, mof_open_pct (middle of field)
    """
    path = FPD_DATA_DIR / "coverageMatrixExport.csv"

    # Skip the first header row (merged cells)
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    # Select relevant columns
    cols_to_keep = [
        'season', 'team', 'g',
        'man_pct', 'zone_pct',
        '1_hi_mof_c_pct', '2_hi_mof_o_pct',
        'cover_0_pct', 'cover_1_pct', 'cover_2_pct', 'cover_2_man_pct',
        'cover_3_pct', 'cover_4_pct', 'cover_6_pct'
    ]

    # Rename for clarity
    rename_map = {
        '1_hi_mof_c_pct': 'mof_closed_pct',
        '2_hi_mof_o_pct': 'mof_open_pct'
    }
    df = df.rename(columns=rename_map)

    # Filter to available columns
    available_cols = [c for c in cols_to_keep if c in df.columns or c in rename_map.values()]

    return df[available_cols]


def load_qb_coverage_matchup(season: int = 2025) -> pd.DataFrame:
    """
    Load QB coverage matchup data (QB vs different coverage types).

    Args:
        season: NFL season year

    Returns:
        DataFrame with team-level QB performance vs coverage types
    """
    path = FPD_DATA_DIR / "qbCoverageMatchupExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_wr_coverage_matchup(season: int = 2025) -> pd.DataFrame:
    """
    Load WR coverage matchup data (WR vs different coverage types).

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level WR performance vs coverage types
    """
    path = FPD_DATA_DIR / "wrCoverageMatchupExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_receiving_man_vs_zone(season: int = 2025) -> pd.DataFrame:
    """
    Load receiving performance vs man and zone coverage.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level receiving splits vs man/zone
    """
    path = FPD_DATA_DIR / "receivingManVsZoneExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_receiving_separation_by_coverage(season: int = 2025) -> pd.DataFrame:
    """
    Load receiving separation metrics by coverage type.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level separation by coverage
    """
    path = FPD_DATA_DIR / "receivingSeparationByCoverageExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_receiving_routes_run(season: int = 2025) -> pd.DataFrame:
    """
    Load receiving routes run data.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level route running metrics
    """
    path = FPD_DATA_DIR / "receivingRoutesRunExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_receiving_separation_by_alignment(season: int = 2025) -> pd.DataFrame:
    """
    Load receiving separation by alignment.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level separation by alignment
    """
    path = FPD_DATA_DIR / "receivingSeparationByAlignmentExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_receiving_separation_by_breaks(season: int = 2025) -> pd.DataFrame:
    """
    Load receiving separation by break types.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level separation by breaks
    """
    path = FPD_DATA_DIR / "receivingSeparationByBreaksExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_receiving_advanced(season: int = 2025) -> pd.DataFrame:
    """
    Load advanced receiving metrics.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level advanced receiving stats
    """
    path = FPD_DATA_DIR / "receivingAdvancedExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_run_pass_report(season: int = 2025) -> pd.DataFrame:
    """
    Load run-pass tendency report (offensive identity).

    Args:
        season: NFL season year

    Returns:
        DataFrame with team-level run/pass tendencies by situation
    """
    path = FPD_DATA_DIR / "runPassReportExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_line_matchups(season: int = 2025) -> pd.DataFrame:
    """
    Load offensive/defensive line matchup data.

    Args:
        season: NFL season year

    Returns:
        DataFrame with team-level OL/DL metrics
    """
    path = FPD_DATA_DIR / "lineMatchupsExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_rushing_advanced(season: int = 2025) -> pd.DataFrame:
    """
    Load advanced rushing metrics.

    Args:
        season: NFL season year

    Returns:
        DataFrame with player-level advanced rushing stats
    """
    path = FPD_DATA_DIR / "rushingAdvancedExport.csv"

    # Skip the first header row
    df = pd.read_csv(path, skiprows=1)

    # Clean and normalize
    df = _clean_dataframe(df)

    # Add season
    df['season'] = season

    return df


def load_all_tables(season: int = 2025) -> Dict[str, pd.DataFrame]:
    """
    Load all FantasyPoints tables for a given season.

    Args:
        season: NFL season year

    Returns:
        Dictionary mapping table names to DataFrames
    """
    return {
        'coverage_matrix': load_coverage_matrix(season),
        'qb_coverage_matchup': load_qb_coverage_matchup(season),
        'wr_coverage_matchup': load_wr_coverage_matchup(season),
        'receiving_man_vs_zone': load_receiving_man_vs_zone(season),
        'receiving_separation_by_coverage': load_receiving_separation_by_coverage(season),
        'receiving_routes_run': load_receiving_routes_run(season),
        'receiving_separation_by_alignment': load_receiving_separation_by_alignment(season),
        'receiving_separation_by_breaks': load_receiving_separation_by_breaks(season),
        'receiving_advanced': load_receiving_advanced(season),
        'run_pass_report': load_run_pass_report(season),
        'line_matchups': load_line_matchups(season),
        'rushing_advanced': load_rushing_advanced(season),
    }
