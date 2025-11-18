"""
Ball Knower v1.2 Dataset Builder

Canonical training dataset for v1.2 model.
Loads historical nfelo data and constructs one-row-per-game training frame.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings


def build_training_frame(
    data_dir: Optional[Path] = None,
    start_season: int = 2009,
    end_season: int = 2024,
) -> pd.DataFrame:
    """
    Build the canonical v1.2 training dataset.

    Loads historical nfelo game data and constructs a clean training frame
    with one row per game containing:
      - Identifiers (season, week, teams, game_id)
      - Outcomes (final scores, margins, results)
      - Market info (Vegas closing spread)
      - Base Ball_Knower inputs (nfelo ratings, situational factors)
      - Derived feature columns (rating diffs, structural factors)

    Args:
        data_dir: Optional data directory (not currently used; loads from nfelo URL)
        start_season: First season to include (default: 2009)
        end_season: Last season to include (default: 2024)

    Returns:
        DataFrame with training data, one row per game

    Example:
        >>> df = build_training_frame(start_season=2015, end_season=2024)
        >>> print(df.shape)
        >>> print(df.columns.tolist())
    """

    print(f"\nBuilding v1.2 training frame ({start_season}-{end_season})...")

    # ========================================================================
    # LOAD HISTORICAL DATA
    # ========================================================================

    print("  [1/4] Loading nfelo historical data from GitHub...")

    # Load nfelo historical games (this is the canonical source for v1.2)
    nfelo_url = 'https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv'
    df = pd.read_csv(nfelo_url)

    print(f"    Loaded {len(df):,} raw games")

    # ========================================================================
    # PARSE IDENTIFIERS
    # ========================================================================

    print("  [2/4] Parsing game identifiers...")

    # Extract season/week/teams from game_id (format: YYYY_WW_AWAY_HOME)
    df[['season', 'week', 'away_team', 'home_team']] = df['game_id'].str.extract(
        r'(\d{4})_(\d+)_(\w+)_(\w+)'
    )
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)

    # Filter to requested season range
    df = df[(df['season'] >= start_season) & (df['season'] <= end_season)].copy()

    print(f"    Filtered to {len(df):,} games in {start_season}-{end_season}")

    # ========================================================================
    # FILTER TO COMPLETE DATA
    # ========================================================================

    print("  [3/4] Filtering to complete games...")

    # Require Vegas closing line (this is our target)
    df = df[df['home_line_close'].notna()].copy()

    # Require starting nfelo ratings (these are our base features)
    df = df[df['starting_nfelo_home'].notna()].copy()
    df = df[df['starting_nfelo_away'].notna()].copy()

    print(f"    {len(df):,} games with complete data")

    # ========================================================================
    # CONSTRUCT FEATURES
    # ========================================================================

    print("  [4/4] Engineering features...")

    # --- Identifiers ---
    # (already extracted: season, week, away_team, home_team, game_id)

    # --- Outcomes / Targets ---
    # NOTE: nfelo data doesn't include actual game scores
    # For actual outcomes, would need to merge with nflverse data
    # For now, v1.2 trains on Vegas lines only

    # Vegas closing spread (home team perspective; negative = home favored)
    df['vegas_closing_spread'] = df['home_line_close']

    # Placeholders for future: actual game outcomes (when merged with nflverse)
    df['home_points'] = pd.NA
    df['away_points'] = pd.NA
    df['home_margin'] = pd.NA

    # --- Base "football" prediction fields ---
    # Raw nfelo power ratings (these feed into our base spread prediction)
    df['nfelo_power_home'] = df['starting_nfelo_home']
    df['nfelo_power_away'] = df['starting_nfelo_away']

    # QB adjustments from 538
    df['qb_adj_home'] = df['home_538_qb_adj'].fillna(0)
    df['qb_adj_away'] = df['away_538_qb_adj'].fillna(0)

    # Situational modifiers from nfelo
    df['home_bye_mod'] = df['home_bye_mod'].fillna(0)
    df['away_bye_mod'] = df['away_bye_mod'].fillna(0)
    df['div_game_mod'] = df['div_game_mod'].fillna(0)
    df['surface_mod'] = df['dif_surface_mod'].fillna(0)
    df['time_advantage_mod'] = df['home_time_advantage_mod'].fillna(0)

    # --- Feature fields (rating differences and structural factors) ---
    # Primary feature: ELO differential
    df['diff_nfelo_power'] = df['nfelo_power_home'] - df['nfelo_power_away']

    # Combined rest advantage
    df['rest_advantage'] = df['home_bye_mod'] + df['away_bye_mod']

    # Divisional game flag
    df['div_game'] = df['div_game_mod']

    # Surface difference
    df['surface_diff'] = df['surface_mod']

    # Time zone advantage
    df['time_advantage'] = df['time_advantage_mod']

    # QB quality differential
    df['diff_qb_adj'] = df['qb_adj_home'] - df['qb_adj_away']

    # Neutral site flag (if available in nfelo data)
    df['neutral_site'] = df.get('neutral', 0)

    # --- Base Ball_Knower spread (for reference) ---
    # This is what the model "thinks" before calibration
    # We can compute this from nfelo_diff + modifiers, but for now just store the components
    # The actual formula will be learned during training
    df['bk_base_spread'] = df['diff_nfelo_power']  # Simplest version

    # ========================================================================
    # SELECT FINAL COLUMNS
    # ========================================================================

    output_cols = [
        # Identifiers
        'game_id',
        'season',
        'week',
        'home_team',
        'away_team',

        # Outcomes / Targets
        'home_points',
        'away_points',
        'home_margin',
        'vegas_closing_spread',

        # Base prediction fields (raw ratings)
        'nfelo_power_home',
        'nfelo_power_away',
        'qb_adj_home',
        'qb_adj_away',
        'home_bye_mod',
        'away_bye_mod',
        'div_game_mod',
        'surface_mod',
        'time_advantage_mod',

        # Feature fields (diffs and flags)
        'diff_nfelo_power',
        'rest_advantage',
        'div_game',
        'surface_diff',
        'time_advantage',
        'diff_qb_adj',
        'neutral_site',
        'bk_base_spread',
    ]

    # Filter to columns that exist
    available_cols = [col for col in output_cols if col in df.columns]
    df_final = df[available_cols].copy()

    # Remove any rows with NaN in core features
    core_features = [
        'diff_nfelo_power',
        'rest_advantage',
        'div_game',
        'surface_diff',
        'time_advantage',
        'diff_qb_adj',
        'vegas_closing_spread'
    ]

    mask = df_final[core_features].notna().all(axis=1)
    df_final = df_final[mask].reset_index(drop=True)

    print(f"  ✓ Built training frame: {len(df_final):,} games x {len(df_final.columns)} features")

    return df_final


def save_training_frame(
    output_path: Path,
    data_dir: Optional[Path] = None,
    start_season: int = 2009,
    end_season: int = 2024,
) -> Path:
    """
    Convenience wrapper that builds and saves the v1.2 training dataset.

    Args:
        output_path: Path where to save the Parquet file
        data_dir: Optional data directory (passed to build_training_frame)
        start_season: First season to include
        end_season: Last season to include

    Returns:
        Path to the saved file

    Example:
        >>> from pathlib import Path
        >>> output = Path("data/v1_2_training.parquet")
        >>> saved_path = save_training_frame(output)
        >>> print(f"Saved to {saved_path}")
    """

    # Build the training frame
    df = build_training_frame(
        data_dir=data_dir,
        start_season=start_season,
        end_season=end_season
    )

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Parquet (efficient, preserves types)
    df.to_parquet(output_path, index=False, engine='pyarrow')

    print(f"\n✓ Saved v1.2 training dataset to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    return output_path
