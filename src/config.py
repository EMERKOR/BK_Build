"""
Configuration Module

Global constants and settings for Ball Knower.

IMPORTANT CHANGES (v1.2+):
- Week/season are now DYNAMIC - pass as function arguments, not from config
- Data file paths are handled by ball_knower.io.loaders (category-first naming)
- Provider-specific features replaced by canonical features (ball_knower.io.feature_maps)

This module now contains ONLY truly global constants that don't change with week/season.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CURRENT_SEASON_DIR = DATA_DIR / 'current_season'  # Default data directory
REFERENCE_DIR = DATA_DIR / 'reference'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SRC_DIR = PROJECT_ROOT / 'src'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA FILES (STATIC REFERENCE DATA ONLY)
# ============================================================================

# NOTE: Week-specific data files are now loaded via ball_knower.io.loaders
# Use loaders.load_all_sources(season=XXXX, week=YY) instead of these constants

# Reference files (not week-specific)
NFL_HEAD_COACHES = REFERENCE_DIR / 'nfl_head_coaches.csv'
NFL_AV_DATA = REFERENCE_DIR / 'nfl_AV_data_through_2024.xlsx'

# Output file templates (use .format() or f-strings for dynamic naming)
def get_output_path(filename):
    """Get output file path. Use this for dynamic output file naming."""
    return OUTPUT_DIR / filename

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Historical data range for training (defaults)
TRAINING_START_YEAR = 2015
TRAINING_END_YEAR = 2024
VALIDATION_YEAR = 2024

# Rolling window sizes for EPA features (must be leak-free)
EPA_ROLLING_WINDOWS = [3, 5, 10]  # games

# Home field advantage (points)
HOME_FIELD_ADVANTAGE = 2.5

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# NOTE: Provider-specific feature lists have been DEPRECATED.
# Use ball_knower.io.feature_maps for canonical, provider-agnostic features.
# See ball_knower.io.feature_maps.CANONICAL_FEATURE_MAP for available features.

# ============================================================================
# SPREAD CONVENTIONS
# ============================================================================
# All spreads are stored from HOME TEAM perspective
# Negative spread = home team favored
# Positive spread = home team underdog
# Example: Patriots vs Bills, home team (Bills) spread = -3 means Bills favored by 3

# ============================================================================
# VALIDATION & TESTING
# ============================================================================

# Leakage detection settings
STRICT_DATE_VALIDATION = True
VALIDATE_ROLLING_FEATURES = True

# Backtest settings
MIN_BET_EDGE = 0.5  # Minimum edge (in points) to recommend a bet
EDGE_BINS = [0.5, 1.0, 2.0, 3.0, 5.0]  # Edge bins for ROI analysis

# ============================================================================
# GOOGLE COLAB COMPATIBILITY
# ============================================================================

def setup_colab_paths():
    """
    Adjust paths for Google Colab environment with mounted Drive.
    Call this function at the start of the notebook if running in Colab.
    """
    global DATA_DIR, CURRENT_SEASON_DIR, REFERENCE_DIR, OUTPUT_DIR

    DRIVE_ROOT = Path('/content/drive/MyDrive/Ball_Knower')

    if DRIVE_ROOT.exists():
        DATA_DIR = DRIVE_ROOT
        CURRENT_SEASON_DIR = DRIVE_ROOT / '5_current_season'
        REFERENCE_DIR = DRIVE_ROOT  # Reference files in root
        OUTPUT_DIR = DRIVE_ROOT / 'output'
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"✓ Google Drive paths configured: {DRIVE_ROOT}")
    else:
        print("✗ Google Drive not found. Using local paths.")


def get_config_summary(season=None, week=None):
    """
    Print configuration summary for verification.

    Args:
        season (int, optional): Season to display in summary
        week (int, optional): Week to display in summary

    Returns:
        str: Formatted configuration summary
    """
    season_info = f"{season} Week {week}" if season and week else "Dynamic (pass as arguments)"

    summary = f"""
    ═══════════════════════════════════════════════════════════
    BALL KNOWER - CONFIGURATION SUMMARY
    ═══════════════════════════════════════════════════════════

    PROJECT ROOT:     {PROJECT_ROOT}
    DATA DIRECTORY:   {DATA_DIR}
    OUTPUT DIRECTORY: {OUTPUT_DIR}

    TRAINING PERIOD:  {TRAINING_START_YEAR}-{TRAINING_END_YEAR}
    SEASON/WEEK:      {season_info}

    DATA FILES:       {len([f for f in Path(CURRENT_SEASON_DIR).glob('*.csv')])} CSV files found

    HOME FIELD ADV:   {HOME_FIELD_ADVANTAGE} points
    MIN BET EDGE:     {MIN_BET_EDGE} points

    NOTE: Use ball_knower.io.loaders.load_all_sources(season, week)
          to load week-specific data dynamically.

    ═══════════════════════════════════════════════════════════
    """
    return summary
