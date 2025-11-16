"""
Configuration Module

Single source of truth for all paths, constants, and settings.
NO DUPLICATE CONFIGS ALLOWED.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CURRENT_SEASON_DIR = DATA_DIR / 'current_season'
REFERENCE_DIR = DATA_DIR / 'reference'
OUTPUT_DIR = PROJECT_ROOT / 'output'
SRC_DIR = PROJECT_ROOT / 'src'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

# nfelo data files (Week 11, 2025)
NFELO_POWER_RATINGS = CURRENT_SEASON_DIR / 'nfelo_power_ratings_2025_week_11.csv'
NFELO_SOS = CURRENT_SEASON_DIR / 'nfelo_strength_of_schedule_2025_week_11.csv'
NFELO_EPA_TIERS = CURRENT_SEASON_DIR / 'nfelo_epa_tiers_off_def_2025_week_11.csv'
NFELO_WIN_TOTALS = CURRENT_SEASON_DIR / 'nfelo_nfl_win_totals_2025_week_11 (1).csv'
NFELO_RECEIVING = CURRENT_SEASON_DIR / 'nfelo_nfl_receiving_leaders_2025_week_11.csv'
NFELO_QB_RANKINGS = CURRENT_SEASON_DIR / 'nfelo_qb_rankings_2025_week_11.csv'

# Substack data files (Week 11, 2025)
SUBSTACK_POWER_RATINGS = CURRENT_SEASON_DIR / 'substack_power_ratings_2025_week_11.csv'
SUBSTACK_QB_EPA = CURRENT_SEASON_DIR / 'substack_qb_epa_2025_week_11.csv'
SUBSTACK_WEEKLY_PROJ_ELO = CURRENT_SEASON_DIR / 'substack_weekly_proj_elo_2025_week_11.csv'
SUBSTACK_WEEKLY_PROJ_PPG = CURRENT_SEASON_DIR / 'substack_weekly_proj_ppg_2025_week_11.csv'

# Reference files
NFL_HEAD_COACHES = REFERENCE_DIR / 'nfl_head_coaches.csv'
NFL_AV_DATA = REFERENCE_DIR / 'nfl_AV_data_through_2024.xlsx'

# Output files
BACKTEST_RESULTS = OUTPUT_DIR / 'backtest_results.csv'
WEEKLY_PREDICTIONS = OUTPUT_DIR / 'weekly_predictions.csv'
MODEL_DIAGNOSTICS = OUTPUT_DIR / 'model_diagnostics.csv'

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Historical data range for training
TRAINING_START_YEAR = 2015
TRAINING_END_YEAR = 2024
VALIDATION_YEAR = 2024

# Current season info
CURRENT_SEASON = 2025
CURRENT_WEEK = 11

# Rolling window sizes for EPA features (must be leak-free)
EPA_ROLLING_WINDOWS = [3, 5, 10]  # games

# Home field advantage (points)
HOME_FIELD_ADVANTAGE = 2.5

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Columns to use from nfelo
NFELO_FEATURES = [
    'nfelo',           # Main ELO rating
    'QB Adj',          # QB adjustment
    'Value',           # Overall value
    'WoW',             # Week over week change
    'YTD',             # Year to date performance
]

# Columns to use from Substack
SUBSTACK_FEATURES = [
    'Off.',            # Offensive rating
    'Def.',            # Defensive rating
    'Ovr.',            # Overall rating
]

# EPA features to engineer
EPA_FEATURES = [
    'epa_off',         # Offensive EPA per play
    'epa_def',         # Defensive EPA per play
    'epa_margin',      # EPA differential
]

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


def get_config_summary():
    """Print configuration summary for verification."""
    summary = f"""
    ═══════════════════════════════════════════════════════════
    BALL KNOWER - CONFIGURATION SUMMARY
    ═══════════════════════════════════════════════════════════

    PROJECT ROOT:     {PROJECT_ROOT}
    DATA DIRECTORY:   {DATA_DIR}
    OUTPUT DIRECTORY: {OUTPUT_DIR}

    TRAINING PERIOD:  {TRAINING_START_YEAR}-{TRAINING_END_YEAR}
    CURRENT SEASON:   {CURRENT_SEASON} (Week {CURRENT_WEEK})

    nfelo FILES:      {len([f for f in Path(CURRENT_SEASON_DIR).glob('nfelo_*.csv')])} found
    Substack FILES:   {len([f for f in Path(CURRENT_SEASON_DIR).glob('substack_*.csv')])} found

    HOME FIELD ADV:   {HOME_FIELD_ADVANTAGE} points
    MIN BET EDGE:     {MIN_BET_EDGE} points

    ═══════════════════════════════════════════════════════════
    """
    return summary
