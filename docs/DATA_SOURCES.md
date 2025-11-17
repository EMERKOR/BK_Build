# Ball Knower Data Sources

## Overview

This document describes the data files used by Ball Knower for NFL betting analytics. All data files are stored in `data/current_season/` and follow a standardized naming convention.

## Target Naming Convention

The Ball Knower project uses a **category-first** naming convention for data files:

```
{category}_{provider}_{season}_week_{week}.csv
```

### Examples

- `power_ratings_nfelo_2025_week_11.csv`
- `power_ratings_substack_2025_week_11.csv`
- `epa_tiers_nfelo_2025_week_11.csv`
- `qb_epa_substack_2025_week_11.csv`

This convention groups related data together (e.g., all power ratings files) and makes it easier to discover available data sources for a given category.

## Data Categories

### 1. Power Ratings

Team power ratings from various analytical sources.

**Files:**
- `power_ratings_nfelo_2025_week_11.csv` - nfelo team ratings with QB adjustments
- `power_ratings_substack_2025_week_11.csv` - Substack offensive/defensive/overall ratings

**Key Fields:**
- `team` - Standard NFL team abbreviation
- `nfelo` - nfelo power rating (1300-1700 scale)
- `Ovr.` - Substack overall rating

### 2. EPA Tiers

Expected Points Added (EPA) metrics for offense and defense.

**Files:**
- `epa_tiers_nfelo_2025_week_11.csv` - Offensive and defensive EPA per play

**Key Fields:**
- `team` - Standard NFL team abbreviation
- `EPA/Play` - Offensive EPA per play
- `EPA/Play Against` - Defensive EPA per play (allowed)

### 3. Strength of Schedule

Historical and projected opponent strength metrics.

**Files:**
- `strength_of_schedule_nfelo_2025_week_11.csv` - Season-to-date and remaining SOS

**Key Fields:**
- `team` - Standard NFL team abbreviation
- Various SOS metrics (past, future, weighted)

### 4. QB Rankings

Quarterback performance and ratings.

**Files:**
- `qb_rankings_nfelo_2025_week_11.csv` - nfelo QB rankings
- `qb_epa_substack_2025_week_11.csv` - Substack QB EPA metrics

**Key Fields:**
- `team` - Team abbreviation
- QB-specific metrics (EPA, completion %, rating, etc.)

### 5. Weekly Projections

Game-by-game point and win probability projections.

**Files:**
- `weekly_projections_ppg_substack_2025_week_11.csv` - Points per game projections
- `weekly_projections_elo_substack_2025_week_11.csv` - Elo-based game projections

**Key Fields:**
- `team` - Team abbreviation
- Projected points, win probabilities, spreads

### 6. Win Totals

Season-long win total projections.

**Files:**
- `win_totals_nfelo_2025_week_11.csv` - Projected season wins and playoff odds

**Key Fields:**
- `team` - Team abbreviation
- Projected wins, playoff probability, championship probability

### 7. Receiving Leaders

Top receivers by various metrics.

**Files:**
- `receiving_leaders_nfelo_2025_week_11.csv` - Reception statistics leaders

**Key Fields:**
- Player name, team, receptions, yards, touchdowns, etc.

## Data Providers

### nfelo
Source: https://nfelo.com/
- Power ratings with QB adjustments
- EPA metrics (offense/defense)
- Strength of schedule
- QB rankings
- Win totals and playoff odds
- Receiving statistics

### Substack
Source: Various Substack NFL analytics newsletters
- Alternative power ratings (Off/Def/Overall)
- QB EPA metrics
- Weekly game projections (PPG and Elo-based)

### Future Providers
Additional providers may be added:
- **PFF** - Pro Football Focus grades and analytics
- **ESPN** - ESPN FPI and power rankings
- **BK** - Ball Knower internal models

## Loading Data

### Using the Unified Loader (Recommended)

```python
from ball_knower.io import load_power_ratings, load_all_sources

# Load a specific category and provider
nfelo_ratings = load_power_ratings("nfelo", week=11, season=2025)

# Load all available data sources
data = load_all_sources(week=11, season=2025)

# Access merged team ratings
merged = data["merged_ratings"]  # One row per team with all key metrics
```

### Legacy Loading

The legacy `src/data_loader.py` module is still available but will be deprecated in future versions. New code should use `ball_knower.io.loaders` instead.

## Current Status (Week 11, 2025)

**Note:** The repository currently uses **provider-first** filenames from the legacy naming convention:

- `nfelo_power_ratings_2025_week_11.csv`
- `nfelo_epa_tiers_off_def_2025_week_11.csv`
- `nfelo_strength_of_schedule_2025_week_11.csv`
- `nfelo_qb_rankings_2025_week_11.csv`
- `nfelo_nfl_win_totals_2025_week_11 (1).csv`
- `nfelo_nfl_receiving_leaders_2025_week_11.csv`
- `substack_power_ratings_2025_week_11.csv`
- `substack_qb_epa_2025_week_11.csv`
- `substack_weekly_proj_elo_2025_week_11.csv`
- `substack_weekly_proj_ppg_2025_week_11.csv`

The new unified loader system (`ball_knower/io/loaders.py`) **automatically handles both naming conventions** via fallback logic:

1. First tries the new category-first pattern
2. If not found, falls back to the known legacy pattern
3. Issues a deprecation warning when using legacy filenames

This allows the new loader system to work with existing files while supporting the eventual migration to category-first naming.

## Migration Timeline

Files will be renamed to the category-first convention during Phase 4 of the data migration plan (see `DATA_MIGRATION_SUMMARY.md`). The unified loaders are designed to work seamlessly before, during, and after this transition.

## Team Abbreviations

All data uses standardized NFL team abbreviations from `nfl_data_py`:

```
ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB,
HOU, IND, JAX, KC, LAC, LAR, LV, MIA, MIN, NE, NO, NYG, NYJ,
PHI, PIT, SEA, SF, TB, TEN, WAS
```

Team names are automatically normalized using `src/team_mapping.py` to handle variations like:
- Full names ("Kansas City Chiefs" → "KC")
- Nicknames ("Chiefs" → "KC")
- Lowercase codes ("kan" → "KC")
- Alternative abbreviations ("GNB" → "GB")

## Data Update Frequency

Data files are typically updated:
- **During the season:** Weekly (Tuesday-Wednesday after game week completes)
- **Format:** CSV files with headers
- **Encoding:** UTF-8
- **Location:** `data/current_season/`

## Questions or Issues?

For questions about data sources, file formats, or the loader system, please open an issue on the repository or consult the migration summary at `docs/DATA_MIGRATION_SUMMARY.md`.
